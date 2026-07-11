import logging
import re

import torch
import torch.nn.functional as F

from trainer.unlearn.ceu import compute_batch_ceu
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import (
    compute_batch_nll,
    compute_dpo_loss,
    compute_undial_loss,
    compute_wga_loss,
    compute_satimp_loss,
)


logger = logging.getLogger(__name__)


# =========================
# Utils: auto-select last K trainable layers
# =========================
def build_last_layers_regex(model, last_k=2, include_lm_head=True, include_final_norm=True):
    names = [n for n, _ in model.named_parameters()]

    # Collect all possible "layer index" naming
    idxs = []
    for n in names:
        for pat in [
            r"\.(?:layers|h|blocks)\.(\d+)\.",      # .layers.XX. / .h.XX. / .blocks.XX.
            r"\.decoder\.layers\.(\d+)\.",          # .decoder.layers.XX.
        ]:
            m = re.search(pat, n)
            if m:
                idxs.append(int(m.group(1)))
                break

    if not idxs:
        # If no hierarchy is recognized (e.g. only lm_head), only head is returned (if needed).
        regex = []
        if include_lm_head and any(n.startswith("lm_head.") for n in names):
            regex.append(r"lm_head\..*")
        # Compatible with common head names
        for head in ("embed_out", "output_projection"):
            if include_lm_head and any(n.startswith(head + ".") for n in names):
                regex.append(fr"{head}\..*")
        return regex

    max_idx = max(idxs)
    chosen = list(range(max(0, max_idx - last_k + 1), max_idx + 1))

    # Generate a fullmatch regular that "matches all strings"
    # (note the . * to accommodate different prefixes)
    regex = [rf".*\.(?:layers|h|blocks)\.{i}\..*" for i in chosen]
    if any("decoder.layers" in n for n in names):
        regex += [rf".*\.decoder\.layers\.{i}\..*" for i in chosen]

    # Optional: include final layer norm (different models have different names, 
    # so include a few common ones)
    if include_final_norm:
        regex += [r".*\.ln_f\..*", r".*\.final_layernorm\..*", r".*\.norm\..*"]

    # lm_head / Compatible with other head names
    if include_lm_head:
        if any(n.startswith("lm_head.") for n in names):
            regex.append(r"lm_head\..*")
        else:
            for head in ("embed_out", "output_projection"):
                if any(n.startswith(head + ".") for n in names):
                    regex.append(fr"{head}\..*")
                    break

    # de-emphasize
    regex = list(dict.fromkeys(regex))
    return regex


# =========================
# Retain Null Projector
# =========================
class RetainNullProjector:
    """
    Retain subspace per-parameter under Adam metric H=W^T W, memory-optimized.

    mode:
      - 'perp' : H-orthogonal complement (retain-orthogonal)
      - 'tan'  : projection onto U (retain subspace)
    """
    def __init__(self, model, param_name_regex, k=8, use_adam_diag=True, ema=0.98,
                    use_opt_state=True,
                    basis_dtype=torch.float16, 
                    basis_update_every=1, 
                    residual_keep_thresh=1e-3 
                ):
        self.model = model
        self.k = int(k)
        self.use_adam_diag = use_adam_diag
        self.ema = ema
        self.use_opt_state = use_opt_state
        self.basis_dtype = basis_dtype
        self.basis_update_every = int(basis_update_every)
        self.residual_keep_thresh = float(residual_keep_thresh)

        self.param_names = [
            n for n, p in model.named_parameters()
            if any(re.fullmatch(rx, n) for rx in param_name_regex)
        ]
        self.basis = {n: [] for n in self.param_names}  # list[Tensor(basis_dtype)] per param
        self.v_ema = {n: None for n in self.param_names}  # fallback when optimizer state not bound
        self._opt_bound = False
        self._exp_avg_sq = {}  # name -> Tensor, from optimizer state
        self._step = 0

    def bind_optimizer(self, optimizer):
        """Try to bind Adam exp_avg_sq to avoid duplicating second-moment memory."""
        name_map = dict(self.model.named_parameters())
        found = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p is None:
                    continue
                for n, pp in name_map.items():
                    if pp is p and n in self.param_names:
                        st = optimizer.state.get(p, {})
                        v = st.get('exp_avg_sq', None)
                        if v is not None:
                            self._exp_avg_sq[n] = v
                            found += 1
                        break
        self._opt_bound = (found > 0)

    @torch.no_grad()
    def _get_v(self, name, g=None, update=False):
        """Return second moment v for whitening: prefer optimizer exp_avg_sq; else local EMA."""
        if self.use_opt_state and self._opt_bound and (name in self._exp_avg_sq):
            return self._exp_avg_sq[name]
        if not self.use_adam_diag:
            return None
        # fallback: local EMA
        if update:
            v = self.v_ema[name]
            v = g.pow(2) if v is None else self.ema * v + (1 - self.ema) * g.pow(2)
            self.v_ema[name] = v
            return v
        else:
            return self.v_ema[name]

    @torch.no_grad()
    def _precond(self, name, g, update=False):
        """Whiten: g~ = W g with W = 1/sqrt(v+eps)."""
        if not self.use_adam_diag:
            return g
        v = self._get_v(name, g=g, update=update)
        if v is None:
            return g
        return g / (v.sqrt() + 1e-8)

    @torch.no_grad()
    def _deprecond(self, name, g_tilde):
        """Map back: g = W^{-1} g~."""
        if not self.use_adam_diag:
            return g_tilde
        v = self._get_v(name, update=False)
        if v is None:
            return g_tilde
        return g_tilde * (v.sqrt() + 1e-8)

    @torch.no_grad()
    def maybe_update_basis_with_retain(self, retain_loss):
        self._step += 1
        if (self._step - 1) % self.basis_update_every != 0:
            return
        params = [
            p for n, p in self.model.named_parameters()
            if n in self.param_names and p.requires_grad
        ]
        grads = torch.autograd.grad(retain_loss, params, retain_graph=False, allow_unused=True)
        for (name, p), g in zip(
            [(n, p) for n, p in self.model.named_parameters() if n in self.param_names], grads
        ):
            if g is None:
                continue
            g_tilde = self._precond(name, g.detach(), update=True).to(torch.float32)
            # Gram-Schmidt in whitened coords (float32 accumulate for stability)
            U = self.basis[name]
            for u in U:
                uu = u.to(torch.float32)
                g_tilde -= (g_tilde * uu).sum().div(uu.pow(2).sum().clamp_min(1e-12)) * uu
            norm = g_tilde.norm()
            if norm > 1e-12 and len(U) < self.k:
                rel = (norm / (g.detach().norm() + 1e-12)).item()
                if rel >= self.residual_keep_thresh:
                    U.append((g_tilde / norm).to(self.basis_dtype))

    @torch.no_grad()
    def _project_tensor(self, g_tilde, U, mode='perp'):
        """Input/output: whitened coords; U stored in basis_dtype, upcast to float32 for math."""
        if not U:
            return g_tilde
        if mode == 'perp':
            for u in U:
                uu = u.to(torch.float32)
                coef = (g_tilde * uu).sum().div(uu.pow(2).sum().clamp_min(1e-12))
                g_tilde = g_tilde - coef * uu
            return g_tilde
        elif mode == 'tan':
            g_tan = torch.zeros_like(g_tilde, dtype=torch.float32)
            for u in U:
                uu = u.to(torch.float32)
                coef = (g_tilde * uu).sum().div(uu.pow(2).sum().clamp_min(1e-12))
                g_tan = g_tan + coef * uu
            return g_tan
        else:
            raise ValueError(f"Unknown mode={mode}")


class GeometricUnlearn(GradDiff):

    @staticmethod
    def _global_dot(left, right):
        result = None
        for name in left.keys() & right.keys():
            value = (left[name].float() * right[name].float()).sum()
            result = value if result is None else result + value

        if result is None:
            raise RuntimeError("No common tensors for global dot product.")

        return result

    @staticmethod
    def _to_adam_coordinates(gradient, sqrt_h):
        if sqrt_h is None:
            return gradient.float()
        return gradient.float() / sqrt_h

    @staticmethod
    def _from_adam_coordinates(vector, sqrt_h):
        if sqrt_h is None:
            return vector.float()
        return vector.float() * sqrt_h

    @classmethod
    def _project_rank_one(cls, forget, retain, eps):
        denominator = cls._global_dot(retain, retain).clamp_min(eps)
        coefficient = cls._global_dot(forget, retain) / denominator
        projected = {
            name: (
                tensor.float() - coefficient * retain[name].float()
                if name in retain
                else tensor.float().clone()
            )
            for name, tensor in forget.items()
        }
        return projected, coefficient

    def __init__(self, *args, **kwargs):
        self.geometric_config = kwargs.pop("geometric_config")
        self.simnpo_config = kwargs.pop("simnpo_config", None)
        self.npo_config = kwargs.pop("npo_config", None)
        self.dpo_config = kwargs.pop("dpo_config", None)
        self.undial_config = kwargs.pop("undial_config", None)
        self.wga_config = kwargs.pop("wga_config", None)
        self.satimp_config = kwargs.pop("satimp_config", None)

        self.loss_name = str(self.geometric_config.loss).lower()
        config_map = {
            "npo": self.npo_config,
            "simnpo": self.simnpo_config,
            "dpo": self.dpo_config,
            "undial": self.undial_config,
            "wga": self.wga_config,
            "satimp": self.satimp_config,
        }
        method_config = config_map.get(self.loss_name)
        if method_config is not None:
            kwargs["gamma"] = float(method_config.gamma)
            kwargs["alpha"] = float(method_config.alpha)
            kwargs["retain_loss_type"] = str(method_config.retain_loss_type)

        super().__init__(*args, **kwargs)

        if self.gamma <= 0:
            raise ValueError("GU requires gamma > 0.")
        if self.alpha < 0:
            raise ValueError("GU requires alpha >= 0.")

        if self.ref_model is None and self.loss_name in {"npo", "dpo", "undial"}:
            self.ref_model = self._prepare_ref_model(self.model)

        self.gu_enabled = bool(getattr(self.geometric_config, "gu_enabled", True))
        self.projection_eps = float(
            getattr(self.geometric_config, "projection_eps", 1e-12)
        )
        self.trainable_params_regex = list(
            getattr(self.geometric_config, "trainable_params_regex", [".*"])
        )
        self._init_gu_buffers()
        self.gu_projection_calls = 0
        self.last_gu_diagnostics = {}

    def _selected_named_parameters(self, model=None):
        model = self.model if model is None else model
        return [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and any(
                re.fullmatch(pattern, name)
                for pattern in self.trainable_params_regex
            )
        ]

    def _init_gu_buffers(self):
        self._gu_forget_buffer = {}
        self._gu_retain_buffer = {}

    @torch.no_grad()
    def _accumulate_component_grads(
        self,
        named_params,
        forget_grads,
        retain_grads,
    ):
        for (name, _), forget_grad, retain_grad in zip(
            named_params,
            forget_grads,
            retain_grads,
        ):
            if forget_grad is not None:
                forget_grad = forget_grad.detach().float()
                if name not in self._gu_forget_buffer:
                    self._gu_forget_buffer[name] = torch.zeros_like(
                        forget_grad,
                        dtype=torch.float32,
                    )
                self._gu_forget_buffer[name].add_(forget_grad)

            if retain_grad is not None:
                retain_grad = retain_grad.detach().float()
                if name not in self._gu_retain_buffer:
                    self._gu_retain_buffer[name] = torch.zeros_like(
                        retain_grad,
                        dtype=torch.float32,
                    )
                self._gu_retain_buffer[name].add_(retain_grad)

    def _clear_gu_buffers(self):
        self._gu_forget_buffer.clear()
        self._gu_retain_buffer.clear()

    def _unwrap_optimizer(self):
        optimizer = self.optimizer
        visited = set()
        while hasattr(optimizer, "optimizer") and id(optimizer) not in visited:
            visited.add(id(optimizer))
            optimizer = optimizer.optimizer
        return optimizer

    @staticmethod
    def _optimizer_groups_by_parameter(optimizer):
        return {
            id(parameter): group
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    @staticmethod
    def _optimizer_step_value(step):
        if isinstance(step, torch.Tensor):
            return int(step.item())
        return int(step)

    def _frozen_sqrt_denominator(self, optimizer, parameter, group):
        state = optimizer.state.get(parameter, {})
        exp_avg_sq = state.get("exp_avg_sq")
        step = self._optimizer_step_value(state.get("step", 0))
        if exp_avg_sq is None or step <= 0:
            return None

        beta2 = float(group["betas"][1])
        bias_correction = 1.0 - beta2**step
        v_hat = exp_avg_sq.float() / bias_correction
        return (v_hat.sqrt() + float(group["eps"])).sqrt()

    def _to_frozen_adam_coordinates(
        self,
        buffer,
        named_params,
        optimizer,
        groups_by_parameter,
    ):
        transformed = {}
        identity_fallback_names = set()
        for name, parameter in named_params:
            tensor = buffer.get(name)
            if tensor is None:
                continue
            sqrt_h = self._frozen_sqrt_denominator(
                optimizer,
                parameter,
                groups_by_parameter[id(parameter)],
            )
            if sqrt_h is None:
                identity_fallback_names.add(name)
            transformed[name] = self._to_adam_coordinates(tensor, sqrt_h)
        return transformed, identity_fallback_names

    @torch.no_grad()
    def _finalize_gu_gradients(self, named_params):
        if not self._gu_forget_buffer:
            raise RuntimeError("Forget gradient buffer is empty.")
        if not self._gu_retain_buffer:
            raise RuntimeError("Retain gradient buffer is empty.")

        optimizer = self._unwrap_optimizer()
        groups_by_parameter = self._optimizer_groups_by_parameter(optimizer)
        transformed_forget, forget_fallbacks = self._to_frozen_adam_coordinates(
            self._gu_forget_buffer,
            named_params,
            optimizer,
            groups_by_parameter,
        )
        transformed_retain, retain_fallbacks = self._to_frozen_adam_coordinates(
            self._gu_retain_buffer,
            named_params,
            optimizer,
            groups_by_parameter,
        )
        dot_before = self._global_dot(transformed_forget, transformed_retain)
        projected_forget, coefficient = self._project_rank_one(
            transformed_forget,
            transformed_retain,
            self.projection_eps,
        )
        dot_after = self._global_dot(projected_forget, transformed_retain)
        forget_norm = self._global_dot(
            transformed_forget,
            transformed_forget,
        ).sqrt()
        retain_norm = self._global_dot(
            transformed_retain,
            transformed_retain,
        ).sqrt()
        projected_norm = self._global_dot(
            projected_forget,
            projected_forget,
        ).sqrt()

        for name, parameter in named_params:
            forget_component = projected_forget.get(name)
            retain_component = transformed_retain.get(name)
            final_coordinates = None
            if forget_component is not None:
                final_coordinates = self.gamma * forget_component
            if retain_component is not None and self.alpha != 0:
                retain_term = self.alpha * retain_component
                final_coordinates = (
                    retain_term
                    if final_coordinates is None
                    else final_coordinates + retain_term
                )
            if final_coordinates is None:
                continue

            sqrt_h = self._frozen_sqrt_denominator(
                optimizer,
                parameter,
                groups_by_parameter[id(parameter)],
            )
            final_gradient = self._from_adam_coordinates(
                final_coordinates,
                sqrt_h,
            )
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.copy_(
                final_gradient.to(
                    device=parameter.grad.device,
                    dtype=parameter.grad.dtype,
                )
            )

        relative_residual = dot_after.abs() / (
            projected_norm * retain_norm + self.projection_eps
        )
        self.gu_projection_calls += 1
        self.last_gu_diagnostics = {
            "mode": "approximate_adam_stage_a",
            "projection_calls": self.gu_projection_calls,
            "coefficient": float(coefficient.item()),
            "forget_norm": float(forget_norm.item()),
            "retain_norm": float(retain_norm.item()),
            "dot_before": float(dot_before.item()),
            "dot_after": float(dot_after.item()),
            "relative_orthogonality_residual": float(
                relative_residual.item()
            ),
            "identity_fallback_parameters": len(
                forget_fallbacks | retain_fallbacks
            ),
        }
        logger.info(
            "GU projection step=%d mode=%s coefficient=%.8e residual=%.8e "
            "identity_fallback_parameters=%d",
            self.gu_projection_calls,
            self.last_gu_diagnostics["mode"],
            self.last_gu_diagnostics["coefficient"],
            self.last_gu_diagnostics["relative_orthogonality_residual"],
            self.last_gu_diagnostics["identity_fallback_parameters"],
        )
        self._clear_gu_buffers()

    def training_step(self, model, inputs):
        if not self.gu_enabled:
            return super().training_step(model, inputs)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            forget_loss, retain_loss, _ = self.compute_component_losses(
                model,
                inputs,
            )
            total_loss = self.gamma * forget_loss + self.alpha * retain_loss

        if self.args.n_gpu > 1:
            forget_loss = forget_loss.mean()
            retain_loss = retain_loss.mean()
            total_loss = total_loss.mean()

        named_params = self._selected_named_parameters(model)
        params = [parameter for _, parameter in named_params]
        scale = 1.0 / self.args.gradient_accumulation_steps
        forget_grads = torch.autograd.grad(
            forget_loss * scale,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        retain_grads = torch.autograd.grad(
            retain_loss * scale,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        self._accumulate_component_grads(
            named_params,
            forget_grads,
            retain_grads,
        )

        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self._finalize_gu_gradients(named_params)

        return total_loss.detach() * scale

    def compute_forget_loss(self, model, forget_inputs):
        if self.loss_name == "npo":
            return compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.npo_config.beta,
            )
        if self.loss_name == "dpo":
            return compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=forget_inputs["alternate"],
                lose_inputs=forget_inputs["original"],
                beta=self.dpo_config.beta,
            )
        if self.loss_name == "undial":
            return compute_undial_loss(
                model,
                self.ref_model,
                forget_inputs,
                self.undial_config.beta,
            )
        if self.loss_name == "simnpo":
            forget_labels = forget_inputs["labels"]
            loss_mask = forget_labels != -100
            forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
            forget_loss = forget_loss / loss_mask.sum(-1) - self.simnpo_config.delta
            beta = self.simnpo_config.beta
            forget_loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta
            return forget_loss, forget_outputs
        if self.loss_name == "ceu":
            return compute_batch_ceu(
                model,
                forget_inputs,
                ignore_first_n_answer_tokens=1,
            )
        if self.loss_name == "wga":
            return compute_wga_loss(
                model=model,
                inputs=forget_inputs,
                beta=self.wga_config.beta,
            )
        if self.loss_name == "satimp":
            return compute_satimp_loss(
                model=model,
                inputs=forget_inputs,
                beta1=self.satimp_config.beta1,
                beta2=self.satimp_config.beta2,
            )

        forget_outputs = model(**forget_inputs)
        return -forget_outputs.loss, forget_outputs

    def compute_component_losses(self, model, inputs):
        forget_loss, forget_outputs = self.compute_forget_loss(
            model,
            inputs["forget"],
        )
        retain_loss = self.compute_retain_loss(model, inputs["retain"])
        return forget_loss, retain_loss, forget_outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_loss, retain_loss, forget_outputs = self.compute_component_losses(
            model,
            inputs,
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
