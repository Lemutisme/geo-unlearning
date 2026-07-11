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
        self.geometric_config = kwargs.pop('geometric_config')
        self.simnpo_config = kwargs.pop("simnpo_config")
        self.npo_config = kwargs.pop("npo_config")
        self.dpo_config = kwargs.pop("dpo_config", None)
        self.undial_config = kwargs.pop("undial_config", None)
        self.wga_config = kwargs.pop("wga_config", None)
        self.satimp = kwargs.pop("satimp_config", None)
        super().__init__(*args, **kwargs)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self._setup_projector(geometric_config=self.geometric_config)

        self._last_forget_inputs = None
        self._last_retain_inputs = None

        # sign-aware selective projection
        self.sign_selective = getattr(self.geometric_config, "sign_selective", True)
        self.sign_tau = float(getattr(self.geometric_config, "sign_tau", 0.0))
        self.sign_cap_ratio = float(getattr(self.geometric_config, "sign_cap_ratio", 0.5))

    def _setup_projector(self, geometric_config):
        auto_regex = build_last_layers_regex(
            self.model,
            last_k=int(geometric_config.auto_last_k_layers),
            include_lm_head=True,
            include_final_norm=True
        )
        self.null_proj = RetainNullProjector(
            self.model,
            param_name_regex=list(dict.fromkeys(auto_regex)),
            k=geometric_config.null_k,
            use_adam_diag=True,
            use_opt_state=True,
            basis_dtype=torch.float16,
            basis_update_every=getattr(geometric_config, "basis_update_every", 2),
            residual_keep_thresh=getattr(geometric_config, "residual_keep_thresh", 1e-3),
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"] if self.geometric_config.loss != "dpo" else inputs["forget"]["original"]
        self._last_forget_inputs = inputs["forget"] if "forget" in inputs else forget_inputs
        retain_inputs = inputs.get("retain")
        self._last_retain_inputs = retain_inputs

        lt = self.geometric_config.loss
        if lt == 'npo':
            forget_loss, f_out = compute_dpo_loss(
                model=model, ref_model=self.ref_model,
                win_inputs=None, lose_inputs=forget_inputs,
                beta=self.npo_config.beta,
            )
        elif lt == 'dpo':
            original_inputs = forget_inputs['original'] if isinstance(forget_inputs, dict) and "original" in forget_inputs else inputs["forget"]["original"]
            alternate_inputs = inputs["forget"]["alternate"]
            forget_loss, f_out = compute_dpo_loss(
                model=model, ref_model=self.ref_model,
                win_inputs=alternate_inputs, lose_inputs=original_inputs,
                beta=self.dpo_config.beta,
            )
        elif lt == 'undial':
            forget_loss, f_out = compute_undial_loss(model, self.ref_model, forget_inputs, self.undial_config.beta)
        elif lt == 'simnpo':
            forget_labels = forget_inputs["labels"]
            loss_mask = forget_labels != -100
            forget_loss, f_out = compute_batch_nll(model, forget_inputs)
            forget_loss = forget_loss / loss_mask.sum(-1) - self.simnpo_config.delta
            forget_loss = -F.logsigmoid(self.simnpo_config.beta * forget_loss).mean() * 2 / self.simnpo_config.beta
            self.gamma = self.simnpo_config.gamma
        elif lt == 'ceu':
            forget_loss, f_out = compute_batch_ceu(model, forget_inputs, ignore_first_n_answer_tokens=1)
            self.alpha = 0.0
        elif lt == 'wga':
            forget_loss, f_out = compute_wga_loss(model=model, inputs=forget_inputs, beta=self.wga_config.beta)
        elif lt == 'satimp':
            forget_loss, f_out = compute_satimp_loss(model=model, inputs=forget_inputs, beta1=self.satimp.beta1, beta2=self.satimp.beta2)
            self.gamma = self.satimp.gamma
        else:
            f_out = model(**forget_inputs)
            forget_loss = -f_out.loss
            if lt == 'gradacend':
                self.alpha = 0.0

        retain_loss = self.compute_retain_loss(model, retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, f_out) if return_outputs else loss
