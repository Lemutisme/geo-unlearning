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
# Retain Null Projector (parameter-space protection)
# =========================
class RetainNullProjector:
    def __init__(self, model, param_name_regex, k=8, use_adam_diag=True, ema=0.98):
        self.model = model
        self.k = k
        self.use_adam_diag = use_adam_diag
        self.ema = ema
        self.param_names = [
            n for n, p in model.named_parameters()
            if any(re.fullmatch(rx, n) for rx in param_name_regex)
        ]
        # basis per-parameter: dict[name] -> list[tensor_like_param] (u_j same shape as param)
        self.basis = {n: [] for n in self.param_names}
        # Adam-diag preconditioner EMA (optional)
        self.v_ema = {n: None for n in self.param_names}

    @torch.no_grad()
    def _precond(self, name, g):
        if not self.use_adam_diag:
            return g
        v = self.v_ema[name]
        if v is None:
            v = g.pow(2)
        else:
            v = self.ema * v + (1 - self.ema) * g.pow(2)
        self.v_ema[name] = v
        return g / (v.sqrt() + 1e-8)

    @torch.no_grad()
    def update_basis_with_retain(self, retain_loss):
        # retain_loss: scalar computed on a small retain batch (e.g., KL to ref)
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
            g = g.detach()
            g_tilde = self._precond(name, g)  # whiten for stability
            g_tilde = g_tilde.to(torch.bfloat16)
            # Gram-Schmidt against existing basis
            U = self.basis[name]
            for u in U:
                g_tilde -= (g_tilde * u).sum().div(u.pow(2).sum() + 1e-12) * u
            norm = g_tilde.norm()
            if norm > 1e-6 and len(U) < self.k:
                U.append(g_tilde / norm)

    @torch.no_grad()
    def _project_tensor(self, g, U, mode='perp', lambda_align=0.0):
        if not U:
            return g
        if mode == 'perp':
            if lambda_align > 0:
                for u in U:
                    u = u.to(g.dtype)
                    coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                    g = g - lambda_align * coef * u
            for u in U:
                u = u.to(g.dtype)
                coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                g = g - coef * u
            return g
        elif mode == 'tan':
            g_tan = torch.zeros_like(g)
            for u in U:
                u = u.to(g.dtype)
                coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                g_tan = g_tan + coef * u
            return g_tan
        else:
            raise ValueError(f"Unknown mode={mode}")

    @torch.no_grad()
    def project_current_grads(self, mode='perp', lambda_align=0.0):
        for name, p in self.model.named_parameters():
            if name not in self.param_names or p.grad is None:
                continue
            U = self.basis[name]
            if not U:
                continue
            p.grad.copy_( self._project_tensor(p.grad, U, mode=mode, lambda_align=lambda_align) )

class GeometricUnlearn(GradDiff):

    def __init__(self, *args, **kwargs):
        self.geometric_config = kwargs.pop('geometric_config')
        self.simnpo_config = kwargs.pop("simnpo_config")
        self.npo_config = kwargs.pop("npo_config")
        self.dpo_config = kwargs.pop("dpo_config", None)
        self.undial_config = kwargs.pop("undial_config", None)
        self.wga_config = kwargs.pop("wga_config", None)
        self.satimp = kwargs.pop("satimp_config", None)

        super().__init__(*args, **kwargs)
        print("self.retain_loss_type")

        # === Automatic extrapolation of the last K layers ===
        auto_k = getattr(self.geometric_config, "auto_last_k_layers", 5)
        if auto_k is not None:
            auto_regex = build_last_layers_regex(
                self.model,
                last_k=int(auto_k),
                include_lm_head=True,
                include_final_norm=True
            )
            self.geometric_config.trainable_params_regex = auto_regex
            print(f"[GU] auto_last_k_layers={auto_k}, regex={auto_regex}")

        # === (Optional) Freeze non-target parameters, save video memory ===
        if getattr(self.geometric_config, "freeze_others", False):
            pats = [re.compile(rx) for rx in self.geometric_config.trainable_params_regex]
            for n, p in self.model.named_parameters():
                keep = any(r.fullmatch(n) for r in pats)
                p.requires_grad_(keep)

        # Parameter-space projector
        self.null_proj = RetainNullProjector(
            self.model,
            param_name_regex=self.geometric_config.trainable_params_regex,
            k=getattr(self.geometric_config, "null_k", 8),
            use_adam_diag=False, ema=0.98
        )

        # Reference model
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    # -------- main loss --------
    def compute_loss(self, model, inputs, return_outputs=False):

        # 取 forget/retain inputs
        forget_inputs = inputs["forget"] if self.geometric_config.loss != "dpo" else inputs["forget"]["original"]
        retain_inputs = inputs.get("retain")

        # (A) Calculate forget base loss
        if self.geometric_config.loss == 'npo':
            forget_loss, f_out = compute_dpo_loss(
                model=model, ref_model=self.ref_model,
                win_inputs=None, lose_inputs=forget_inputs,
                beta=self.npo_config.beta,
            )
        elif self.geometric_config.loss == 'dpo':
            original_inputs = forget_inputs['original'] if isinstance(forget_inputs, dict) and "original" in forget_inputs else inputs["forget"]["original"]
            alternate_inputs = inputs["forget"]["alternate"]
            forget_loss, f_out = compute_dpo_loss(
                model=model, ref_model=self.ref_model,
                win_inputs=alternate_inputs, lose_inputs=original_inputs,
                beta=self.dpo_config.beta,
            )
        elif self.geometric_config.loss == 'undial':
            forget_loss, f_out = compute_undial_loss(
                model, self.ref_model, forget_inputs, self.undial_config.beta
            )
        elif self.geometric_config.loss == 'simnpo':
            forget_labels = forget_inputs["labels"]
            loss_mask = forget_labels != -100
            forget_loss, f_out = compute_batch_nll(model, forget_inputs)
            forget_loss = forget_loss / loss_mask.sum(-1) - self.simnpo_config.delta
            forget_loss = -F.logsigmoid(self.simnpo_config.beta * forget_loss).mean() * 2 / self.simnpo_config.beta
        elif self.geometric_config.loss == 'ceu':
            forget_loss, f_out = compute_batch_ceu(
                model, forget_inputs, ignore_first_n_answer_tokens=1,
            )
        elif self.geometric_config.loss == 'wga':
            forget_loss, f_out = compute_wga_loss(
                model=model, inputs=forget_inputs, beta=self.wga_config.beta
            )
        elif self.geometric_config.loss == 'satimp':
            forget_loss, f_out = compute_satimp_loss(
                model=model, inputs=forget_inputs,
                beta1=self.satimp.beta1, beta2=self.satimp.beta2
            )
        else:
            f_out = model(**forget_inputs)
            forget_loss = -f_out.loss

        # Update tangent bases with small batch of retain_inputs
        if retain_inputs is not None:
            self._last_retain_inputs = retain_inputs
            with torch.enable_grad():
                with torch.no_grad():
                    ref_logits = self.ref_model(**retain_inputs).logits
                logits = model(**retain_inputs).logits
                retain_kl = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean"
                )
                self.null_proj.update_basis_with_retain(retain_kl)

        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        log_dict = {
            "train/loss": loss.item(),
            f"train/{self.geometric_config.loss}": forget_loss.item(),
            "train/retain_loss": retain_loss.item(),
        }

        self.log(log_dict)

        return (loss, f_out) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        # 1) 先把现有 grads 投到法向（忘记项已在 loss.backward 里产生）
        self.null_proj.project_current_grads(
            mode='perp',
            lambda_align=getattr(self.geometric_config, "lambda_align", 0.0)
        )

        # 2) 注入切向 KL 梯度（让 KL 起作用，但只在切向）
        if getattr(self.geometric_config, "inject_retain_tangent", True) and hasattr(self, "_last_retain_inputs"):
            retain_inputs = self._last_retain_inputs
            # 关闭现有 grad 的累积影响，仅做 autograd.grad，不改现有 p.grad：
            params = [p for n,p in self.model.named_parameters() if n in self.null_proj.param_names and p.requires_grad]
            with torch.enable_grad():
                with torch.no_grad():
                    ref_logits = self.ref_model(**retain_inputs).logits
                logits = self.model(**retain_inputs).logits
                retain_kl = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean"
                )
            gR = torch.autograd.grad(retain_kl, params, retain_graph=False, allow_unused=True)

            # 将 gR 的切向分量加到现有 p.grad 上
            alpha = getattr(self, "alpha", 1.0)
            i = 0
            for name, p in self.model.named_parameters():
                if name not in self.null_proj.param_names or p.requires_grad is False:
                    continue
                gRi = gR[i]; i += 1
                if gRi is None:
                    continue
                U = self.null_proj.basis.get(name, [])
                if not U:
                    continue
                gRi_tan = self.null_proj._project_tensor(gRi, U, mode='tan')
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_( alpha * gRi_tan )

        return super().optimizer_step(*args, **kwargs)
