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
                    use_opt_state=True,  # 优先用优化器 exp_avg_sq，免重复存 v_ema
                    basis_dtype=torch.float16,  # U 存半精度，计算时上采样到 FP32
                    basis_update_every=1,      # 基更新频率（步）
                    residual_keep_thresh=1e-3  # 新向量加入阈值（白化坐标残差范数比）
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
                # 找到该参数的名字
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
        """低频更新 retain 基；用一次 retain-KL 的梯度构造/刷新 U（白化坐标）。"""
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
            # 用 local EMA 更新 v（若未绑定优化器）
            g_tilde = self._precond(name, g.detach(), update=True).to(torch.float32)
            # Gram-Schmidt in whitened coords (float32 accumulate for stability)
            U = self.basis[name]
            for u in U:
                uu = u.to(torch.float32)
                g_tilde -= (g_tilde * uu).sum().div(uu.pow(2).sum().clamp_min(1e-12)) * uu
            # 自适应纳入：残差/原范数比值门控 + 限秩
            norm = g_tilde.norm()
            if norm > 1e-12 and len(U) < self.k:
                # 相对残差阈值（减少近依赖向量）
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
            use_opt_state=True,               # 优先用优化器状态
            basis_dtype=torch.float16,        # U 半精度存储
            basis_update_every=getattr(geometric_config, "basis_update_every", 2),
            residual_keep_thresh=getattr(geometric_config, "residual_keep_thresh", 1e-3),
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # Cache only inputs; 不在这里更新基（避免重复前后向）
        forget_inputs = inputs["forget"] if self.geometric_config.loss != "dpo" else inputs["forget"]["original"]
        self._last_forget_inputs = inputs["forget"] if "forget" in inputs else forget_inputs
        retain_inputs = inputs.get("retain")
        self._last_retain_inputs = retain_inputs

        # ---- forget base loss（与你现有一致）----
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

        # retain loss（供总 loss 用；基更新挪到 optimizer_step）
        retain_loss = self.compute_retain_loss(model, retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, f_out) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        """
        Memory-optimized final grad:
            Let g_tot = p.grad = γ g_f + α g_r (from previous backward).
            Compute g_r once; recover g_f = (g_tot - α g_r) / γ.
            In whitened coords:
                g_f_sel = P_⊥ g_f + (sign-aware capped) P_U g_f,
                g_r_nor = P_U g_r.
            Overwrite p.grad = γ g_f_sel + α g_r_nor.
        """
        # 0) 尝试绑定 optimizer，复用 exp_avg_sq
        optimizer = kwargs.get('optimizer', None)
        if optimizer is None and len(args) > 0 and hasattr(args[0], 'state'):
            optimizer = args[0]
        if optimizer is not None and not self.null_proj._opt_bound:
            self.null_proj.bind_optimizer(optimizer)

        # 若无 retain/forget 缓存，退化为简单正交
        if self._last_retain_inputs is None or self._last_forget_inputs is None:
            self.null_proj.project_current_grads(mode='perp')
            return super().optimizer_step(*args, **kwargs)

        retain_inputs = self._last_retain_inputs
        forget_inputs = self._last_forget_inputs

        # 获取目标参数与 g_tot 快照（避免中途被覆盖）
        named_params = [(n, p) for n, p in self.model.named_parameters()
                        if n in self.null_proj.param_names and p.requires_grad]
        params = [p for _, p in named_params]
        gtot = [ (p.grad.detach().clone() if p.grad is not None else None) for _, p in named_params ]

        # 1) 只计算一次 g_r；同时用它来低频刷新基（减少一次重复前后向）
        with torch.enable_grad():
            with torch.no_grad():
                ref_logits = self.ref_model(**retain_inputs).logits
            logits_r = self.model(**retain_inputs).logits
            retain_kl = F.kl_div(
                F.log_softmax(logits_r, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="batchmean"
            )
            # 刷新基（白化坐标）——低频
            self.null_proj.maybe_update_basis_with_retain(retain_kl)
            gR = torch.autograd.grad(retain_kl, params, retain_graph=False, allow_unused=True)

        gamma = self.gamma
        alpha = self.alpha
        inv_gamma = (1.0 / gamma) if gamma != 0 else 0.0

        # 2) 逐参数构造最终梯度（流式，避免大临时张量）
        for (name, p), g_tot_i, gRi in zip(named_params, gtot, gR):
            if g_tot_i is None and gRi is None:
                continue

            U = self.null_proj.basis.get(name, [])

            # recover g_f = (g_tot - α g_r) / γ
            gFi = None
            if g_tot_i is not None:
                if gRi is not None:
                    gFi = (g_tot_i - alpha * gRi) * inv_gamma
                else:
                    gFi = g_tot_i * inv_gamma

            # whiten
            gF_t = self.null_proj._precond(name, gFi, update=False).to(torch.float32) if gFi is not None else None
            gR_t = self.null_proj._precond(name, gRi, update=False).to(torch.float32) if gRi is not None else None

            # retain 法向
            gR_nor_t = None
            if gR_t is not None:
                gR_nor_t = self.nullProjProjectTan(name= name, g_t = gR_t, U = U)  # helper 调用见下
            # forget：正交 +（符号感知）保留反向切向并限幅
            gF_sel_t = None
            if gF_t is not None:
                if U:
                    # perp
                    gF_perp_t = self.null_proj._project_tensor(gF_t, U, mode='perp')
                    if self.sign_selective and (gR_t is not None):
                        a_list, b_list = [], []
                        for u in U:
                            uu = u.to(torch.float32)
                            a_list.append((gF_t * uu).sum())
                            b_list.append((gR_t * uu).sum())
                        a = torch.stack(a_list)
                        b = torch.stack(b_list)
                        keep = (a * b) < (-self.sign_tau)
                        gF_tan_keep_t = torch.zeros_like(gF_t)
                        if keep.any():
                            idx = torch.nonzero(keep, as_tuple=False).flatten()
                            for j in idx.tolist():
                                gF_tan_keep_t = gF_tan_keep_t + a[j] * U[j].to(torch.float32)
                        # cap
                        perp_norm = gF_perp_t.norm().clamp_min(1e-12)
                        tan_norm = gF_tan_keep_t.norm()
                        if tan_norm > self.sign_cap_ratio * perp_norm:
                            gF_tan_keep_t = gF_tan_keep_t * (self.sign_cap_ratio * perp_norm / tan_norm)
                        gF_sel_t = gF_perp_t + gF_tan_keep_t
                    else:
                        gF_sel_t = gF_perp_t
                else:
                    gF_sel_t = gF_t

            # 反白化并写回最终梯度
            new_grad = None
            if gF_sel_t is not None:
                new_grad = gamma * self.null_proj._deprecond(name, gF_sel_t)
            if gR_nor_t is not None:
                add = alpha * self.null_proj._deprecond(name, gR_nor_t)
                new_grad = add if new_grad is None else (new_grad + add)

            if new_grad is not None:
                if p.grad is None or p.grad.shape != new_grad.shape:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(new_grad)

        return super().optimizer_step(*args, **kwargs)

    # 小 helper：tan 投影（用现有 _project_tensor 保持语义清晰）
    @torch.no_grad()
    def nullProjProjectTan(self, name, g_t, U):
        if not U:
            return g_t
        return self.null_proj._project_tensor(g_t, U, mode='tan')
