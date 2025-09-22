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

    # 收集所有可能的 "层索引" 命名
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
        # 没识别到分层结构（例如只有 lm_head），就只返回 head（若需要）
        regex = []
        if include_lm_head and any(n.startswith("lm_head.") for n in names):
            regex.append(r"lm_head\..*")
        # 兼容常见 head 名
        for head in ("embed_out", "output_projection"):
            if include_lm_head and any(n.startswith(head + ".") for n in names):
                regex.append(fr"{head}\..*")
        return regex

    max_idx = max(idxs)
    chosen = list(range(max(0, max_idx - last_k + 1), max_idx + 1))

    # 生成“匹配全部字符串”的 fullmatch 正则（注意加了前后的 .* 以适配不同前缀）
    regex = [rf".*\.(?:layers|h|blocks)\.{i}\..*" for i in chosen]
    if any("decoder.layers" in n for n in names):
        regex += [rf".*\.decoder\.layers\.{i}\..*" for i in chosen]

    # 可选：包含最后的 layer norm（不同模型命名不同，放几个常见的）
    if include_final_norm:
        regex += [r".*\.ln_f\..*", r".*\.final_layernorm\..*", r".*\.norm\..*"]

    # lm_head / 兼容其他 head 名
    if include_lm_head:
        if any(n.startswith("lm_head.") for n in names):
            regex.append(r"lm_head\..*")
        else:
            for head in ("embed_out", "output_projection"):
                if any(n.startswith(head + ".") for n in names):
                    regex.append(fr"{head}\..*")
                    break

    # 去重
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


# =========================
# Representation geometry (EMA whitening, anchor, costs)
# =========================
class OnlineDistanceBuilder:
    """
    Online distance builder adapted for multi-GPU training.
    Provides:
      - EMA whitening (mean/var),
      - retain anchor,
      - cosine to anchor as distance,
      - pairwise cosine cost (median-normalized later).
    """
    def __init__(self, hidden_size: int, momentum: float = 0.99, device=None,
                 dtype=torch.float32, freeze_anchor: bool = True):
        self.initial_device = device or torch.device("cpu")
        self.dtype = dtype
        H = hidden_size
        self.m = momentum
        self._ema_mean = torch.zeros(H, device=self.initial_device, dtype=self.dtype)
        self._ema_var  = torch.ones(H,  device=self.initial_device, dtype=self.dtype)
        self._anchor = None
        self._eps = 1e-8
        self._freeze_anchor = freeze_anchor

    @torch.no_grad()
    def update_whitener(self, E: torch.Tensor):
        current_device = E.device
        # keep EMA tensors on correct device
        self._ema_mean = self._ema_mean.to(current_device)
        self._ema_var = self._ema_var.to(current_device)
        bmean = E.mean(dim=0)
        bvar  = E.var(dim=0, unbiased=False)
        self._ema_mean = self.m * self._ema_mean + (1 - self.m) * bmean
        self._ema_var  = self.m * self._ema_var  + (1 - self.m) * bvar

    def whiten(self, E: torch.Tensor) -> torch.Tensor:
        current_device = E.device
        ema_mean = self._ema_mean.to(current_device)
        ema_var = self._ema_var.to(current_device)
        return (E - ema_mean) / (ema_var + self._eps).sqrt()

    @torch.no_grad()
    def update_retain_anchor(self, E_retain_white):
        current_device = E_retain_white.device
        rmean = E_retain_white.mean(dim=0)
        if self._anchor is None:
            self._anchor = rmean.clone().to(current_device)
        elif not self._freeze_anchor:
            self._anchor = self._anchor.to(current_device)
            self._anchor = self.m * self._anchor + (1 - self.m) * rmean

    def dvec_to_anchor(self, E_forget_white):
        # cosine distance to anchor: 1 - cos
        current_device = E_forget_white.device
        anchor = self._anchor.to(current_device)
        num = (E_forget_white * anchor).sum(dim=1)
        den = E_forget_white.norm(dim=1).clamp_min(1e-8) * anchor.norm().clamp_min(1e-8)
        return 1 - (num / den)

    def pairwise_cost(self, E_white):
        X = F.normalize(E_white, dim=1)
        C = 1 - X @ X.t()
        C = C.clamp(min=0)
        C = C - torch.diag_embed(torch.diagonal(C))
        return C


# =========================
# Main class with improved OT slope
# =========================
class GeometricUnlearn(GradDiff):
    """
    Geometric Unlearning with:
      - representation disentanglement map (EMA whiten + anchor),
      - improved OT slope penalty (cost-only neighborhoods),
      - parameter-space retain-orthogonal projection.
    """
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

        # === 自动推断最后 K 层 ===
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

        # ===（可选）冻结非目标参数，省显存 ===
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

        # Representation geometry
        H = self.model.config.hidden_size
        self.fastdist = OnlineDistanceBuilder(
            hidden_size=H,
            momentum=getattr(self.geometric_config, "ema_momentum", 0.99),
            device=self.model.device,
            dtype=torch.float32,
        )

    # -------- helpers for OT slope --------
    def _hidden_embed(self, outputs, labels):
        """
        Pool last-layer hidden states on answer tokens (mean over answer positions).
        Returns [B, D].
        """
        H = outputs.hidden_states[-1]                # [B, S, D]
        mask = (labels != -100).unsqueeze(-1)        # [B, S, 1], bool
        Hw = H.masked_fill(~mask, 0.0)               # zero out non-answer tokens
        # keep a singleton dim so it broadcasts over D:
        cnt = mask.sum(dim=1, keepdim=True).clamp_min(1).to(Hw.dtype)   # [B, 1, 1]
        cnt = cnt.squeeze(-1)                        # [B, 1]
        return Hw.sum(dim=1) / cnt                   # [B, D] / [B, 1] -> [B, D]


    def _normalize_cost(self, C: torch.Tensor) -> torch.Tensor:
        Cn = C.clone()
        pos = Cn[(Cn > 0) & torch.isfinite(Cn)]
        if pos.numel() > 0:
            med = pos.median()
            if med > 0:
                Cn = Cn / med
        return Cn

    def _row_softmax_offdiag(self, A: torch.Tensor, temp: float = 1.0, topk: int = 8):
        """
        按行 softmax，mask 对角，支持 top-k 邻域截断；A 为 '负代价' 或相似度矩阵。
        """
        B = A.shape[0]
        big_neg = torch.finfo(A.dtype).min / 4
        M = A.clone()
        M.fill_diagonal_(big_neg)
        if topk is not None and topk < B - 1:
            vals, idx = torch.topk(M, k=topk, dim=1)
            keep = torch.zeros_like(M, dtype=torch.bool)
            keep.scatter_(1, idx, True)
            M = torch.where(keep, M, big_neg)
        pi = torch.softmax(M / max(temp, 1e-6), dim=1)
        return pi

    def _cal_distance_n_cost(self, per_sample_loss: torch.Tensor, E_f_white: torch.Tensor):
        """
        计算 d_hat, C，并自适应 μ（使 μ·median(d) ≈ β·median(per-sample loss)）。
        """
        d_hat = self.fastdist.dvec_to_anchor(E_f_white)  # [B]
        C = self.fastdist.pairwise_cost(E_f_white)       # [B, B]

        with torch.no_grad():
            ce_med = per_sample_loss.median()
            d_med  = (d_hat + 1e-6).median()
            beta = getattr(self.geometric_config, "mu_auto_coeff", 0.5)
            mu_max = getattr(self.geometric_config, "mu_max", 0.1)
            mu_eff = (beta * ce_med / d_med).clamp(1e-3, mu_max)

        return d_hat, C, mu_eff

    def _per_sample_loss(self, model, forget_inputs, f_out):
        """
        仅使用你已有的工具/模型，构造每样本损失:
        - 函数内 per_sample_loss 定义为：越大=越想忘
        - 末尾统一返回其相反数（配合你外层的最小化）
        """
        loss_name = self.geometric_config.loss

        # ---- 常用辅助 ----
        def _valid_mask(labels_shifted):
            return (labels_shifted != -100)

        def _seq_avg_from_token_values(tok_vals, labels_shifted):
            valid = _valid_mask(labels_shifted)
            return (tok_vals * valid).sum(-1) / valid.sum(-1).clamp_min(1)

        # 统一对齐：shift one token（和 compute_batch_nll/CE 一致）
        logits = f_out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = forget_inputs["labels"][..., 1:].contiguous()

        # ---------- 1) 纯 CE ----------
        if loss_name == "ce":
            # token-CE（reduction='none'）后按样本平均
            ce_tok = F.cross_entropy(
                shift_logits.permute(0, 2, 1),  # [B,V,S-1]
                shift_labels,
                reduction="none",
                ignore_index=-100
            )  # [B, S-1]
            per_sample_loss = _seq_avg_from_token_values(ce_tok, shift_labels)

        # ---------- 2) CEU（排除真标签的软目标；仅用已有 logits 构造） ----------
        elif loss_name == "ceu":
            # 可选忽略前 n 个答案 token（和你 pipeline 对齐）
            n = getattr(self.geometric_config, "ignore_first_n_answer_tokens", 1)
            labels = forget_inputs["labels"].clone()
            if n and n > 0:
                valid = (labels != -100)
                head_mask = (valid.cumsum(dim=-1) <= n) & valid
                labels[head_mask] = -100
            shift_labels = labels[..., 1:].contiguous()
            valid = _valid_mask(shift_labels)

            # soft 目标：把真标签 logit = -inf，softmax 得到“排除真标签”的均匀分布
            with torch.no_grad():
                target_logits = shift_logits.detach().clone()
                target_logits.scatter_(dim=-1, index=shift_labels.unsqueeze(-1), value=float("-inf"))
                q = F.softmax(target_logits, dim=-1)  # [B,S-1,V]

            log_p = F.log_softmax(shift_logits, dim=-1)
            soft_ce_tok = -(q * log_p).sum(-1)  # [B,S-1]
            per_sample_loss = _seq_avg_from_token_values(soft_ce_tok, shift_labels)

        # ---------- 3) SimNPO（只用 compute_batch_nll） ----------
        elif loss_name == "simnpo":
            ce_sum, _ = compute_batch_nll(model, forget_inputs)  # [B] 为 sum over (S-1)
            denom = _valid_mask(shift_labels).sum(-1).clamp_min(1)
            ce_avg = ce_sum / denom
            per_sample_loss = F.logsigmoid(self.simnpo_config.beta * (ce_avg - self.simnpo_config.delta)) * (2.0 / self.simnpo_config.beta)

        # ---------- 4) NPO（CE - CE_ref；只用 compute_batch_nll） ----------
        elif loss_name == "npo":
            ce_m, _ = compute_batch_nll(model, forget_inputs)      # [B]
            with torch.no_grad():
                ce_r, _ = compute_batch_nll(self.ref_model, forget_inputs)  # [B]
            # 注意方向：用 (CE - CE_ref)
            per_sample_loss = F.logsigmoid(self.npo_config.beta * (ce_m - ce_r)) * (2.0 / self.npo_config.beta)

        # ---------- 5) DPO（win/lose 的 log 比率；只用 compute_batch_nll） ----------
        elif loss_name == "dpo":
            original_inputs = forget_inputs["original"]
            alternate_inputs = forget_inputs["alternate"]
            # NLL（sum over (S-1)）
            lose_m, _ = compute_batch_nll(model, original_inputs)
            win_m,  _ = compute_batch_nll(model, alternate_inputs)
            with torch.no_grad():
                lose_r, _ = compute_batch_nll(self.ref_model, original_inputs)
                win_r,  _ = compute_batch_nll(self.ref_model, alternate_inputs)
            # log 比率（注意这里是 -NLL 的差）
            win_log_ratio  = -(win_m  - win_r)
            lose_log_ratio = -(lose_m - lose_r)
            per_sample_loss = F.logsigmoid(self.dpo_config.beta * (win_log_ratio - lose_log_ratio)) * (2.0 / self.dpo_config.beta)

        # ---------- 6) UNDIAL（教师分布来自 ref logits − β·onehot(y)） ----------
        elif loss_name == "undial":
            with torch.no_grad():
                ref_logits = self.ref_model(**forget_inputs).logits
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()

            # 构造教师 q
            with torch.no_grad():
                mask = torch.zeros_like(shift_ref_logits)
                b = torch.arange(mask.size(0), device=mask.device).view(-1,1,1)
                t = torch.arange(mask.size(1), device=mask.device).view(1,-1,1)
                mask[b, t, shift_labels.unsqueeze(-1)] = 1.0
                teacher = F.softmax(shift_ref_logits - self.undial_config.beta * mask, dim=-1)  # [B,S-1,V]

            log_p = F.log_softmax(shift_logits, dim=-1)
            tok_kl = F.kl_div(log_p, teacher, reduction="none", log_target=False).sum(-1)  # [B,S-1]
            per_sample_loss = _seq_avg_from_token_values(tok_kl, shift_labels)

        # ---------- 7) WGA（仅用现有 logits + ref logits 的“指导式 KL” proxy） ----------
        elif loss_name == "wga":
            # 若 compute_wga_loss 只返回 batch 标量，这里给 per-sample 等价式（无需改 wga.py）
            with torch.no_grad():
                ref_logits = self.ref_model(**forget_inputs).logits
            # 令教师分布为带温度/偏置的 ref 分布（温度/偏置用已有 beta）
            teacher = F.softmax(ref_logits[..., :-1, :].contiguous() / max(self.wga_config.beta, 1e-6), dim=-1)
            log_p = F.log_softmax(shift_logits, dim=-1)
            tok_kl = F.kl_div(log_p, teacher, reduction="none", log_target=False).sum(-1)
            per_sample_loss = _seq_avg_from_token_values(tok_kl, shift_labels)

        # ---------- 8) SATIMP（偏好式 proxy：只用现有 compute_batch_nll 取对的 margin） ----------
        elif loss_name == "satimp":
            # SATIMP 通常是“偏好对”的稳定版；如果 batch 里给了 original/alternate，就按 DPO 的 margin 近似 per-sample
            if isinstance(forget_inputs, dict) and "original" in forget_inputs and "alternate" in forget_inputs:
                lose_m, _ = compute_batch_nll(model, forget_inputs["original"])
                win_m,  _ = compute_batch_nll(model, forget_inputs["alternate"])
                margin = (lose_m - win_m)  # 越大说明更偏向“选错”
                # 两个 beta 控制斜率/边界，既然 compute_satimp_loss 已在外汇总，这里只做可导的 margin surrogate
                per_sample_loss = F.relu(self.satimp.beta1 * margin - self.satimp.beta2)
            else:
                # 若不是成对样本，就退化为“把置信度压散”的 KL proxy
                log_p = F.log_softmax(shift_logits, dim=-1)
                uni = torch.full_like(log_p, 1.0 / log_p.size(-1))
                tok_kl = F.kl_div(log_p, uni, reduction="none", log_target=False).sum(-1)
                per_sample_loss = _seq_avg_from_token_values(tok_kl, shift_labels)

        # ---------- 9) RMU 风格（仅用现有 hidden_states + ref 做 forget 子项的逐样本距离） ----------
        elif loss_name == "rmu":
            # φ 取你已经算过的最后层隐藏状态在答案 token 的均值（E），只用 forget 项 per-sample
            if not hasattr(f_out, "hidden_states") or f_out.hidden_states is None:
                with torch.no_grad():
                    f_out = model(**forget_inputs)
            H = f_out.hidden_states[-1]                    # [B,S,D]
            valid = (forget_inputs["labels"] != -100).unsqueeze(-1)
            E = (H * valid).sum(1) / valid.sum(1).clamp_min(1)  # [B,D]

            # 用 ref 的同构表示作为“目标”（无需其它锚）
            with torch.no_grad():
                r_out = self.ref_model(**forget_inputs)
                Hr = r_out.hidden_states[-1]
                Er = (Hr * valid).sum(1) / valid.sum(1).clamp_min(1)

            per_sample_loss = (E - Er).pow(2).sum(-1).sqrt()  # L2，越大越“记住旧表示”

        else:
            # 回退：用 f_out.loss 广播
            psl = f_out.loss
            per_sample_loss = psl if psl.ndim > 0 else psl.repeat(forget_inputs["input_ids"].size(0))

        # 统一返回“要最小化”的量
        return -per_sample_loss


    # -------- main loss --------
    def compute_loss(self, model, inputs, return_outputs=False):
        # 让模型返回 hidden_states（用于表示侧几何）
        unwrapped = self.accelerator.unwrap_model(model)
        unwrapped.config.output_hidden_states = True

        # 取 forget/retain inputs
        forget_inputs = inputs["forget"] if self.geometric_config.loss != "dpo" else inputs["forget"]["original"]
        retain_inputs = inputs.get("retain")

        # (A) 计算 forget 基础损失（与原流程一致）
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
        #f_out = model(**forget_inputs)  # 先前向一次以拿 hidden_states

        # (B) 用 retain 小 batch 更新切向基（可降频）
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

        # (C) 计算 retain 锚 + 表示侧几何（仅在启用 OT slope 时执行）
        if getattr(self.geometric_config, "use_ot_slope", True):
            # 取 forget 的隐藏表示
            with torch.no_grad():
                # 若 f_out 没有 hidden_states，补一次前向以拿表征（不反传）
                if not hasattr(f_out, "hidden_states") or f_out.hidden_states is None:
                    f_out_no_grad = model(**forget_inputs)
                    E_f_raw = self._hidden_embed(f_out_no_grad, forget_inputs["labels"])
                else:
                    E_f_raw = self._hidden_embed(f_out, forget_inputs["labels"])

                # 更新 whitener 与 anchor
                r_out = model(**retain_inputs)
                E_r_raw = self._hidden_embed(r_out, retain_inputs["labels"])
                self.fastdist.update_whitener(torch.cat([E_f_raw, E_r_raw], dim=0))
                E_r_white = self.fastdist.whiten(E_r_raw)
                self.fastdist.update_retain_anchor(E_r_white)

            # 白化并得到距离/代价
            E_f_white = self.fastdist.whiten(E_f_raw)  # [B, D]

            # # 构造 OT slope 用的 per-sample g（可选是否加入 μ·d）
            g_i = self._per_sample_loss(model, inputs["forget"], f_out)
            # d_hat, C, mu_eff = self._cal_distance_n_cost(g_base, E_f_white)

            # 1) 交叉代价: forget×retain（cosine cost，median 归一 + detach 稳定）
            X = F.normalize(E_f_white, dim=1)                       # [n, D]
            Y = F.normalize(E_r_white, dim=1)                       # [m, D]
            C_fr = (1 - X @ Y.T).clamp(min=0)                       # [n, m]
            with torch.no_grad():
                pos = C_fr[C_fr > 0]
                med = pos.median() if pos.numel() > 0 else torch.tensor(1.0, device=C_fr.device, dtype=C_fr.dtype)
            C_fr = (C_fr / med.clamp_min(1e-6)).detach()

            # 3) 一维对偶 λ 的几步更新，使平均运输成本≈ε
            eps = getattr(self.geometric_config, "wasserstein_epsilon", 0.5)
            eta = getattr(self.geometric_config, "ot_smoothing_eta", 0.2)
            lam = torch.tensor(getattr(self.geometric_config, "ot_lambda_init", 1.0),
                            device=g_i.device, dtype=g_i.dtype)

            for _ in range(getattr(self.geometric_config, "ot_lambda_iters", 5)):
                scores = (g_i[:, None] - lam * C_fr) / max(eta, 1e-6)   # [n, m]
                # 按列 softmax：每个 retain 样本 j 上对 i 归一，得到 π(i|j)
                pi_cols = torch.softmax(scores, dim=0)                   # [n, m]
                avg_cost = (pi_cols * C_fr).sum(dim=0).mean()            # 1/m ∑_j ∑_i π_ij C_ij
                lam = (lam + 0.5 * (avg_cost - eps)).clamp_min(0.0)      # 简单比例步可收敛

            print("lam", lam.item(), "avg_cost", avg_cost.item())
            # 4) 由对偶产生权重并规范化（等价于对 θ 的梯度加权）
            scores = (g_i[:, None] - lam * C_fr) / max(eta, 1e-6)
            pi_cols = torch.softmax(scores, dim=0)                       # [n, m]
            w = pi_cols.mean(dim=1)                                      # [n]
            w = w / w.mean().clamp_min(1e-12)
            print("w", w)

            # 5) Forget loss
            # (a) weighted expectation：
            forget_weighted_mean = (w * g_i).mean()
            # (b) dual upper bound:
            # forget_weighted_mean = lam * eps + (eta / C_fr.size(1)) * torch.logsumexp(scores, dim=0).sum()

        # (D) 计算 retain anchor 损失与总损失
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        # 记录日志
        log_dict = {
            "train/loss": loss.item(),
            f"train/{self.geometric_config.loss}": forget_weighted_mean.item(),
            "train/retain_loss": retain_loss.item(),
        }

        self.log(log_dict)

        return (loss, f_out) if return_outputs else loss

    # # 在 optimizer_step 前显式投影：
    # def optimizer_step(self, *args, **kwargs):
    #     self.null_proj.project_current_grads(
    #         lambda_align=getattr(self.geometric_config, "lambda_align", 0)
    #     )
    #     return super().optimizer_step(*args, **kwargs)
    
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
