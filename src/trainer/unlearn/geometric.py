import re
import torch
import torch.nn as nn
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
    def project_current_grads(self, lambda_align=0.05):
        for name, p in self.model.named_parameters():
            if name not in self.param_names or p.grad is None:
                continue
            g = p.grad
            U = self.basis[name]
            if not U:
                continue

            # Soft alignment: g ← g - λ * P_T g
            if lambda_align > 0:
                for u in U:
                    u = u.to(g.dtype)
                    coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                    g = g - lambda_align * coef * u

            # Hard projection: g ← P_⊥ g
            for u in U:
                u = u.to(g.dtype)
                coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                g = g - coef * u

            p.grad.copy_(g)


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

        # === 自动推断最后 K 层 ===
        auto_k = getattr(self.geometric_config, "auto_last_k_layers", 3)
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

    def _row_softmax_offdiag(self, A: torch.Tensor, temp: float = 1.0, topk: int = None):
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

    def _ot_penalty(self, g_for_grad: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        改进版 OT 斜率正则：
          - 仅用 cost 矩阵构造 soft 邻域（无需求解对偶）；
          - slope = Δg / C，行 softmax 权重；可 top-k 稀疏；
          - 归一化 cost 的中位数，稳定跨 batch 标度；
          - 可设置目标斜率 τ（通常 0）。
        """
        if (not getattr(self.geometric_config, "use_ot_slope", True)) or g_for_grad.numel() < 2:
            return torch.zeros((), device=g_for_grad.device)

        target_slope = getattr(self.geometric_config, "gp_target", 0.0)
        topk = getattr(self.geometric_config, "gp_topk", 8)
        temp = getattr(self.geometric_config, "gp_softmax_temp", 1.0)
        lam_gp = getattr(self.geometric_config, "lambda_gp", 0.0)

        if lam_gp <= 0.0:
            return torch.zeros((), device=g_for_grad.device)

        # 归一化 cost 并构建 soft 邻域（对角置 -inf）
        with torch.no_grad():
            Cn = self._normalize_cost(C.detach())
            pi_off = self._row_softmax_offdiag(-Cn, temp=temp, topk=topk)

        denom = (Cn + 1e-8)
        gi = g_for_grad.view(-1, 1)
        gj = g_for_grad.view(1, -1)
        slopes = (gj - gi) / denom
        per_pair = (slopes - target_slope) ** 2

        penalty = (per_pair * pi_off).sum() / (pi_off.sum().clamp_min(1e-12))
        return lam_gp * penalty

    def _per_sample_loss(self, model, forget_inputs, f_out):
        """
        生成与选择的遗忘损失一致的 per-sample 度量（用于 OT slope 的 g）。
        注意：为稳定性，OT slope 的几何（C、π）不反传，g 会参与反传。
        """
        loss_name = self.geometric_config.loss
        if loss_name == "simnpo":
            forget_loss, _ = compute_batch_nll(model, forget_inputs)
            forget_labels = forget_inputs["labels"]
            loss_mask = forget_labels != -100
            per_sample_loss = F.logsigmoid(
                self.simnpo_config.beta * (forget_loss / loss_mask.sum(-1) - self.simnpo_config.delta)
            ) * 2 / self.simnpo_config.beta

        elif loss_name == "npo":
            forget_loss, _ = compute_batch_nll(model, forget_inputs)
            with torch.no_grad():
                ref_forget_loss, _ = compute_batch_nll(self.ref_model, forget_inputs)
            log_ratio = -(forget_loss - ref_forget_loss)
            per_sample_loss = F.logsigmoid(self.npo_config.beta * log_ratio) * 2 / self.npo_config.beta

        elif loss_name == "dpo":
            original_inputs = forget_inputs["original"]
            alternate_inputs = forget_inputs["alternate"]
            win_loss, _ = compute_batch_nll(model, alternate_inputs)
            lose_loss, _ = compute_batch_nll(model, original_inputs)
            with torch.no_grad():
                ref_win_loss, _ = compute_batch_nll(self.ref_model, alternate_inputs)
                ref_lose_loss, _ = compute_batch_nll(self.ref_model, original_inputs)
            win_log_ratio = -(win_loss - ref_win_loss)
            lose_log_ratio = -(lose_loss - ref_lose_loss)
            per_sample_loss = F.logsigmoid(
                self.dpo_config.beta * (win_log_ratio - lose_log_ratio)
            ) * 2 / self.dpo_config.beta

        elif loss_name == "undial":
            logits = f_out.logits
            with torch.no_grad():
                ref_logits = self.ref_model(**forget_inputs).logits
            shift_labels = forget_inputs["labels"][..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_ref_logits = ref_logits[..., :-1, :].contiguous()

            mask = torch.zeros_like(shift_ref_logits)
            batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
            seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
            mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

            pre_softmax = shift_ref_logits - mask * self.undial_config.beta
            soft_label = F.softmax(pre_softmax, dim=-1)

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_sample_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                soft_label.view(-1, soft_label.size(-1)),
            ).view(shift_logits.size(0), -1).mean(dim=1)

        elif loss_name in ["ce", "ceu"]:
            # 对 CE/CEU：取 token-CE 的句子平均
            logits = f_out.logits
            ce_tok = F.cross_entropy(
                logits.permute(0, 2, 1), forget_inputs["labels"], reduction="none", ignore_index=-100
            )
            mask = (forget_inputs["labels"] != -100)
            denom = mask.sum(dim=1).clamp_min(1)
            per_sample_loss = (ce_tok * mask).sum(dim=1) / denom

        else:
            # fallback：直接用 f_out.loss（若为标量则 broadcast）
            psl = f_out.loss
            per_sample_loss = psl if psl.ndim > 0 else psl.repeat(forget_inputs["input_ids"].size(0))

        return per_sample_loss

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
            forget_loss = forget_loss / loss_mask.sum(-1) - self.npo_config.delta
            forget_loss = -F.logsigmoid(self.npo_config.beta * forget_loss).mean() * 2 / self.npo_config.beta
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

        # (B) 用 retain 小 batch 更新切向基（可降频）
        if retain_inputs is not None:
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
        ot_pen = torch.zeros((), device=forget_loss.device)
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
            if retain_inputs is not None:
                with torch.no_grad():
                    r_out = model(**retain_inputs)
                    E_r_raw = self._hidden_embed(r_out, retain_inputs["labels"])
                    self.fastdist.update_whitener(torch.cat([E_f_raw, E_r_raw], dim=0))
                    E_r_white = self.fastdist.whiten(E_r_raw)
                    self.fastdist.update_retain_anchor(E_r_white)
            else:
                # 无 retain 时，用 ref 在相同输入上构造 teacher anchor
                with torch.no_grad():
                    t_out = self.ref_model(**forget_inputs)
                    E_t_raw = self._hidden_embed(t_out, forget_inputs["labels"])
                    self.fastdist.update_whitener(torch.cat([E_f_raw, E_t_raw], dim=0))
                    E_t_white = self.fastdist.whiten(E_t_raw)
                    self.fastdist.update_retain_anchor(E_t_white)

            # 白化并得到距离/代价
            E_f_white = self.fastdist.whiten(E_f_raw)  # [B, D]
            # 用一个简洁稳定的 CE per-sample 近似做 μ 自适应（仅用于 μ 估计，不反传）
            with torch.no_grad():
                logits_ps = f_out.logits
                ce_tok = F.cross_entropy(
                    logits_ps.permute(0, 2, 1), forget_inputs["labels"],
                    reduction="none", ignore_index=-100
                )
                mask = (forget_inputs["labels"] != -100)
                denom = mask.sum(dim=1).clamp_min(1)
                per_sample_ce = (ce_tok * mask).sum(dim=1) / denom

            d_hat, C, mu_eff = self._cal_distance_n_cost(per_sample_ce, E_f_white)

            # 构造 OT slope 用的 per-sample g（可选是否加入 μ·d）
            g_base = self._per_sample_loss(model, inputs["forget"], f_out)
            if getattr(self.geometric_config, "gp_include_d", True):
                g_for_grad = g_base + mu_eff.detach() * d_hat.detach()
            else:
                g_for_grad = g_base

            # 计算改进 OT slope 罚项
            ot_pen = self._ot_penalty(g_for_grad=g_for_grad, C=C)

        # (D) 计算 retain anchor 损失与总损失
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss + ot_pen

        # 记录日志
        log_dict = {
            "train/loss": loss.item(),
            f"train/{self.geometric_config.loss}": forget_loss.item(),
            "train/retain_loss": retain_loss.item(),
        }
        if getattr(self.geometric_config, "use_ot_slope", True):
            log_dict["train/ot_penalty"] = ot_pen.item()
        self.log(log_dict)

        return (loss, f_out) if return_outputs else loss

    # 在 optimizer_step 前显式投影：
    def optimizer_step(self, *args, **kwargs):
        self.null_proj.project_current_grads(
            lambda_align=getattr(self.geometric_config, "lambda_align", 0.05)
        )
        return super().optimizer_step(*args, **kwargs)
