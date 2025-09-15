import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_kl_divergence

class OnlineDistanceBuilder:
    """
    Online distance builder adapted for multi-GPU training.
    """
    def __init__(self, hidden_size: int, momentum: float = 0.99, device=None, dtype=torch.float32):
        self.initial_device = device or torch.device("cpu")
        self.dtype = dtype
        H = hidden_size
        self.m = momentum
        # Initialize on the initial device, but they will be moved as needed.
        self._ema_mean = torch.zeros(H, device=self.initial_device, dtype=self.dtype)
        self._ema_var  = torch.ones(H,  device=self.initial_device, dtype=self.dtype)
        self._anchor = None
        self._eps = 1e-8

    @torch.no_grad()
    def update_whitener(self, E: torch.Tensor):
        current_device = E.device
        self._ema_mean = self._ema_mean.to(current_device)
        self._ema_var  = self._ema_var.to(current_device)

        # 更稳的聚合：聚合一阶/二阶矩，避免“直接平均方差”的偏差
        n = torch.tensor([E.size(0)], device=current_device, dtype=E.dtype)
        sum_x  = E.sum(dim=0)
        sum_x2 = (E * E).sum(dim=0)

        # DDP 同步
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(n,     op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_x, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_x2,op=torch.distributed.ReduceOp.SUM)

        N = n.clamp_min(1.0)
        mean = sum_x / N
        var  = (sum_x2 / N) - mean * mean
        var  = var.clamp_min(0.0)

        self._ema_mean = self.m * self._ema_mean + (1 - self.m) * mean
        self._ema_var  = self.m * self._ema_var  + (1 - self.m) * var

    def whiten(self, E: torch.Tensor) -> torch.Tensor:
        current_device = E.device
        # Ensure EMA tensors are on the correct device for the operation
        ema_mean = self._ema_mean.to(current_device)
        ema_var = self._ema_var.to(current_device)
        return (E - ema_mean) / (ema_var + self._eps).sqrt()

    @torch.no_grad()
    def update_retain_anchor(self, E_retain_white: torch.Tensor):
        current_device = E_retain_white.device
        rmean = E_retain_white.mean(dim=0)

        # DDP 同步
        rmean = _ddp_mean(rmean)

        if self._anchor is None:
            self._anchor = rmean.clone().to(current_device)
        else:
            self._anchor = self._anchor.to(current_device)
            self._anchor = self.m * self._anchor + (1 - self.m) * rmean

    def dvec_to_anchor(self, E_forget_white: torch.Tensor, metric="l2") -> torch.Tensor:
        current_device = E_forget_white.device
        if self._anchor is None:
            return torch.zeros(E_forget_white.size(0), device=current_device, dtype=E_forget_white.dtype)
        
        # Ensure anchor is on the correct device before calculating distance
        anchor = self._anchor.to(current_device)
        diff = E_forget_white - anchor.unsqueeze(0)
        if metric == "l2":
            return diff.norm(dim=1)
        elif metric == "l1":
            return diff.abs().sum(dim=1)
        else:  # cosine
            num = (E_forget_white * anchor).sum(dim=1)
            den = E_forget_white.norm(dim=1).clamp_min(1e-8) * anchor.norm().clamp_min(1e-8)
            return 1 - (num / den)

    # The pairwise_cost method doesn't use internal state, so it doesn't need changes.
    def pairwise_cost(self, E_white: torch.Tensor, metric="l2") -> torch.Tensor:
        # ... (no changes needed here)
        if metric == "l2":
            C = torch.cdist(E_white, E_white, p=2)
        elif metric == "l1":
            C = torch.cdist(E_white, E_white, p=1)
        else:  # cosine
            X = F.normalize(E_white, dim=1)
            C = 1 - X @ X.t()
            C = C.clamp(min=0)
        C = C - torch.diag_embed(torch.diagonal(C))
        return C

def _ddp_mean(t: torch.Tensor):
    if t is None:
        return None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = t.clone()
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        t /= torch.distributed.get_world_size()
    return t

class GeometricUnlearn(GradDiff):
    def __init__(self, *args, **kwargs):
        # Pop your custom config from kwargs
        self.geometric_config = kwargs.pop('geometric_config')
        super().__init__(*args, **kwargs)
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        H = self.model.config.hidden_size
        self.fastdist = OnlineDistanceBuilder(
            hidden_size=H,
            momentum=self.geometric_config.ema_momentum,
            device=self.model.device,
            dtype=torch.float32,
        )

    def _hidden_embed(self, outputs, labels, mode="hidden_answer", pool="mean"):
        H = outputs.hidden_states[-1]
        if mode == "hidden_answer":
            mask = (labels != -100).unsqueeze(-1)
        else:
            mask = (labels == -100).unsqueeze(-1)
        Hw = H.masked_fill(~mask, 0.0)
        cnt = mask.sum(dim=1).clamp_min(1)
        if pool == "mean":
            return Hw.sum(dim=1) / cnt
        else:
            idx = mask.squeeze(-1).int().argmax(dim=1)
            return H[torch.arange(H.size(0), device=H.device), idx, :]

    def _per_sample_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_tok = F.cross_entropy(
            logits.permute(0, 2, 1), labels, reduction="none", ignore_index=-100
        )
        mask = (labels != -100)
        denom = mask.sum(dim=1).clamp_min(1)
        return (ce_tok * mask).sum(dim=1) / denom

    def _normalize_cost(self, C: torch.Tensor) -> torch.Tensor:
        Cn = C.clone()
        pos = Cn[(Cn > 0) & torch.isfinite(Cn)]
        if pos.numel() > 0:
            med = pos.median()
            if med > 0:
                Cn = Cn / med
        return Cn

    def _row_softmax_offdiag(self, M: torch.Tensor, topk: int | None = None) -> torch.Tensor:
        B = M.shape[0]
        big_neg = torch.finfo(M.dtype).min / 4
        M_masked = M.clone()
        M_masked.fill_diagonal_(big_neg)
        if topk is not None and topk < B - 1:
            vals, idx = torch.topk(M_masked, k=topk, dim=1)
            keep = torch.zeros_like(M_masked, dtype=torch.bool)
            keep.scatter_(1, idx, True)
            M_masked = torch.where(keep, M_masked, big_neg)
        pi = torch.softmax(M_masked, dim=1)
        return pi

    def _ot_penalty(self, g_for_grad, C, M_logits, target_slope=1.0, eps=1e-6, topk=8):
        if g_for_grad.numel() < 2:
            return torch.zeros((), device=g_for_grad.device)
        # C 已在外部 normalize 过，这里不要再 normalize，避免二次缩放的漂移
        denom = C.clamp_min(0).sqrt() + eps

        # 用 logits 计算 off-diagonal 概率，且仅保留 topk
        with torch.no_grad():
            pi_off = self._row_softmax_offdiag(M_logits, topk=topk)   # logits -> softmax
        gi = g_for_grad.view(-1, 1)
        gj = g_for_grad.view(1, -1)
        slopes = (gj - gi) / denom
        per_pair = (slopes - target_slope) ** 2
        penalty = (per_pair * pi_off).sum() / (pi_off.sum().clamp_min(1e-12))
        return penalty

    def _solve_kantorovich_dual(self, g_batch, cost_matrix, epsilon, eta):
        g = g_batch.detach()
        C = cost_matrix.detach()
        lambda_param = torch.nn.Parameter(torch.tensor(1.0, device=g.device))
        optimizer = torch.optim.Adam([lambda_param], lr=0.1)
        B = g.numel()
        for _ in range(50):
            optimizer.zero_grad()
            lam = F.softplus(lambda_param)
            M = (g.unsqueeze(0) - lam * C) / eta           # (B,B) logits
            dual_obj = lam * epsilon + (eta / B) * torch.logsumexp(M, dim=1).sum()
            dual_obj.backward()
            optimizer.step()

        lam_star = F.softplus(lambda_param).detach()
        with torch.no_grad():
            M_star = (g.unsqueeze(0) - lam_star * C) / eta  # logits
            robust_risk = lam_star * epsilon + (eta / B) * torch.logsumexp(M_star, dim=1).sum()
            # 列正则化得到列边缘（与你原来的做法一致）
            pi_star_cols = torch.softmax(M_star.T, dim=1)
            adv_weights = pi_star_cols.mean(dim=0)          # (B,)

        return robust_risk, adv_weights, M_star

    def compute_loss(self, model, inputs, return_outputs=False):
        ## Get the actual model object, whether it's wrapped or not
        unwrapped_model = self.accelerator.unwrap_model(model)

        # Enable hidden state outputs on the actual model
        unwrapped_model.config.output_hidden_states = True

        forget_inputs = inputs["forget"]
        retain_inputs = inputs.get("retain")

        # 1) Forward pass on forget set
        f_out = model(**forget_inputs)
        with torch.no_grad():
            per_sample_loss = self._per_sample_ce(f_out.logits.detach(), forget_inputs["labels"])

        # 2) Get hidden embeddings and whiten them
        E_f_raw = self._hidden_embed(
            f_out, forget_inputs["labels"],
            mode=self.geometric_config.distance_mode,
            pool=self.geometric_config.hidden_pool
        )

        # 3) Update whitener and anchor using retain set (if available) or oracle
        if retain_inputs:
            with torch.no_grad():
                r_out = model(**retain_inputs)
                E_r_raw = self._hidden_embed(
                    r_out, retain_inputs["labels"],
                    mode=self.geometric_config.distance_mode,
                    pool=self.geometric_config.hidden_pool
                )
                self.fastdist.update_whitener(torch.cat([E_f_raw.detach(), E_r_raw.detach()], dim=0))
                E_r_white = self.fastdist.whiten(E_r_raw)
                self.fastdist.update_retain_anchor(E_r_white.detach())
        elif self.ref_model:
            with torch.no_grad():
                t_out = self.ref_model(**forget_inputs)
                E_t_raw = self._hidden_embed(
                    t_out, forget_inputs["labels"],
                    mode=self.geometric_config.distance_mode,
                    pool=self.geometric_config.hidden_pool
                )
                self.fastdist.update_whitener(torch.cat([E_f_raw.detach(), E_t_raw.detach()], dim=0))
                E_t_white = self.fastdist.whiten(E_t_raw)
                self.fastdist.update_retain_anchor(E_t_white.detach())
        else: # Fallback
            self.fastdist.update_whitener(E_f_raw.detach())

        E_f_white = self.fastdist.whiten(E_f_raw)

        # 4) Calculate distance and cost matrix
        d_vec = self.fastdist.dvec_to_anchor(E_f_white, metric="l2")
        C = self.fastdist.pairwise_cost(E_f_white, metric=self.geometric_config.cost_metric)
        
        # 5) Compute g(x) and solve dual for adversarial weights
        # 把 base_g 的尺度对齐到 CE 的中位数
        with torch.no_grad():
            ce_med = per_sample_loss.median()                       # 标尺
            d_med  = (d_vec + 1e-6).median()
            s = (ce_med / d_med).clamp(0.25, 4.0)                  # 防爆、防塌
        d_aligned = s * d_vec
        g_batch = per_sample_loss + self.geometric_config.mu * d_aligned

        robust_risk, adv_weights, M_star = self._solve_kantorovich_dual(
            g_batch.detach(), C.detach(),
            self.geometric_config.wasserstein_epsilon,
            self.geometric_config.ot_smoothing_eta
        )

        # 6) Calculate the final differentiable loss
        g_batch_for_grad = self._per_sample_ce(f_out.logits, forget_inputs["labels"]) + self.geometric_config.mu * d_vec
        robust_forget_risk = torch.sum(adv_weights * g_batch_for_grad)

        # 7) KL Anchor on retain set
        if retain_inputs and self.ref_model:
            kl_term, _ = compute_kl_divergence(model, self.ref_model, retain_inputs)
        else:
            kl_term = torch.tensor(0.0, device=model.device)

        # 8) Gradient Penalty
        if self.geometric_config.lambda_gp > 0.0:
            C_norm = self._normalize_cost(C)
            gp_pen_raw = self._ot_penalty(
                g_for_grad=g_batch_for_grad, C=C_norm, M_logits=M_star,
                target_slope=self.geometric_config.gp_target,
                topk=self.geometric_config.gp_topk,
            )
            gp_pen = self.geometric_config.lambda_gp * gp_pen_raw
        else:
            gp_pen = torch.tensor(0.0, device=model.device)

        # Final loss
        loss = -robust_forget_risk + kl_term + gp_pen

        # Logging
        self.log({
            "train/loss": loss.item(),
            "train/robust_forget_risk": -robust_forget_risk.item(),
            "train/kl_anchor": kl_term.item(),
            "train/gp_penalty": gp_pen.item()
        })

        return (loss, f_out) if return_outputs else loss
