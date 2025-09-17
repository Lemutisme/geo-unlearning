import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll

class OnlineDistanceBuilder:
    """
    Online distance builder adapted for multi-GPU training.
    """
    def __init__(self, hidden_size: int, momentum: float = 0.99, device=None, dtype=torch.float32,
                 freeze_anchor: bool = True):
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
        # E is the source of truth for the current device
        current_device = E.device

        # Ensure EMA tensors are on the correct device before the operation
        self._ema_mean = self._ema_mean.to(current_device)
        self._ema_var = self._ema_var.to(current_device)
        bmean = E.mean(dim=0)
        bvar  = E.var(dim=0, unbiased=False)
        self._ema_mean = self.m * self._ema_mean + (1 - self.m) * bmean
        self._ema_var  = self.m * self._ema_var  + (1 - self.m) * bvar

    def whiten(self, E: torch.Tensor) -> torch.Tensor:
        current_device = E.device
        # Ensure EMA tensors are on the correct device for the operation
        ema_mean = self._ema_mean.to(current_device)
        ema_var = self._ema_var.to(current_device)
        return (E - ema_mean) / (ema_var + self._eps).sqrt()

    @torch.no_grad()
    def update_retain_anchor(self, E_retain_white):
        # Freeze-after-first if configured
        current_device = E_retain_white.device
        rmean = E_retain_white.mean(dim=0)
        if self._anchor is None:
            self._anchor = rmean.clone().to(current_device)
        elif not self._freeze_anchor:
            self._anchor = self._anchor.to(current_device)
            self._anchor = self.m * self._anchor + (1 - self.m) * rmean

    def dvec_to_anchor(self, E_forget_white):
        # distance to anchor: cosine
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


class GeometricUnlearn(GradDiff):
    def __init__(self, *args, **kwargs):
        # Pop your custom config from kwargs
        self.geometric_config = kwargs.pop('geometric_config')
        self.simnpo_config = kwargs.pop("simnpo_config")
        self.npo_config = kwargs.pop("npo_config")
        self.dpo_config = kwargs.pop("dpo_config", None)

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

    def _hidden_embed(self, outputs, labels):
        H = outputs.hidden_states[-1]
        # hidden answer
        mask = (labels != -100).unsqueeze(-1)
        Hw = H.masked_fill(~mask, 0.0)
        cnt = mask.sum(dim=1).clamp_min(1)
        return Hw.sum(dim=1) / cnt

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

    def _cal_distance_n_cost(self, per_sample_loss, E_f_white):
        d_vec = self.fastdist.dvec_to_anchor(E_f_white)
        C = self.fastdist.pairwise_cost(E_f_white)

        # ---- 距离无量纲化：d_hat ~ O(1) ----
        H = E_f_white.size(1)
        scale = (2.0 * H) ** 0.5
        d_hat = d_vec / (scale + 1e-6)

        # 5) Compute g(x) and solve dual for adversarial weights
        # ---- 自动调 μ：让 μ·median(d_hat) ≈ β·median(CE) ----
        with torch.no_grad():
            ce_med = per_sample_loss.median()
            d_med  = (d_hat + 1e-6).median()
            beta = getattr(self.geometric_config, "mu_auto_coeff", 0.5)
            mu_max = getattr(self.geometric_config, "mu_max", 0.2)
            mu_eff = (beta * ce_med / d_med).clamp(1e-3, mu_max)
            C_off = C[C > 0]
            C_med = C_off.median() if C_off.numel() else torch.tensor(1.0, device=C.device)
            rho = getattr(self.geometric_config, "eps_rho", 0.3)  # 推荐 0.2~0.5
            eps_adapt = rho * C_med
        return d_hat, C, mu_eff, eps_adapt

    def _ot_penalty(self, g_for_grad, C, M_star, target_slope=1.0, eps=1e-8, topk=8) -> torch.Tensor:
        # Use cost (not sqrt) as denominator per doc; weight pairs by row-softmax over M_star with masked diagonal
        if g_for_grad.numel() < 2:
            return torch.zeros((), device=g_for_grad.device)
        target_slope = getattr(self.geometric_config, "gp_target", 1.0)
        topk = getattr(self.geometric_config, "gp_topk", 8)
        Cn = self._normalize_cost(C.detach())
        denom = Cn + eps
        with torch.no_grad():
            pi_off = self._row_softmax_offdiag(M_star.detach(), topk=topk)
        gi = g_for_grad.view(-1, 1)
        gj = g_for_grad.view(1, -1)
        slopes = (gj - gi) / denom
        per_pair = (slopes - target_slope) ** 2
        penalty = (per_pair * pi_off).sum() / (pi_off.sum().clamp_min(1e-12))
        return self.geometric_config.lambda_gp * penalty

    def _solve_kantorovich_dual(self, g_batch, cost_matrix, epsilon, eta):
        # Fix: return M_star (score matrix), and compute row-softmax π(j|i), then column-marginal adv weights
        g = g_batch.detach()
        C = cost_matrix.detach()
        lam_param = torch.nn.Parameter(torch.tensor(1.0, device=g.device))
        opt = torch.optim.Adam([lam_param], lr=0.1)
        B = g.numel()
        big_neg = -1e9

        for _ in range(50):
            opt.zero_grad()
            lam = F.softplus(lam_param)
            M = (g.unsqueeze(0) - lam * C) / eta                 # shape [B, B], row i, col j
            # mask diagonal to avoid self-transport
            M_masked = M.masked_fill(torch.eye(B, device=M.device, dtype=torch.bool), big_neg)
            dual = lam * epsilon + (eta / B) * torch.logsumexp(M_masked, dim=1).sum()
            dual.backward()
            opt.step()

        with torch.no_grad():
            lam_star = F.softplus(lam_param)
            M_star = (g.unsqueeze(0) - lam_star * C) / eta
            M_star = M_star.masked_fill(torch.eye(B, device=M_star.device, dtype=torch.bool), big_neg)
            robust_risk = lam_star * epsilon + (eta / B) * torch.logsumexp(M_star, dim=1).sum()
            pi_row = torch.softmax(M_star, dim=1)                 # π(j|i)
            adv_weights = pi_row.mean(dim=0)                      # \bar a_j = (1/n) sum_i π(j|i)
        return robust_risk, adv_weights, M_star

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        )  # Shape (b, s, d)
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)  # Shape: [b, s, d]
        squared_diff_sum = (
            (squared_diff * expanded_mask).mean(dim=2).sum(dim=(1))
        )  # Shape: [b, 1]
        num_tokens = mask.sum(dim=-1, keepdim=True)  # Sum over seq_len, Shape: [b, 1]
        return (squared_diff_sum / num_tokens)
    
    def _per_sample_loss(self, model, forget_inputs, f_out):
        if self.geometric_config.loss == "simnpo":
            forget_loss, _ = compute_batch_nll(model, forget_inputs)
            forget_labels = forget_inputs["labels"]
            loss_mask = forget_labels != -100
            per_sample_loss = F.logsigmoid(
                self.simnpo_config.beta * (forget_loss / loss_mask.sum(-1) - self.simnpo_config.delta)
            ) * 2 / self.simnpo_config.beta
        elif self.geometric_config.loss == "npo":
            # NPO loss is the negative log-likelihood ratio against a reference model.
            forget_loss, _ = compute_batch_nll(model, forget_inputs)
            with torch.no_grad():
                ref_forget_loss, _ = compute_batch_nll(self.ref_model, forget_inputs)
            log_ratio = -(forget_loss - ref_forget_loss)
            per_sample_loss = F.logsigmoid(self.npo_config.beta * log_ratio) * 2 / self.npo_config.beta
        elif self.geometric_config.loss == "dpo":
            # DPO requires 'original' and 'alternate' inputs for chosen and rejected responses.
            original_inputs = forget_inputs["original"]
            alternate_inputs = forget_inputs["alternate"]
            
            win_loss, _ = compute_batch_nll(model, alternate_inputs)
            lose_loss, _ = compute_batch_nll(model, original_inputs)
            with torch.no_grad():
                ref_win_loss, _ = compute_batch_nll(self.ref_model, alternate_inputs)
                ref_lose_loss, _ = compute_batch_nll(self.ref_model, original_inputs)

            win_log_ratio = -(win_loss - ref_win_loss)
            lose_log_ratio = -(lose_loss - ref_lose_loss)
            per_sample_loss = F.logsigmoid(self.dpo_config.beta * (win_log_ratio - lose_log_ratio)) * 2 / self.dpo_config.beta
        elif self.geometric_config.loss == "rmu":
            # RMU minimizes the distance between forget activations and a control vector.
            model_forget_activations, _ = self.forward_with_cache(
                model, forget_inputs, self.model_module, no_grad=False
            )
            control_vec = self.get_control_vector(model_forget_activations.shape[-1])
            control_vec = control_vec.to(
                dtype=model_forget_activations.dtype, device=model_forget_activations.device
            )
            control_vec = control_vec.expand_as(model_forget_activations)
            mask = forget_inputs["labels"] != -100
            per_sample_loss = self.compute_activation_loss(
                model_forget_activations, control_vec, mask, reduction='none' 
            )
        elif self.geometric_config.loss == "undial":
            # UNDIAL uses KL divergence against a reference model with adjusted logits.
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
        else:
            logits = f_out.logits
            ce_tok = F.cross_entropy(
                logits.permute(0, 2, 1), forget_inputs["labels"], reduction="none", ignore_index=-100
            )
            mask = (forget_inputs["labels"] != -100)
            denom = mask.sum(dim=1).clamp_min(1)
            per_sample_loss = (ce_tok * mask).sum(dim=1) / denom
        return per_sample_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        ## Get the actual model object, whether it's wrapped or not
        unwrapped_model = self.accelerator.unwrap_model(model)
        # Enable hidden state outputs on the actual model
        unwrapped_model.config.output_hidden_states = True

        forget_inputs = inputs["forget"]
        retain_inputs = inputs.get("retain")

        # 1) Forward pass using per sample loss on forget set
        f_out = model(**forget_inputs)
        with torch.no_grad():
            per_sample_loss = self._per_sample_loss(model, forget_inputs, f_out)

        # 2) Get hidden embeddings and whiten them
        E_f_raw = self._hidden_embed(f_out, forget_inputs["labels"])

        # 3) Update whitener and anchor using retain set (if available) or oracle
        if retain_inputs:
            with torch.no_grad():
                r_out = model(**retain_inputs)
                E_r_raw = self._hidden_embed(
                    r_out, retain_inputs["labels"],
                )
                self.fastdist.update_whitener(torch.cat([E_f_raw.detach(), E_r_raw.detach()], dim=0))
                E_r_white = self.fastdist.whiten(E_r_raw)
                self.fastdist.update_retain_anchor(E_r_white.detach())
        else:
            with torch.no_grad():
                t_out = self.ref_model(**forget_inputs)
                E_t_raw = self._hidden_embed(
                    t_out, forget_inputs["labels"],
                )
                self.fastdist.update_whitener(torch.cat([E_f_raw.detach(), E_t_raw.detach()], dim=0))
                E_t_white = self.fastdist.whiten(E_t_raw)
                self.fastdist.update_retain_anchor(E_t_white.detach())

        E_f_white = self.fastdist.whiten(E_f_raw)

        # 4) Calculate distance and cost matrix
        d_hat, C, mu_eff, eps_adapt = self._cal_distance_n_cost(per_sample_loss, E_f_white)

        # 5) Calculate g(x) = \ell + μ·d_hat
        g_batch = per_sample_loss + mu_eff * d_hat
        _, adv_weights, M_star = self._solve_kantorovich_dual(
            g_batch, C, eps_adapt, self.geometric_config.ot_smoothing_eta
        )

        # 6) Calculate the final differentiable loss
        g_batch_for_grad = self._per_sample_loss(model, forget_inputs, f_out) + mu_eff * d_hat

        robust_forget_risk = torch.sum(adv_weights * g_batch_for_grad)

        # 7) Get the final loss by adding retain loss and OT penalty
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        ot_pen = self._ot_penalty(g_for_grad=g_batch_for_grad, C=C, M_star=M_star)
        loss = - self.gamma * robust_forget_risk + self.alpha * retain_loss + ot_pen

        # Logging
        self.log({
            "train/loss": loss.item(),
            "train/robust_forget_risk": -robust_forget_risk.item(),
            "train/retain_loss": retain_loss.item(),
            "train/ot_penalty": ot_pen.item()
        })

        return (loss, f_out) if return_outputs else loss
