import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.unlearn.ceu import compute_batch_ceu
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll, compute_dpo_loss, compute_undial_loss, compute_wga_loss, compute_satimp_loss

import re

def build_last_layers_regex(model, last_k=2, include_lm_head=True, include_final_norm=True):
    """
    自动根据命名推断“最后 K 个 block”的正则，兼容多种命名：
    - Llama/Mistral:  model.layers.{i}.
    - GPT/OPT 类:     transformer.h.{i}. / blocks.{i}. / decoder.layers.{i}.
    还可附带 lm_head 和最后一层 norm。
    返回: List[str] -> 可直接塞给 geometric_config.trainable_params_regex
    """
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

class RetainNullProjector:
    def __init__(self, model, param_name_regex, k=8, use_adam_diag=True, ema=0.98):
        self.model = model
        self.k = k
        self.use_adam_diag = use_adam_diag
        self.ema = ema
        self.param_names = [n for n, p in model.named_parameters()
                           if any(re.fullmatch(rx, n) for rx in param_name_regex)]
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
        params = [p for n, p in self.model.named_parameters() if n in self.param_names and p.requires_grad]
        grads = torch.autograd.grad(retain_loss, params, retain_graph=False, allow_unused=True)
        for (name, p), g in zip([(n, p) for n,p in self.model.named_parameters() if n in self.param_names], grads):
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

            # 先做一阶软惩罚：g ← g - λ * P_T g
            if lambda_align > 0:
                for u in U:
                    u = u.to(g.dtype)
                    coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                    g = g - lambda_align * coef * u

            # 再做硬投影：g ← P_⊥ g
            for u in U:
                u = u.to(g.dtype)
                coef = (g * u).sum().div(u.pow(2).sum().clamp_min(1e-12))
                g = g - coef * u

            p.grad.copy_(g)


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
        # Pop your custom config from kwargs
        # === 自动推断最后 K 层 ===
        auto_k = getattr(self.geometric_config, "auto_last_k_layers", 3)
        if auto_k is not None:
            auto_regex = build_last_layers_regex(
                self.model,
                last_k=int(auto_k),
                include_lm_head=True,
                include_final_norm=True
            )
            # 覆盖/填充到 config，供下游 projector 使用
            self.geometric_config.trainable_params_regex = auto_regex

            # 打印一下，方便确认
            print(f"[GU] auto_last_k_layers={auto_k}, regex={auto_regex}")

        # ===（可选）强力省显存：只让匹配的参数参与训练，其余冻结 ===
        if getattr(self.geometric_config, "freeze_others", False):
            pats = [re.compile(rx) for rx in self.geometric_config.trainable_params_regex]
            for n, p in self.model.named_parameters():
                keep = any(r.fullmatch(n) for r in pats)
                p.requires_grad_(keep)

        self.null_proj = RetainNullProjector(
            self.model,
            param_name_regex=self.geometric_config.trainable_params_regex, # 复用 RMU 的选参方式
            k=getattr(self.geometric_config, "null_k", 8),
            use_adam_diag=False, ema=0.98
        )
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        H = self.model.config.hidden_size
        self.fastdist = OnlineDistanceBuilder(
            hidden_size=H,
            momentum=self.geometric_config.ema_momentum,
            device=self.model.device,
            dtype=torch.float32,
        )

    def compute_loss(self, model, inputs, return_outputs=False):

        # 1) Calculate forget loss (various options)
        forget_inputs = inputs["forget"]
        retain_inputs = inputs.get("retain")

        if self.geometric_config.loss == 'npo':
            forget_loss, f_out = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.npo_config.beta,
            )
        elif self.geometric_config.loss == 'dpo':
            original_inputs = forget_inputs['original']
            alternate_inputs = inputs["forget"]["alternate"]
            forget_loss, f_out = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=alternate_inputs,
                lose_inputs=original_inputs,
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
                model,
                forget_inputs,
                ignore_first_n_answer_tokens=1,
            )
        elif self.geometric_config.loss == 'wga':
            forget_loss, f_out = compute_wga_loss(
                model=model, inputs=forget_inputs, beta=self.wga_config.beta
            )
        elif self.geometric_config.loss == 'satimp':
            forget_loss, f_out = compute_satimp_loss(
                model=model, inputs=forget_inputs, beta1=self.satimp_config.beta1, beta2=self.satimp_config.beta2
            )
        else:
            f_out = model(**forget_inputs)
            forget_loss = -f_out.loss

        # 2) 用 retain 小 batch 更新切向基（减少频率即可，比如每N步或用较小batch）
        if "retain" in inputs and inputs["retain"] is not None:
            with torch.enable_grad():
                # 用 KL(anchor) 最稳：让模型对保留数据“看起来像 ref”
                with torch.no_grad():
                    ref_logits = self.ref_model(**inputs["retain"]).logits
                logits = model(**inputs["retain"]).logits
                retain_kl = F.kl_div(
                    F.log_softmax(logits, dim=-1), 
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean"
                )
                self.null_proj.update_basis_with_retain(retain_kl)

        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        # 3) 返回标量损失；梯度会在 backward hook 里被投影

        self.log({
            "train/loss": loss.item(),
            f"train/{self.geometric_config.loss}": forget_loss.item(),
            "train/retain_loss": retain_loss.item()
        })
        return (loss, f_out) if return_outputs else loss

    # 在 optimizer_step 前显式投影：
    def optimizer_step(self, *args, **kwargs):
        self.null_proj.project_current_grads()
        return super().optimizer_step(*args, **kwargs)

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

class GeometricUnlearn2(GradDiff):
    def __init__(self, *args, **kwargs):
        # Pop your custom config from kwargs
        self.geometric_config = kwargs.pop('geometric_config')
        self.simnpo_config = kwargs.pop("simnpo_config")
        self.npo_config = kwargs.pop("npo_config")
        self.dpo_config = kwargs.pop("dpo_config", None)
        self.undial_config = kwargs.pop("undial_config", None)
        self.wga_config = kwargs.pop("wga_config", None)

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

    def _row_softmax_offdiag(self, M, topk=None, temp=1e3):
        B = M.shape[0]
        big_neg = torch.finfo(M.dtype).min / 4
        M_masked = M.clone()
        M_masked.fill_diagonal_(big_neg)
        if topk is not None and topk < B - 1:
            vals, idx = torch.topk(M_masked, k=topk, dim=1)
            keep = torch.zeros_like(M_masked, dtype=torch.bool)
            keep.scatter_(1, idx, True)
            M_masked = torch.where(keep, M_masked, big_neg)
        pi = torch.softmax(M_masked / temp, dim=1)
        return pi

    def _cal_distance_n_cost(self, per_sample_loss, E_f_white):
        d_hat = self.fastdist.dvec_to_anchor(E_f_white)
        C = self.fastdist.pairwise_cost(E_f_white)

        # 5) Compute g(x) and solve dual for adversarial weights
        # ---- 自动调 μ：让 μ·median(d_hat) ≈ β·median(CE) ----
        with torch.no_grad():
            ce_med = per_sample_loss.median()
            d_med  = (d_hat + 1e-6).median()
            beta = getattr(self.geometric_config, "mu_auto_coeff", 0.5)
            mu_max = getattr(self.geometric_config, "mu_max", 0.1)
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

    # RMU utils
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
        elif self.geometric_config.loss == "ce":
            logits = f_out.logits
            ce_tok = F.cross_entropy(
                logits.permute(0, 2, 1), forget_inputs["labels"], reduction="none", ignore_index=-100
            )
            mask = (forget_inputs["labels"] != -100)
            denom = mask.sum(dim=1).clamp_min(1)
            per_sample_loss = (ce_tok * mask).sum(dim=1) / denom
        else:
            per_sample_loss = f_out.loss
        return per_sample_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        ## Get the actual model object, whether it's wrapped or not
        unwrapped_model = self.accelerator.unwrap_model(model)
        # Enable hidden state outputs on the actual model
        unwrapped_model.config.output_hidden_states = True

        forget_inputs = inputs["forget"] if self.geometric_config.loss not in ["dpo"] else inputs["forget"]["original"]
        retain_inputs = inputs.get("retain")

        # 1) Forward pass using per sample loss on forget set
        f_out = model(**forget_inputs)
        with torch.no_grad():
            logits = f_out.logits
            ce_tok = F.cross_entropy(
                logits.permute(0, 2, 1), forget_inputs["labels"], reduction="none", ignore_index=-100
            )
            mask = (forget_inputs["labels"] != -100)
            denom = mask.sum(dim=1).clamp_min(1)
            per_sample_loss = (ce_tok * mask).sum(dim=1) / denom

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
        g_batch_for_grad = self._per_sample_loss(model, inputs["forget"], f_out) #+ mu_eff * d_hat

        print("adv_weights:", adv_weights)

        robust_forget_risk = -g_batch_for_grad.mean()

        robust_forget_risk_, _ = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.npo_config.beta,
        )

        # 7) Get the final loss by adding retain loss and OT penalty
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        ot_pen = self._ot_penalty(g_for_grad=g_batch_for_grad, C=C, M_star=M_star)
        loss = self.gamma * robust_forget_risk + self.alpha * retain_loss #+ ot_pen

        # Logging
        self.log({
            "train/loss": loss.item(),
            "train/robust_forget_risk": robust_forget_risk.item(),
            "train/robust_forget_risk_": robust_forget_risk_.item(),
            "train/retain_loss": retain_loss.item(),
            "train/ot_penalty": ot_pen.item()
        })

        return (loss, f_out) if return_outputs else loss
