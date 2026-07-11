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
        self._gu_runtime_validated = False

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

    def _validate_gu_runtime(self):
        if self._gu_runtime_validated:
            return

        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "Approximate Adam GU does not support DeepSpeed."
            )
        if self.is_fsdp_enabled:
            raise NotImplementedError("Approximate Adam GU does not support FSDP.")
        if self.args.fp16:
            raise NotImplementedError(
                "Approximate Adam GU supports BF16/FP32 only."
            )
        if self.use_apex:
            raise NotImplementedError("Approximate Adam GU does not support Apex.")
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise NotImplementedError(
                "Approximate Adam GU supports one process and one GPU only."
            )

        if self.args.gradient_checkpointing:
            checkpointing_kwargs = self.args.gradient_checkpointing_kwargs or {}
            if checkpointing_kwargs.get("use_reentrant", True):
                raise NotImplementedError(
                    "Approximate Adam GU requires use_reentrant=false."
                )

        named_params = self._selected_named_parameters()
        if not named_params:
            raise ValueError("GU requires at least one selected trainable parameter.")

        optimizer = self._unwrap_optimizer()
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise NotImplementedError(
                "Approximate Adam GU requires torch.optim.Adam or AdamW."
            )

        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        missing = [
            name
            for name, parameter in named_params
            if id(parameter) not in optimizer_parameter_ids
        ]
        if missing:
            raise ValueError(
                "Selected parameters are absent from the optimizer: "
                + ", ".join(missing[:3])
            )

        for group in optimizer.param_groups:
            beta1 = float(group["betas"][0])
            if beta1 != 0.0:
                raise NotImplementedError(
                    "Approximate Adam GU requires beta1=0."
                )
            if float(group.get("weight_decay", 0.0)) != 0.0:
                raise NotImplementedError(
                    "Approximate Adam GU requires weight_decay=0."
                )
            if bool(group.get("amsgrad", False)):
                raise NotImplementedError(
                    "Approximate Adam GU does not support AMSGrad."
                )

        self._gu_runtime_validated = True

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

        if exp_avg_sq.shape != parameter.shape:
            raise RuntimeError("Adam exp_avg_sq shape does not match parameter.")
        if not torch.isfinite(exp_avg_sq).all():
            raise RuntimeError("Adam exp_avg_sq contains non-finite values.")

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

        self._validate_gu_runtime()
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
