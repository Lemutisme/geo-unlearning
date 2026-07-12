import logging
import math
import re

import torch
import torch.nn.functional as F

from trainer.resource_profiler import ResourceProfileCallback
from trainer.unlearn.component_buffers import ComponentGradientBuffers
from trainer.unlearn.ceu import compute_batch_ceu
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.gradient_surgery import (
    apply_surgery_tensor,
    decide_global_surgery,
)
from trainer.unlearn.gu_diagnostics import (
    ActualDeltaCallback,
    SurgeryDiagnosticsWriter,
)
from trainer.unlearn.optimizer_geometry import make_optimizer_geometry_adapter
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
        for name in sorted(left.keys() & right.keys()):
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
        self.gradient_surgery = str(
            getattr(self.geometric_config, "gradient_surgery", "gu")
        ).lower()
        self.optimizer_geometry = str(
            getattr(self.geometric_config, "optimizer_geometry", "adam")
        ).lower()
        self.projection_eps = float(
            getattr(self.geometric_config, "projection_eps", 1e-12)
        )
        self.trainable_params_regex = list(
            getattr(self.geometric_config, "trainable_params_regex", [".*"])
        )
        self.component_buffer_device = str(
            getattr(self.geometric_config, "component_buffer_device", "parameter")
        )
        self.component_buffers = ComponentGradientBuffers(
            self.component_buffer_device,
            pin_memory=True,
        )
        self.gu_projection_calls = 0
        self.last_gu_diagnostics = {}
        self.surgery_calls = 0
        self.last_surgery_diagnostics = {}
        self._gu_runtime_validated = False
        self._optimizer_geometry_adapter = None
        diagnostics_path = getattr(
            self.geometric_config,
            "diagnostics_path",
            None,
        )
        self.diagnostics_writer = (
            SurgeryDiagnosticsWriter(diagnostics_path) if diagnostics_path else None
        )
        actual_delta_mode = str(
            getattr(self.geometric_config, "actual_delta_mode", "off")
        ).lower()
        self.actual_delta_callback = ActualDeltaCallback(
            mode=actual_delta_mode,
            steps=list(getattr(self.geometric_config, "actual_delta_steps", [1, 10])),
            sample_elements=int(
                getattr(
                    self.geometric_config,
                    "actual_delta_sample_elements",
                    1_000_000,
                )
            ),
            writer=self.diagnostics_writer,
        )
        if self.diagnostics_writer is not None or actual_delta_mode != "off":
            self.add_callback(self.actual_delta_callback)
        resource_profile_path = getattr(
            self.geometric_config,
            "resource_profile_path",
            None,
        )
        self.resource_profiler = ResourceProfileCallback(resource_profile_path)
        if self.resource_profiler.enabled:
            self.add_callback(self.resource_profiler)

    def _profile_phase(self, name):
        return self.resource_profiler.phase(name)

    def _selected_named_parameters(self, model=None):
        model = self.model if model is None else model
        return [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and any(
                re.fullmatch(pattern, name) for pattern in self.trainable_params_regex
            )
        ]

    @torch.no_grad()
    def _accumulate_component_grads(
        self,
        named_params,
        forget_grads,
        retain_grads,
    ):
        self.component_buffers.add("forget", named_params, forget_grads)
        self.component_buffers.add("retain", named_params, retain_grads)

    def _clear_gu_buffers(self):
        self.component_buffers.clear()

    def _is_short_final_accumulation_step(self):
        if not self.accelerator.gradient_state.end_of_dataloader:
            return False
        if not hasattr(self.train_dataset, "__len__"):
            return False

        dataset_length = len(self.train_dataset)
        batch_size = self._train_batch_size
        if self.args.dataloader_drop_last:
            steps_in_epoch = dataset_length // batch_size
        else:
            steps_in_epoch = math.ceil(dataset_length / batch_size)
        return steps_in_epoch <= self.args.gradient_accumulation_steps

    def _validate_gu_runtime(self):
        if self._gu_runtime_validated:
            return

        if self.is_deepspeed_enabled:
            raise NotImplementedError("Approximate Adam GU does not support DeepSpeed.")
        if self.is_fsdp_enabled:
            raise NotImplementedError("Approximate Adam GU does not support FSDP.")
        if self.args.fp16:
            raise NotImplementedError("Approximate Adam GU supports BF16/FP32 only.")
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
        if self.gradient_surgery not in {"gu", "pcgrad"}:
            raise ValueError(
                f"Unsupported gradient surgery mode: {self.gradient_surgery}"
            )
        if self.optimizer_geometry not in {"adam", "euclidean"}:
            raise ValueError(
                f"Unsupported optimizer geometry: {self.optimizer_geometry}"
            )
        if self.component_buffer_device == "cpu":
            ComponentGradientBuffers.validate_host_memory(
                sum(parameter.numel() for _, parameter in named_params)
            )

        adapter = make_optimizer_geometry_adapter(self.optimizer)
        adapter.validate(named_params)
        self._optimizer_geometry_adapter = adapter

        self._gu_runtime_validated = True

    def _sqrt_denominator(self, adapter, parameter, group):
        if self.optimizer_geometry == "euclidean":
            return None
        return adapter.sqrt_denominator(parameter, group)

    def _component_in_frozen_coordinates(
        self,
        component,
        name,
        parameter,
        sqrt_denominator,
    ):
        tensor = self.component_buffers.tensor(
            component,
            name,
            parameter.device,
        )
        if tensor is None:
            return None
        if tensor.dtype != torch.float32:
            raise RuntimeError("GU component buffers must use FP32.")
        tensor = tensor.float()
        if sqrt_denominator is not None:
            tensor = tensor / sqrt_denominator
        return tensor

    @torch.no_grad()
    def _finalize_gu_gradients(self, named_params):
        if not self.component_buffers.has_component("forget"):
            raise RuntimeError("Forget gradient buffer is empty.")
        if not self.component_buffers.has_component("retain"):
            raise RuntimeError("Retain gradient buffer is empty.")

        adapter = self._optimizer_geometry_adapter
        if adapter is None:
            adapter = make_optimizer_geometry_adapter(self.optimizer)
            adapter.validate(named_params)
            self._optimizer_geometry_adapter = adapter
        groups_by_parameter = adapter.groups_by_parameter()
        scalar_device = named_params[0][1].device
        dot_before = torch.zeros((), dtype=torch.float32, device=scalar_device)
        forget_sq = torch.zeros_like(dot_before)
        retain_sq = torch.zeros_like(dot_before)
        identity_fallback_names = set()

        # Pass one computes the single global decision without materializing
        # transformed full-model gradient dictionaries.
        for name, parameter in named_params:
            group = groups_by_parameter[id(parameter)]
            sqrt_denominator = self._sqrt_denominator(adapter, parameter, group)
            forget = self._component_in_frozen_coordinates(
                "forget", name, parameter, sqrt_denominator
            )
            retain = self._component_in_frozen_coordinates(
                "retain", name, parameter, sqrt_denominator
            )
            if forget is None and retain is None:
                continue
            if sqrt_denominator is None:
                identity_fallback_names.add(name)
            if forget is not None:
                forget_sq.add_(forget.square().sum())
            if retain is not None:
                retain_sq.add_(retain.square().sum())
            if forget is not None and retain is not None:
                dot_before.add_((forget * retain).sum())

        decision = decide_global_surgery(
            self.gradient_surgery,
            dot_before,
            retain_sq,
            self.projection_eps,
        )
        gu_decision = decide_global_surgery(
            "gu",
            dot_before,
            retain_sq,
            self.projection_eps,
        )
        dot_after = dot_before - decision.coefficient * retain_sq
        projected_sq = (
            forget_sq
            - 2.0 * decision.coefficient * dot_before
            + decision.coefficient.square() * retain_sq
        ).clamp_min(0.0)
        forget_norm = forget_sq.sqrt()
        retain_norm = retain_sq.sqrt()
        projected_norm = projected_sq.sqrt()
        final_sq = torch.zeros_like(dot_before)
        pcgrad_gu_difference_sq = torch.zeros_like(dot_before)
        gu_final_sq = torch.zeros_like(dot_before)
        forget_final_dot = torch.zeros_like(dot_before)
        retain_final_dot = torch.zeros_like(dot_before)

        # Pass two reconstructs one parameter block at a time, writes the
        # final raw gradient, then releases the transient coordinate tensors.
        for name, parameter in named_params:
            group = groups_by_parameter[id(parameter)]
            sqrt_denominator = self._sqrt_denominator(adapter, parameter, group)
            forget = self._component_in_frozen_coordinates(
                "forget", name, parameter, sqrt_denominator
            )
            retain = self._component_in_frozen_coordinates(
                "retain", name, parameter, sqrt_denominator
            )
            if forget is None and retain is None:
                continue
            if forget is None:
                forget = torch.zeros_like(retain)
            if retain is None:
                retain = torch.zeros_like(forget)

            projected_forget = apply_surgery_tensor(forget, retain, decision)
            gu_projected_forget = apply_surgery_tensor(
                forget,
                retain,
                gu_decision,
            )
            final_coordinates = self.gamma * projected_forget + self.alpha * retain
            gu_final_coordinates = (
                self.gamma * gu_projected_forget + self.alpha * retain
            )
            final_sq.add_(final_coordinates.square().sum())
            gu_final_sq.add_(gu_final_coordinates.square().sum())
            pcgrad_gu_difference_sq.add_(
                (final_coordinates - gu_final_coordinates).square().sum()
            )
            forget_final_dot.add_((forget * final_coordinates).sum())
            retain_final_dot.add_((retain * final_coordinates).sum())

            final_gradient = self._from_adam_coordinates(
                final_coordinates,
                sqrt_denominator,
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
        cosine_before = dot_before / (forget_norm * retain_norm + self.projection_eps)
        cosine_after = dot_after / (projected_norm * retain_norm + self.projection_eps)
        relative_surgery_magnitude = (decision.coefficient.abs() * retain_norm) / (
            forget_norm + self.projection_eps
        )
        relative_pcgrad_gu_distance = None
        if self.gradient_surgery == "pcgrad":
            relative_pcgrad_gu_distance = float(
                (
                    pcgrad_gu_difference_sq.sqrt()
                    / (gu_final_sq.sqrt() + self.projection_eps)
                ).item()
            )

        self.surgery_calls += 1
        self.gu_projection_calls = self.surgery_calls
        diagnostics = {
            "record_type": "geometry",
            "update_step": self.surgery_calls,
            "mode": self.gradient_surgery,
            "projection_calls": self.surgery_calls,
            "surgery_calls": self.surgery_calls,
            "optimizer_geometry": (
                "euclidean" if self.optimizer_geometry == "euclidean" else adapter.name
            ),
            "component_buffer_device": self.component_buffer_device,
            "conflict": decision.conflict,
            "zero_retain_norm": bool(retain_sq.item() == 0.0),
            "coefficient": float(decision.coefficient.item()),
            "raw_coefficient": float(decision.raw_coefficient.item()),
            "forget_norm": float(forget_norm.item()),
            "retain_norm": float(retain_norm.item()),
            "projected_forget_norm": float(projected_norm.item()),
            "final_coordinate_norm": float(final_sq.sqrt().item()),
            "dot_before": float(dot_before.item()),
            "dot_after": float(dot_after.item()),
            "cosine_before": float(cosine_before.item()),
            "cosine_after": float(cosine_after.item()),
            "relative_orthogonality_residual": float(relative_residual.item()),
            "relative_surgery_magnitude": float(relative_surgery_magnitude.item()),
            "relative_pcgrad_gu_distance": relative_pcgrad_gu_distance,
            "predicted_forget_directional_derivative": float(
                (-forget_final_dot).item()
            ),
            "predicted_retain_directional_derivative": float(
                (-retain_final_dot).item()
            ),
            "identity_fallback_parameters": len(identity_fallback_names),
        }
        self.last_surgery_diagnostics = diagnostics
        self.last_gu_diagnostics = diagnostics
        if self.diagnostics_writer is not None:
            self.diagnostics_writer.write_step(diagnostics)
        self.actual_delta_callback.prepare_step(
            self.surgery_calls,
            named_params,
            self.component_buffers,
        )
        logger.info(
            "Gradient surgery step=%d mode=%s coefficient=%.8e residual=%.8e "
            "identity_fallback_parameters=%d",
            self.surgery_calls,
            diagnostics["mode"],
            diagnostics["coefficient"],
            diagnostics["relative_orthogonality_residual"],
            diagnostics["identity_fallback_parameters"],
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
        with self._profile_phase("component_gradients"):
            forget_grads = torch.autograd.grad(
                forget_loss * scale,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            self.component_buffers.add("forget", named_params, forget_grads)
            del forget_grads

            retain_grads = torch.autograd.grad(
                retain_loss * scale,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            self.component_buffers.add("retain", named_params, retain_grads)
            del retain_grads

            self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients or self._is_short_final_accumulation_step():
            with self._profile_phase("projection_writeback"):
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
