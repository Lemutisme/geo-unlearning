import math
import os

import torch

from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.orthograd_geometry import GlobalRetainBasis
from trainer.utils import compute_batch_nll


class OrthogradUnlearn(GeometricUnlearn):
    def __init__(self, *args, **kwargs):
        self.orthograd_config = kwargs.pop("orthograd_config")
        super().__init__(*args, **kwargs)

        self.forget_weight = float(self.orthograd_config.forget_weight)
        self.retain_weight = float(self.orthograd_config.retain_weight)
        self.rank_tolerance = float(self.orthograd_config.rank_tolerance)
        self.maximum_retain_batch_size = int(
            self.orthograd_config.maximum_retain_batch_size
        )
        self.basis_device = str(self.orthograd_config.basis_device).lower()
        self.gamma = self.forget_weight
        self.alpha = self.retain_weight
        self.retain_basis = GlobalRetainBasis(self.rank_tolerance)
        self.orthograd_calls = 0
        self.last_orthograd_diagnostics = {}
        self._orthograd_runtime_validated = False

    @staticmethod
    def _batch_size(inputs):
        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if not isinstance(labels, torch.Tensor) or labels.ndim == 0:
            raise ValueError("Orthograd retain inputs require batched labels.")
        return int(labels.shape[0])

    @classmethod
    def _slice_tree(cls, value, index, batch_size):
        if isinstance(value, dict):
            return {
                key: cls._slice_tree(item, index, batch_size)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._slice_tree(item, index, batch_size) for item in value)
        if isinstance(value, list):
            return [cls._slice_tree(item, index, batch_size) for item in value]
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] != batch_size:
                raise ValueError(
                    "Orthograd retain tensor leading dimensions must match."
                )
            return value[index : index + 1]
        return value

    @classmethod
    def _iter_batch_samples(cls, inputs):
        batch_size = cls._batch_size(inputs)
        for index in range(batch_size):
            yield cls._slice_tree(inputs, index, batch_size)

    def _basis_vector(self, named_params, gradients):
        vector = {}
        for (name, parameter), gradient in zip(named_params, gradients):
            if gradient is None:
                continue
            target_device = (
                torch.device("cpu") if self.basis_device == "cpu" else parameter.device
            )
            vector[name] = gradient.detach().to(
                device=target_device,
                dtype=torch.float32,
            )
        return vector

    def _component_vector(self, component, named_params):
        vector = {}
        for name, parameter in named_params:
            target_device = (
                torch.device("cpu") if self.basis_device == "cpu" else parameter.device
            )
            tensor = self.component_buffers.tensor(component, name, target_device)
            if tensor is not None:
                vector[name] = tensor.float()
        return vector

    def _clear_orthograd_state(self):
        self.component_buffers.clear()
        self.retain_basis.clear()

    @staticmethod
    def _clear_parameter_grads(named_params):
        for _, parameter in named_params:
            parameter.grad = None

    def _collect_retain_gradients(self, model, retain_inputs, named_params):
        batch_size = self._batch_size(retain_inputs)
        labels = retain_inputs["labels"]
        answer_counts = labels[..., 1:].ne(-100).sum(dim=-1)
        if (answer_counts == 0).any():
            self._clear_orthograd_state()
            raise RuntimeError("Orthograd retain sample contains an empty answer mask.")
        total_answer_tokens = int(answer_counts.sum().item())
        params = [parameter for _, parameter in named_params]
        reconstructed_loss = torch.zeros((), device=labels.device)

        try:
            for sample in self._iter_batch_samples(retain_inputs):
                sequence_nll, _ = compute_batch_nll(model, sample)
                sample_loss = sequence_nll.sum() * (
                    float(batch_size) / float(total_answer_tokens)
                )
                gradients = torch.autograd.grad(
                    sample_loss,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                scaled_gradients = tuple(
                    None if gradient is None else gradient / batch_size
                    for gradient in gradients
                )
                self.component_buffers.add(
                    "retain",
                    named_params,
                    scaled_gradients,
                )
                self.retain_basis.add(self._basis_vector(named_params, gradients))
                reconstructed_loss = (
                    reconstructed_loss + sample_loss.detach() / batch_size
                )
                del gradients, scaled_gradients
        except Exception:
            self._clear_orthograd_state()
            raise

        if self.retain_basis.empty:
            self._clear_orthograd_state()
            raise RuntimeError("Orthograd retain gradient basis is empty.")
        return reconstructed_loss, batch_size

    @torch.no_grad()
    def _finalize_orthograd_gradients(self, named_params):
        named_params = list(named_params)
        if not self.component_buffers.has_component("forget"):
            raise RuntimeError("Orthograd forget gradient buffer is empty.")
        if not self.component_buffers.has_component("retain"):
            raise RuntimeError("Orthograd retain gradient buffer is empty.")
        if self.retain_basis.empty:
            raise RuntimeError("Orthograd retain gradient basis is empty.")

        try:
            candidate = self._component_vector("forget", named_params)
            retain_mean = self._component_vector("retain", named_params)
            projected, _, stats = self.retain_basis.project(candidate)

            for name, parameter in named_params:
                projected_tensor = projected.get(name)
                retain_tensor = retain_mean.get(name)
                if projected_tensor is None and retain_tensor is None:
                    continue
                if projected_tensor is None:
                    projected_tensor = torch.zeros_like(retain_tensor)
                if retain_tensor is None:
                    retain_tensor = torch.zeros_like(projected_tensor)
                final_gradient = (
                    self.retain_weight * retain_tensor
                    + self.forget_weight * projected_tensor
                )
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.copy_(
                    final_gradient.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )

            update_step = self.orthograd_calls + 1
            diagnostics = {
                "record_type": "orthograd_geometry",
                "update_step": update_step,
                "orthograd_calls": update_step,
                "mode": "orthograd_euclidean_approx_adam",
                "basis_device": self.basis_device,
                "retain_batch_size": stats.requested_rank,
                "requested_rank": stats.requested_rank,
                "effective_rank": stats.effective_rank,
                "dropped_directions": stats.dropped_directions,
                "candidate_norm": stats.candidate_norm,
                "projected_norm": stats.projected_norm,
                "preserved_signal_ratio": stats.preserved_signal_ratio,
                "maximum_relative_residual": stats.maximum_relative_residual,
                "basis_bytes": stats.basis_bytes,
                "peak_live_vector_count": stats.effective_rank + 2,
                "orthogonalization_flops": stats.orthogonalization_flops,
                "estimated_projection_flops": stats.estimated_projection_flops,
                "selected_numel": sum(
                    parameter.numel() for _, parameter in named_params
                ),
                "forget_weight": self.forget_weight,
                "retain_weight": self.retain_weight,
            }
            if self.diagnostics_writer is not None:
                self.diagnostics_writer.write_step(diagnostics)
            self.actual_delta_callback.prepare_step(
                update_step,
                named_params,
                self.component_buffers,
            )
            self.orthograd_calls = update_step
            self.last_orthograd_diagnostics = diagnostics
            self.last_surgery_diagnostics = diagnostics
        except Exception:
            self._clear_parameter_grads(named_params)
            raise
        finally:
            self._clear_orthograd_state()

    def training_step(self, model, inputs):
        retain_batch_size = self._batch_size(inputs["retain"])
        self._validate_orthograd_runtime(retain_batch_size)
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        named_params = self._selected_named_parameters(model)
        params = [parameter for _, parameter in named_params]
        try:
            with self.compute_loss_context_manager():
                forget_loss, _ = self.compute_forget_loss(model, inputs["forget"])
            forget_gradients = torch.autograd.grad(
                forget_loss,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            self.component_buffers.add("forget", named_params, forget_gradients)
            del forget_gradients

            retain_loss, _ = self._collect_retain_gradients(
                model,
                inputs["retain"],
                named_params,
            )
            self._finalize_orthograd_gradients(named_params)
        except Exception:
            self._clear_parameter_grads(named_params)
            self._clear_orthograd_state()
            raise

        logging_loss = (
            self.forget_weight * forget_loss.detach()
            + self.retain_weight * retain_loss.detach()
        )
        return logging_loss

    @staticmethod
    def _required_vector_storage_bytes(selected_numel, vector_count, headroom=1.2):
        return math.ceil(int(selected_numel) * int(vector_count) * 4 * float(headroom))

    def _validate_orthograd_runtime(self, retain_batch_size):
        if self._orthograd_runtime_validated:
            if retain_batch_size > self.maximum_retain_batch_size:
                raise ValueError(
                    "Orthograd retain batch exceeds maximum retain batch size."
                )
            return

        if self.args.gradient_accumulation_steps != 1:
            raise NotImplementedError(
                "Orthograd MVP requires gradient_accumulation_steps=1."
            )
        if self.basis_device not in {"cpu", "parameter"}:
            raise ValueError(f"Unsupported Orthograd basis device: {self.basis_device}")
        if self.maximum_retain_batch_size <= 0:
            raise ValueError("Orthograd maximum retain batch size must be positive.")
        if retain_batch_size <= 0:
            raise ValueError("Orthograd retain batch must be non-empty.")
        if retain_batch_size > self.maximum_retain_batch_size:
            raise ValueError(
                "Orthograd retain batch exceeds maximum retain batch size."
            )
        if self.forget_weight <= 0.0 or not math.isfinite(self.forget_weight):
            raise ValueError("Orthograd forget_weight must be positive and finite.")
        if self.retain_weight < 0.0 or not math.isfinite(self.retain_weight):
            raise ValueError("Orthograd retain_weight must be non-negative and finite.")

        selected = self._selected_named_parameters()
        trainable = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if {id(parameter) for _, parameter in selected} != {
            id(parameter) for _, parameter in trainable
        }:
            raise NotImplementedError(
                "Orthograd MVP requires all trainable parameters to be selected."
            )
        self._validate_gu_runtime()

        if self.basis_device == "cpu":
            selected_numel = sum(parameter.numel() for _, parameter in selected)
            required = self._required_vector_storage_bytes(
                selected_numel,
                self.maximum_retain_batch_size + 2,
            )
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if available < required:
                raise RuntimeError(
                    "Insufficient host memory for Orthograd basis: "
                    f"requires {required} bytes, found {available} bytes."
                )

        self._orthograd_runtime_validated = True
