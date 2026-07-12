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

    def _clear_orthograd_state(self):
        self.component_buffers.clear()
        self.retain_basis.clear()

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
