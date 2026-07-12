import math
import os

from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.orthograd_geometry import GlobalRetainBasis


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
