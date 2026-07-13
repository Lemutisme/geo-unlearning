import math

import torch

from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.wmdp_representation import (
    find_exact_module,
    forward_representation_pair,
    masked_representation_mse,
)
from trainer.unlearn.wmdp_selection import (
    configure_wmdp_trainable_parameters,
    select_wmdp_parameters,
    validate_wmdp_optimizer,
)


class WMDPRMUUnlearn(GeometricUnlearn):
    def __init__(self, *args, **kwargs):
        self.rmu_config = kwargs.pop("rmu_config")
        super().__init__(*args, **kwargs)

        self.wmdp_module_name = str(
            getattr(self.rmu_config, "module_name", "model.layers.7")
        )
        self.steering_coeff = float(
            getattr(self.rmu_config, "steering_coeff", 20.0)
        )
        seed = getattr(self.rmu_config, "seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("WMDP RMU seed must be an integer.")
        self.wmdp_seed = seed
        self.require_paged_adamw = bool(
            getattr(self.rmu_config, "require_paged_adamw", True)
        )
        self.control_vector = None
        self._control_generator = torch.Generator(device="cpu").manual_seed(seed)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        self.ref_model.eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

        configure_wmdp_trainable_parameters(self.model)
        self.model_module = find_exact_module(self.model, self.wmdp_module_name)
        self.ref_module = find_exact_module(self.ref_model, self.wmdp_module_name)
        self._wmdp_runtime_validated = False

    def _selected_named_parameters(self, model=None):
        model = self.model if model is None else model
        return select_wmdp_parameters(model)

    def create_optimizer(self):
        selected = configure_wmdp_trainable_parameters(self.model)
        optimizer = super().create_optimizer()
        validate_wmdp_optimizer(self.optimizer, selected)
        return optimizer

    def _representation_pair(self, model, inputs):
        return forward_representation_pair(
            model,
            self.ref_model,
            inputs,
            self.model_module,
            self.ref_module,
        )

    def _control_target(self, activation):
        hidden_size = activation.shape[-1]
        if self.control_vector is None:
            direction = torch.rand(
                hidden_size,
                generator=self._control_generator,
                dtype=torch.float32,
                device="cpu",
            )
            norm = torch.linalg.vector_norm(direction)
            if not torch.isfinite(norm).item() or norm.item() == 0.0:
                raise RuntimeError("WMDP RMU control direction has zero or invalid norm.")
            self.control_vector = direction.mul(self.steering_coeff / norm)
        elif self.control_vector.numel() != hidden_size:
            raise RuntimeError(
                "WMDP RMU activation width changed after control initialization."
            )
        return self.control_vector.to(
            device=activation.device,
            dtype=activation.dtype,
        ).view(1, 1, hidden_size).expand_as(activation)

    def compute_forget_loss(self, model, forget_inputs):
        activation, _, outputs, mask = self._representation_pair(
            model,
            forget_inputs,
        )
        target = self._control_target(activation)
        return masked_representation_mse(activation, target, mask), outputs

    def compute_retain_loss(self, model, retain_inputs):
        activation, reference, _, mask = self._representation_pair(
            model,
            retain_inputs,
        )
        return masked_representation_mse(activation, reference, mask)

    def _validate_wmdp_runtime(self):
        if self._wmdp_runtime_validated:
            return
        self._validate_gu_runtime()

        if self.gamma != 1.0:
            raise ValueError("WMDP RMU requires gamma=1.")
        if self.alpha != 100.0:
            raise ValueError("WMDP RMU requires alpha=100.")
        if self.steering_coeff != 20.0:
            raise ValueError("WMDP RMU requires steering_coeff=20.")
        if not math.isfinite(self.steering_coeff) or self.steering_coeff <= 0.0:
            raise ValueError("WMDP RMU steering coefficient must be positive and finite.")
        if self.wmdp_module_name != "model.layers.7":
            raise ValueError("WMDP RMU requires module model.layers.7.")

        selected = self._selected_named_parameters()
        validate_wmdp_optimizer(self.optimizer, selected)
        if (
            self.require_paged_adamw
            and self._optimizer_geometry_adapter.name != "paged_adamw_32bit"
        ):
            raise NotImplementedError("WMDP RMU requires PagedAdamW32.")
        self._wmdp_runtime_validated = True

    def training_step(self, model, inputs):
        self._validate_wmdp_runtime()
        return super().training_step(model, inputs)
