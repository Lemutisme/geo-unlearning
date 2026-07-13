import math

import torch

from trainer.unlearn.uam import UAMUnlearn
from trainer.unlearn.uam_geometry import ResidualGUProjection
from trainer.unlearn.wmdp_representation import (
    find_exact_module,
    forward_representation_pair,
    masked_representation_mse,
    seeded_gaussian_noise,
)
from trainer.unlearn.wmdp_selection import (
    configure_wmdp_trainable_parameters,
    select_wmdp_parameters,
    validate_wmdp_optimizer,
)


class WMDPUAMUnlearn(UAMUnlearn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wmdp_module_name = str(
            getattr(self.uam_config, "module_name", "model.layers.7")
        )
        self.wmdp_noise_std = self._config_float(self.uam_config, "noise_std")
        seed = getattr(self.uam_config, "seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("WMDP UAM seed must be an integer.")
        self.wmdp_seed = seed
        self.require_paged_adamw = bool(
            getattr(self.uam_config, "require_paged_adamw", True)
        )
        self.wmdp_noise_generator = torch.Generator(device="cpu").manual_seed(seed)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        self.ref_model.eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

        configure_wmdp_trainable_parameters(self.model)
        self.model_module = find_exact_module(self.model, self.wmdp_module_name)
        self.ref_module = find_exact_module(self.ref_model, self.wmdp_module_name)
        self._wmdp_runtime_validated = False

    def _supported_forget_signals(self):
        return super()._supported_forget_signals() | {"representation"}

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

    def compute_uam_forget_signal(self, model, forget_inputs):
        activation, reference, outputs, mask = self._representation_pair(
            model,
            forget_inputs,
        )
        noise = seeded_gaussian_noise(
            activation.shape,
            std=self.wmdp_noise_std,
            generator=self.wmdp_noise_generator,
            device=activation.device,
            dtype=activation.dtype,
        )
        signal = masked_representation_mse(activation + noise, reference, mask)
        self._validate_finite_forget_signal(signal)
        return signal, outputs

    def compute_retain_loss(self, model, retain_inputs):
        activation, reference, _, mask = self._representation_pair(
            model,
            retain_inputs,
        )
        return masked_representation_mse(activation, reference, mask)

    def _decide_residual_gu_projection(
        self,
        residual_retain_dot,
        optimizer_retain_sq,
    ):
        if optimizer_retain_sq.item() == 0.0:
            return ResidualGUProjection(
                projection_coefficient=torch.zeros_like(residual_retain_dot),
                residual_lambda=self.residual_lambda,
            )
        return super()._decide_residual_gu_projection(
            residual_retain_dot,
            optimizer_retain_sq,
        )

    def _validate_uam_runtime(self):
        if self._wmdp_runtime_validated:
            return
        super()._validate_uam_runtime()

        if self.forget_signal != "representation":
            raise ValueError("WMDP UAM requires the representation forget signal.")
        if self.reflection_geometry != "euclidean":
            raise NotImplementedError("WMDP UAM requires Euclidean reflection.")
        if self.perturbation_normalization != "fixed_loss":
            raise NotImplementedError(
                "WMDP UAM requires fixed-loss perturbation normalization."
            )
        if self.rho != 5e-5:
            raise ValueError("WMDP UAM primary run requires rho=5e-5.")
        if self.reflection_gamma != 2.0:
            raise ValueError("WMDP UAM requires reflection_gamma=2.")
        if self.wmdp_noise_std != 0.01:
            raise ValueError("WMDP UAM requires noise_std=0.01.")
        if self.wmdp_module_name != "model.layers.7":
            raise ValueError("WMDP UAM requires module model.layers.7.")
        if not math.isfinite(self.wmdp_noise_std) or self.wmdp_noise_std <= 0.0:
            raise ValueError("WMDP UAM noise std must be positive and finite.")

        selected = self._selected_named_parameters()
        validate_wmdp_optimizer(self.optimizer, selected)
        if (
            self.require_paged_adamw
            and self._optimizer_geometry_adapter.name != "paged_adamw_32bit"
        ):
            raise NotImplementedError("WMDP UAM requires PagedAdamW32.")
        self._wmdp_runtime_validated = True
