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
        self.rmu_finalizer_calls = 0
        self.last_rmu_diagnostics = {}

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

    @torch.no_grad()
    def _finalize_rmu_baseline(self, named_params):
        update_step = self.rmu_finalizer_calls + 1
        if not self.component_buffers.has_component("forget"):
            raise RuntimeError("RMU forget gradient buffer is empty.")
        if not self.component_buffers.has_component("retain"):
            raise RuntimeError("RMU retain gradient buffer is empty.")

        forget_sq = torch.zeros((), dtype=torch.float64)
        retain_sq = torch.zeros_like(forget_sq)
        final_sq = torch.zeros_like(forget_sq)
        try:
            for name, parameter in named_params:
                forget = self.component_buffers.tensor(
                    "forget",
                    name,
                    parameter.device,
                )
                retain = self.component_buffers.tensor(
                    "retain",
                    name,
                    parameter.device,
                )
                if forget is None:
                    forget = torch.zeros_like(parameter, dtype=torch.float32)
                if retain is None:
                    retain = torch.zeros_like(parameter, dtype=torch.float32)
                final = self.gamma * forget.float() + self.alpha * retain.float()
                if not torch.isfinite(final).all().item():
                    raise RuntimeError("RMU baseline final gradient is non-finite.")
                forget_sq.add_(forget.double().square().sum().cpu())
                retain_sq.add_(retain.double().square().sum().cpu())
                final_sq.add_(final.double().square().sum().cpu())
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.copy_(final.to(parameter.grad))

            diagnostics = {
                "record_type": "rmu_geometry",
                "update_step": update_step,
                "finalizer_calls": update_step,
                "forget_norm": float(forget_sq.sqrt().item()),
                "retain_norm": float(retain_sq.sqrt().item()),
                "final_gradient_norm": float(final_sq.sqrt().item()),
                "optimizer_geometry": self._optimizer_geometry_adapter.name,
            }
            self.actual_delta_callback.prepare_step(
                update_step,
                named_params,
                self.component_buffers,
            )
            if self.diagnostics_writer is not None:
                self.diagnostics_writer.write_step(diagnostics)
            self.component_buffers.clear()
            self.rmu_finalizer_calls = update_step
            self.last_rmu_diagnostics = diagnostics
        except BaseException:
            self.actual_delta_callback.cancel_step()
            self._clear_gu_failure_state()
            raise

    def training_step(self, model, inputs):
        self._validate_wmdp_runtime()
        if self.gu_enabled:
            return super().training_step(model, inputs)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            forget_loss, retain_loss, _ = self.compute_component_losses(model, inputs)
            total_loss = self.gamma * forget_loss + self.alpha * retain_loss

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
        self.component_buffers.add("forget", named_params, forget_grads)
        self.component_buffers.add("retain", named_params, retain_grads)
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients or self._is_short_final_accumulation_step():
            self._finalize_rmu_baseline(named_params)
        return total_loss.detach() * scale
