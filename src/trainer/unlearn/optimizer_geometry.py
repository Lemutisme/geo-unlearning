import math

import torch


def unwrap_optimizer(optimizer):
    visited = set()
    while hasattr(optimizer, "optimizer") and id(optimizer) not in visited:
        visited.add(id(optimizer))
        optimizer = optimizer.optimizer
    return optimizer


class OptimizerGeometryAdapter:
    name = "base"
    second_moment_key = ""

    def __init__(self, optimizer):
        self.optimizer = unwrap_optimizer(optimizer)

    def groups_by_parameter(self):
        return {
            id(parameter): group
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }

    def validate(self, named_params):
        groups_by_parameter = self.groups_by_parameter()
        missing = [
            name
            for name, parameter in named_params
            if id(parameter) not in groups_by_parameter
        ]
        if missing:
            raise ValueError(
                "Selected parameters are absent from the optimizer: "
                + ", ".join(missing[:3])
            )

        for group in self.optimizer.param_groups:
            beta1 = float(group["betas"][0])
            if beta1 != 0.0:
                raise NotImplementedError(
                    "Approximate Adam gradient surgery requires beta1=0."
                )
            if float(group.get("weight_decay", 0.0)) != 0.0:
                raise NotImplementedError(
                    "Approximate Adam gradient surgery requires weight_decay=0."
                )
            if bool(group.get("amsgrad", False)):
                raise NotImplementedError(
                    "Approximate Adam gradient surgery does not support AMSGrad."
                )
            eps = float(group["eps"])
            if eps <= 0.0 or not math.isfinite(eps):
                raise ValueError(
                    "Approximate Adam gradient surgery requires positive finite eps."
                )

    def prefetch(self, parameter):
        return None

    @staticmethod
    def _step_value(step):
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all():
                raise RuntimeError("Initialized Adam state has an invalid step.")
            return int(step.item())
        try:
            return int(step)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("Initialized Adam state has an invalid step.") from error

    def _initialized_second_moment(self, parameter):
        state = self.optimizer.state.get(parameter, {})
        if not state:
            return None, None
        if "step" not in state:
            raise RuntimeError("Initialized Adam state is missing step.")

        second_moment = state.get(self.second_moment_key)
        if second_moment is None:
            raise RuntimeError(
                f"Initialized Adam state is missing {self.second_moment_key}."
            )
        step = self._step_value(state["step"])
        if step <= 0:
            raise RuntimeError("Initialized Adam state has a non-positive step.")
        if second_moment.shape != parameter.shape:
            raise RuntimeError(
                f"Adam {self.second_moment_key} shape does not match parameter."
            )
        if not torch.isfinite(second_moment).all():
            raise RuntimeError(
                f"Adam {self.second_moment_key} contains non-finite values."
            )
        if (second_moment < 0).any():
            raise RuntimeError(
                f"Adam {self.second_moment_key} contains negative values."
            )
        return second_moment, step

    def sqrt_denominator(self, parameter, group):
        state = self.optimizer.state.get(parameter, {})
        if not state:
            return None
        self.prefetch(parameter)
        second_moment, step = self._initialized_second_moment(parameter)

        beta2 = float(group["betas"][1])
        if beta2 < 0.0 or beta2 >= 1.0 or not math.isfinite(beta2):
            raise ValueError("Adam beta2 must be finite and in [0, 1).")
        bias_correction = 1.0 - beta2**step
        v_hat = (
            second_moment.to(
                device=parameter.device,
                dtype=torch.float32,
            )
            / bias_correction
        )
        sqrt_denominator = (v_hat.sqrt() + float(group["eps"])).sqrt()
        if not torch.isfinite(sqrt_denominator).all():
            raise RuntimeError("Adam square-root denominator is non-finite.")
        return sqrt_denominator


class TorchAdamGeometryAdapter(OptimizerGeometryAdapter):
    name = "torch_adam"
    second_moment_key = "exp_avg_sq"


class PagedAdamW32GeometryAdapter(OptimizerGeometryAdapter):
    name = "paged_adamw_32bit"
    second_moment_key = "state2"

    def prefetch(self, parameter):
        prefetch_state = getattr(self.optimizer, "prefetch_state", None)
        if callable(prefetch_state):
            prefetch_state(parameter)

    def _initialized_second_moment(self, parameter):
        state = self.optimizer.state.get(parameter, {})
        state2 = state.get("state2") if state else None
        if state2 is not None and state2.dtype == torch.uint8:
            raise NotImplementedError(
                "PagedAdamW8 optimizer state is unsupported; use PagedAdamW32."
            )
        if state2 is not None and state2.dtype != torch.float32:
            raise NotImplementedError(
                f"PagedAdamW32 requires FP32 state2; received {state2.dtype}."
            )
        return super()._initialized_second_moment(parameter)


def make_optimizer_geometry_adapter(optimizer):
    optimizer = unwrap_optimizer(optimizer)
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return TorchAdamGeometryAdapter(optimizer)

    module = optimizer.__class__.__module__
    optim_bits = getattr(getattr(optimizer, "args", None), "optim_bits", None)
    if (
        module.startswith("bitsandbytes.optim.adamw")
        and bool(getattr(optimizer, "is_paged", False))
        and int(optim_bits or 0) == 32
    ):
        return PagedAdamW32GeometryAdapter(optimizer)

    class_name = optimizer.__class__.__name__
    raise NotImplementedError(
        "Approximate Adam gradient surgery does not support optimizer " f"{class_name}."
    )
