import math
import os

import torch


class ComponentGradientBuffers:
    _COMPONENTS = {"forget", "retain", "perturbed_retain"}

    def __init__(self, device_mode: str, pin_memory: bool = True):
        if device_mode not in {"cpu", "parameter"}:
            raise ValueError(f"Unsupported component buffer device: {device_mode}")
        self.device_mode = device_mode
        self.pin_memory = bool(pin_memory)
        self._data = {"forget": {}, "retain": {}, "perturbed_retain": {}}

    @property
    def empty(self) -> bool:
        return not any(self._data.values())

    @classmethod
    def _validate_component(cls, component: str) -> None:
        if component not in cls._COMPONENTS:
            raise ValueError(f"Unsupported gradient component: {component}")

    def _allocate(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.device_mode == "parameter":
            return torch.zeros_like(
                gradient,
                dtype=torch.float32,
                memory_format=torch.preserve_format,
            )
        return torch.zeros(
            gradient.shape,
            dtype=torch.float32,
            device="cpu",
            pin_memory=self.pin_memory and torch.cuda.is_available(),
        )

    @torch.no_grad()
    def add(self, component, named_params, gradients) -> None:
        self._validate_component(component)
        named_params = list(named_params)
        gradients = list(gradients)
        if len(named_params) != len(gradients):
            raise ValueError(
                "Named parameters and gradients must have the same length."
            )

        for (name, _), gradient in zip(named_params, gradients):
            if gradient is None:
                continue
            if name not in self._data[component]:
                self._data[component][name] = self._allocate(gradient)
            storage = self._data[component][name]
            if storage.shape != gradient.shape:
                raise RuntimeError(
                    f"Gradient shape for {name} changed during accumulation."
                )
            copied = gradient.detach().to(
                device=storage.device,
                dtype=torch.float32,
                non_blocking=False,
            )
            storage.add_(copied)

    def tensor(self, component, name, device):
        self._validate_component(component)
        tensor = self._data[component].get(name)
        if tensor is None:
            return None
        target_device = torch.device(device)
        if tensor.device == target_device:
            return tensor
        return tensor.to(
            device=target_device,
            dtype=torch.float32,
            non_blocking=tensor.device.type == "cpu" and tensor.is_pinned(),
        )

    def has_component(self, component) -> bool:
        self._validate_component(component)
        return bool(self._data[component])

    def clear_component(self, component) -> None:
        self._validate_component(component)
        self._data[component].clear()

    def clear(self) -> None:
        for component in self._COMPONENTS:
            self._data[component].clear()

    @staticmethod
    def required_host_bytes(
        selected_numel: int,
        headroom: float = 1.2,
        component_count: int = 2,
    ) -> int:
        if selected_numel < 0:
            raise ValueError("selected_numel must be non-negative.")
        if headroom < 1.0 or not math.isfinite(headroom):
            raise ValueError("headroom must be finite and at least 1.0.")
        if component_count <= 0:
            raise ValueError("component_count must be positive.")
        return math.ceil(
            component_count
            * selected_numel
            * torch.finfo(torch.float32).bits
            / 8
            * headroom
        )

    @classmethod
    def validate_host_memory(
        cls,
        selected_numel: int,
        headroom: float = 1.2,
        available_bytes: int | None = None,
        component_count: int = 2,
    ) -> int:
        required = cls.required_host_bytes(
            selected_numel,
            headroom,
            component_count,
        )
        if available_bytes is None:
            available_bytes = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if available_bytes < required:
            raise RuntimeError(
                "Insufficient host memory for GU component buffers: "
                f"requires {required} bytes, found {available_bytes} bytes."
            )
        return required
