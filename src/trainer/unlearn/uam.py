import copy
import math
from builtins import BaseExceptionGroup, ExceptionGroup
from dataclasses import dataclass

import torch

from trainer.unlearn.component_buffers import ComponentGradientBuffers
from trainer.unlearn.geometric import GeometricUnlearn
from trainer.unlearn.optimizer_geometry import make_optimizer_geometry_adapter
from trainer.unlearn.uam_geometry import (
    PerturbationDecision,
    apply_residual_gu_tensor,
    apply_uam_tensor,
    decide_perturbation,
    decide_residual_gu,
    decide_residual_gu_gate,
    decide_uam,
    project_residual_gu_tensor,
)
from trainer.utils import compute_batch_nll


class RetainReplayBuffer:
    _SUPPORTED_DEVICE_MODES = {"cpu", "parameter"}

    def __init__(self, device_mode):
        if device_mode not in self._SUPPORTED_DEVICE_MODES:
            raise ValueError(
                "Retain replay device mode must be 'cpu' or 'parameter', "
                f"got {device_mode!r}."
            )
        self.device_mode = device_mode
        self._batches = []

    @classmethod
    def _map_tensors(cls, value, function):
        if isinstance(value, dict):
            return {
                key: cls._map_tensors(item, function) for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._map_tensors(item, function) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._map_tensors(item, function) for item in value)
        if isinstance(value, torch.Tensor):
            return function(value)
        return copy.deepcopy(value)

    def append(self, inputs):
        target_device = torch.device("cpu") if self.device_mode == "cpu" else None

        def snapshot(tensor):
            device = tensor.device if target_device is None else target_device
            return tensor.detach().to(device).clone()

        with torch.no_grad():
            self._batches.append(self._map_tensors(inputs, snapshot))

    def batches(self, device):
        target_device = torch.device(device)
        for batch in self._batches:
            with torch.no_grad():
                replayed = self._map_tensors(
                    batch,
                    lambda tensor: tensor.detach().to(target_device).clone(),
                )
            yield replayed

    @property
    def empty(self):
        return not self._batches

    def __len__(self):
        return len(self._batches)

    def clear(self):
        self._batches.clear()


@dataclass
class PerturbationStats:
    requested_norm: float = 0.0
    effective_norm: float = 0.0

    @property
    def ratio(self):
        if self.requested_norm == 0.0:
            return 0.0
        return self.effective_norm / self.requested_norm


@dataclass(frozen=True)
class UAMPerturbation:
    deltas: dict[str, torch.Tensor]
    decision: PerturbationDecision
    raw_forget_norm: float
    optimizer_forget_norm: float
    identity_fallback_parameters: int


class TemporaryParameterPerturbation:
    _SUPPORTED_PARAMETER_DTYPES = {torch.float32, torch.bfloat16}

    def __init__(self, named_params, deltas, storage_device="cpu"):
        self.named_params = list(named_params)
        self.deltas = dict(deltas)
        self.storage_device = torch.device(storage_device)
        self.originals = {}
        self.stats = PerturbationStats()
        self._requested = {}
        self._mutated_names = []
        self._state = "new"

    def _validate_and_prepare(self):
        parameters_by_name = {}
        for name, parameter in self.named_params:
            if name in parameters_by_name:
                raise ValueError(f"Duplicate parameter name: {name!r}.")
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"UAM parameter {name!r} must be a tensor.")
            if parameter.dtype not in self._SUPPORTED_PARAMETER_DTYPES:
                raise TypeError(
                    f"UAM parameter {name!r} dtype {parameter.dtype} is unsupported; "
                    "expected torch.float32 or torch.bfloat16."
                )
            parameters_by_name[name] = parameter

        unknown_names = set(self.deltas).difference(parameters_by_name)
        if unknown_names:
            unknown = ", ".join(repr(name) for name in sorted(unknown_names))
            raise ValueError(f"UAM deltas contain unknown parameter names: {unknown}.")

        for name, parameter in self.named_params:
            if not torch.isfinite(parameter.detach()).all().item():
                raise ValueError(f"UAM parameter {name!r} must be finite.")

            if name not in self.deltas:
                continue
            delta = self.deltas[name]
            if not isinstance(delta, torch.Tensor):
                raise TypeError(f"UAM delta for parameter {name!r} must be a tensor.")
            if delta.dtype != torch.float32:
                raise TypeError(
                    f"UAM delta for parameter {name!r} dtype {delta.dtype} is "
                    "unsupported; expected a real torch.float32 tensor."
                )
            if delta.shape != parameter.shape:
                raise ValueError(
                    f"UAM delta shape for parameter {name!r} is {tuple(delta.shape)}, "
                    f"expected {tuple(parameter.shape)}."
                )
            if not torch.isfinite(delta.detach()).all().item():
                raise ValueError(f"UAM delta for parameter {name!r} must be finite.")

            requested = delta.detach().to(
                device=parameter.device,
                dtype=torch.float32,
            )
            if not torch.isfinite(requested).all().item():
                raise ValueError(
                    f"UAM FP32 delta for parameter {name!r} must be finite."
                )
            self._requested[name] = requested

        requested_norm = 0.0
        for requested in self._requested.values():
            requested_norm = math.hypot(
                requested_norm,
                float(
                    torch.linalg.vector_norm(
                        requested,
                        dtype=torch.float64,
                    ).item()
                ),
            )
        self.stats.requested_norm = requested_norm

    def _snapshot_originals(self):
        for name, parameter in self.named_params:
            if name in self._requested:
                self.originals[name] = (
                    parameter.detach().to(self.storage_device).clone()
                )

    @torch.no_grad()
    def _restore_originals(self):
        errors = []
        mutated_names = set(self._mutated_names)
        for name, parameter in self.named_params:
            if name not in mutated_names:
                continue
            original = self.originals[name]
            try:
                expected = original.to(parameter.device)
                parameter.copy_(expected)
                if not torch.equal(parameter.detach(), expected):
                    raise RuntimeError("restored value differs from snapshot")
            except Exception as error:
                errors.append((name, error))

        if errors:
            names = ", ".join(repr(name) for name, _ in errors)
            raise RuntimeError(
                f"UAM failed to restore parameter(s) {names} exactly."
            ) from errors[0][1]

    def _clear_snapshots(self):
        self.originals.clear()
        self._requested.clear()
        self._mutated_names.clear()

    @staticmethod
    def _combine_failures(primary_error, restoration_error):
        message = "UAM parameter perturbation failed and restoration also failed."
        if isinstance(primary_error, Exception) and isinstance(
            restoration_error,
            Exception,
        ):
            return ExceptionGroup(message, [primary_error, restoration_error])
        return BaseExceptionGroup(message, [primary_error, restoration_error])

    @torch.no_grad()
    def __enter__(self):
        if self._state == "active":
            raise RuntimeError("Temporary parameter perturbation is already active.")
        if self._state != "new":
            raise RuntimeError(
                "Temporary parameter perturbation cannot be reused after exit."
            )

        self._state = "active"
        try:
            self._validate_and_prepare()
            self._snapshot_originals()

            effective_norm = 0.0
            for name, parameter in self.named_params:
                requested = self._requested.get(name)
                if requested is None or not torch.count_nonzero(requested).item():
                    continue

                original_fp32 = parameter.detach().float()
                perturbed = (original_fp32 + requested).to(parameter.dtype)
                if not torch.isfinite(perturbed).all().item():
                    raise RuntimeError(
                        f"UAM perturbed parameter {name!r} is non-finite."
                    )
                effective = perturbed.float() - original_fp32
                if not torch.isfinite(effective).all().item():
                    raise RuntimeError(
                        f"UAM effective perturbation for parameter {name!r} "
                        "is non-finite."
                    )

                effective_norm = math.hypot(
                    effective_norm,
                    float(
                        torch.linalg.vector_norm(
                            effective,
                            dtype=torch.float64,
                        ).item()
                    ),
                )
                self._mutated_names.append(name)
                parameter.copy_(perturbed)

            self.stats.effective_norm = effective_norm
            if self.stats.requested_norm > 0.0 and effective_norm == 0.0:
                raise RuntimeError(
                    "UAM perturbation rounded entirely to zero; increase rho "
                    "or use FP32."
                )
            return self.stats
        except BaseException as enter_error:
            try:
                self._restore_originals()
            except BaseException as restoration_error:
                raise self._combine_failures(enter_error, restoration_error)
            finally:
                self._clear_snapshots()
                self._state = "used"
            raise

    @torch.no_grad()
    def __exit__(self, exc_type, exc_value, traceback):
        if self._state != "active":
            raise RuntimeError("Temporary parameter perturbation is not active.")
        try:
            self._restore_originals()
        except BaseException as restoration_error:
            if exc_value is not None:
                raise self._combine_failures(exc_value, restoration_error)
            raise
        finally:
            self._clear_snapshots()
            self._state = "used"
        return False


class UAMUnlearn(GeometricUnlearn):
    _FP64_REDUCTION_CHUNK_SIZE = 1_048_576

    @staticmethod
    def _config_float(config, field):
        value = getattr(config, field)
        if isinstance(value, bool):
            raise ValueError(f"UAM {field} must be numeric, not bool.")
        return float(value)

    @staticmethod
    def _chunked_fp64_dot(
        left,
        right=None,
        chunk_size=_FP64_REDUCTION_CHUNK_SIZE,
    ):
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("FP64 reduction chunk_size must be a positive integer.")
        if right is not None and right.shape != left.shape:
            raise ValueError("FP64 reduction tensors must have matching shapes.")

        left_flat = left.reshape(-1)
        right_flat = None if right is None else right.reshape(-1)
        total = torch.zeros((), dtype=torch.float64, device=left.device)
        for start in range(0, left_flat.numel(), chunk_size):
            stop = min(start + chunk_size, left_flat.numel())
            left_chunk = left_flat[start:stop].to(dtype=torch.float64)
            if right_flat is None:
                total.add_(left_chunk.square().sum())
            else:
                right_chunk = right_flat[start:stop].to(dtype=torch.float64)
                total.add_((left_chunk * right_chunk).sum())
        return total

    def __init__(self, *args, **kwargs):
        self.uam_config = kwargs.pop("uam_config")
        super().__init__(*args, **kwargs)

        self.uam_mode = str(self.uam_config.mode).lower()
        self.forget_signal = str(self.uam_config.forget_signal).lower()
        self.reflection_geometry = str(
            getattr(self.uam_config, "reflection_geometry", "optimizer")
        ).lower()
        normalization = str(self.uam_config.perturbation_normalization).lower()
        if normalization == "auto":
            normalization = "fixed_loss" if self.uam_mode == "uam" else "metric_trust"
        self.perturbation_normalization = normalization
        self.rho = self._config_float(self.uam_config, "rho")
        self.reflection_gamma = self._config_float(
            self.uam_config,
            "reflection_gamma",
        )
        self.residual_lambda = self._config_float(
            self.uam_config,
            "residual_lambda",
        )
        self.sign_tau = self._config_float(self.uam_config, "sign_tau")
        self.replay_device = str(self.uam_config.replay_device).lower()

        self.uam_calls = 0
        self.replay_calls = 0
        self.last_uam_diagnostics = {}
        self._uam_runtime_validated = False
        self._uam_microsteps = 0
        self.replay_buffer = RetainReplayBuffer(self.replay_device)

    def _validate_simnpo_signal_config(self):
        if self.loss_name != "simnpo":
            raise ValueError(
                "UAM SimNPO forget signal requires geometric_config.loss='simnpo'."
            )
        if self.simnpo_config is None:
            raise ValueError(
                "UAM SimNPO forget signal requires a non-null simnpo_config."
            )
        beta = getattr(self.simnpo_config, "beta", None)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise ValueError("UAM SimNPO beta must be positive and finite.")
        if beta <= 0.0 or not math.isfinite(beta):
            raise ValueError("UAM SimNPO beta must be positive and finite.")

        delta = getattr(self.simnpo_config, "delta", None)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError("UAM SimNPO delta must be finite.")
        if not math.isfinite(delta):
            raise ValueError("UAM SimNPO delta must be finite.")

    @staticmethod
    def _validate_finite_forget_signal(signal):
        if not signal.isfinite().item():
            raise RuntimeError("UAM forget signal is non-finite.")

    def compute_uam_forget_signal(self, model, forget_inputs):
        if self.forget_signal == "nll":
            sequence_nll, outputs = compute_batch_nll(model, forget_inputs)
            answer_counts = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
            if (answer_counts == 0).any():
                raise RuntimeError("UAM NLL forget signal has an empty answer mask.")
            answer_counts = answer_counts.to(sequence_nll)
            signal = (sequence_nll / answer_counts).mean()
            self._validate_finite_forget_signal(signal)
            return signal, outputs
        if self.forget_signal == "simnpo":
            self._validate_simnpo_signal_config()
            answer_counts = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
            if (answer_counts == 0).any():
                raise RuntimeError("UAM SimNPO forget signal has an empty answer mask.")
            simnpo_loss, outputs = self.compute_forget_loss(model, forget_inputs)
            signal = -simnpo_loss
            self._validate_finite_forget_signal(signal)
            return signal, outputs
        raise ValueError(f"Unsupported UAM forget signal: {self.forget_signal}")

    def _clear_uam_window(self, model=None, clear_grads=False):
        self.component_buffers.clear()
        self.replay_buffer.clear()
        self._uam_microsteps = 0
        if clear_grads:
            if model is None:
                raise ValueError("Clearing UAM gradients requires a model.")
            parameters = list(model.parameters())
            optimizer = getattr(self, "optimizer", None)
            if optimizer is not None:
                parameters.extend(
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                )
            for parameter in parameters:
                parameter.grad = None

    def _collect_uam_microstep(self, model, inputs):
        try:
            with self.compute_loss_context_manager():
                forget_signal, _ = self.compute_uam_forget_signal(
                    model,
                    inputs["forget"],
                )
                retain_loss = self.compute_retain_loss(model, inputs["retain"])

            named_params = self._selected_named_parameters(model)
            params = [parameter for _, parameter in named_params]
            forget_grads = torch.autograd.grad(
                forget_signal,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            retain_grads = torch.autograd.grad(
                retain_loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            self.component_buffers.add("forget", named_params, forget_grads)
            self.component_buffers.add("retain", named_params, retain_grads)
            del forget_grads, retain_grads
            self.replay_buffer.append(inputs["retain"])
            self._uam_microsteps += 1
            self.accelerator.backward(retain_loss)
        except BaseException:
            self._clear_uam_window(model, clear_grads=True)
            raise

        return retain_loss.detach()

    def training_step(self, model, inputs):
        self._validate_uam_runtime()
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        retain_loss = self._collect_uam_microstep(model, inputs)
        named_params = self._selected_named_parameters(model)
        if self.accelerator.sync_gradients or self._is_short_final_accumulation_step():
            self._finalize_uam_gradients(named_params)

        return retain_loss / self.args.gradient_accumulation_steps

    def _mean_component_coordinate(
        self,
        component,
        name,
        parameter,
        sqrt_denominator=None,
    ):
        if self._uam_microsteps <= 0:
            raise RuntimeError("UAM component mean requires at least one microstep.")
        tensor = self.component_buffers.tensor(
            component,
            name,
            parameter.device,
        )
        if tensor is None:
            return None
        if tensor.dtype != torch.float32:
            raise RuntimeError("UAM component buffers must use FP32.")
        effective_mean = tensor.float() / self._uam_microsteps
        return self._to_adam_coordinates(effective_mean, sqrt_denominator)

    def _reflection_component_coordinate(
        self,
        component,
        name,
        parameter,
        sqrt_denominator,
    ):
        if self.reflection_geometry == "optimizer":
            return self._mean_component_coordinate(
                component,
                name,
                parameter,
                sqrt_denominator,
            )
        return self._mean_component_coordinate(component, name, parameter)

    def _uam_in_optimizer_coordinates(self, uam, sqrt_denominator):
        if self.reflection_geometry == "optimizer":
            return uam
        return self._to_adam_coordinates(uam, sqrt_denominator)

    @torch.no_grad()
    def _build_uam_perturbation(self, named_params):
        named_params = list(named_params)
        if self._uam_microsteps <= 0:
            raise RuntimeError("UAM perturbation requires at least one microstep.")
        if not self.component_buffers.has_component("forget"):
            raise RuntimeError("UAM forget gradient buffer is empty.")
        if not named_params:
            raise RuntimeError("UAM perturbation requires selected parameters.")
        adapter = self._optimizer_geometry_adapter
        if adapter is None or not self._uam_runtime_validated:
            raise RuntimeError(
                "UAM perturbation requires an initialized validated optimizer adapter."
            )

        groups_by_parameter = adapter.groups_by_parameter()
        scalar_device = named_params[0][1].device
        raw_forget_sq = torch.zeros(
            (),
            dtype=torch.float32,
            device=scalar_device,
        )
        optimizer_forget_sq = torch.zeros_like(raw_forget_sq)
        identity_fallback_parameters = 0

        for name, parameter in named_params:
            raw_forget = self._mean_component_coordinate(
                "forget",
                name,
                parameter,
            )
            if raw_forget is None:
                continue
            group = groups_by_parameter[id(parameter)]
            sqrt_denominator = adapter.sqrt_denominator(parameter, group)
            if sqrt_denominator is None:
                identity_fallback_parameters += 1
            optimizer_forget = self._to_adam_coordinates(
                raw_forget,
                sqrt_denominator,
            )
            raw_forget_sq.add_(raw_forget.square().sum())
            optimizer_forget_sq.add_(optimizer_forget.square().sum())

        decision = decide_perturbation(
            self.perturbation_normalization,
            raw_forget_sq,
            optimizer_forget_sq,
            self.rho,
            self.projection_eps,
        )

        deltas = {}
        for name, parameter in named_params:
            raw_forget = self._mean_component_coordinate(
                "forget",
                name,
                parameter,
            )
            if raw_forget is None:
                continue
            coefficient = decision.coefficient.to(raw_forget.device)
            delta = coefficient * raw_forget
            if decision.mode == "metric_trust":
                group = groups_by_parameter[id(parameter)]
                sqrt_denominator = adapter.sqrt_denominator(parameter, group)
                if sqrt_denominator is not None:
                    delta = delta / sqrt_denominator.square()
            if not torch.isfinite(delta).all().item():
                raise RuntimeError(
                    f"UAM perturbation delta for parameter {name!r} is non-finite."
                )
            cpu_delta = delta.detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(cpu_delta).all().item():
                raise RuntimeError(
                    f"UAM CPU perturbation delta for parameter {name!r} "
                    "is non-finite."
                )
            deltas[name] = cpu_delta

        return UAMPerturbation(
            deltas=deltas,
            decision=decision,
            raw_forget_norm=float(raw_forget_sq.sqrt().item()),
            optimizer_forget_norm=float(optimizer_forget_sq.sqrt().item()),
            identity_fallback_parameters=identity_fallback_parameters,
        )

    def _replay_perturbed_retain_gradients(self, named_params, deltas):
        named_params = list(named_params)
        self.component_buffers.clear_component("perturbed_retain")
        replay_batch_count = len(self.replay_buffer)
        if self._uam_microsteps <= 0 or replay_batch_count != self._uam_microsteps:
            raise RuntimeError(
                "UAM replay batch count must match a positive microstep count."
            )
        if not named_params:
            raise RuntimeError("UAM replay requires selected parameters.")

        params = [parameter for _, parameter in named_params]
        try:
            with TemporaryParameterPerturbation(named_params, deltas) as stats:
                completed_batches = 0
                for retain_inputs in self.replay_buffer.batches(
                    self.accelerator.device
                ):
                    with torch.enable_grad():
                        with self.compute_loss_context_manager():
                            retain_loss = self.compute_retain_loss(
                                self.model,
                                retain_inputs,
                            )
                        retain_grads = torch.autograd.grad(
                            retain_loss,
                            params,
                            create_graph=False,
                            allow_unused=True,
                        )
                    buffered_retain_grads = tuple(
                        gradient
                        if gradient is not None
                        else torch.zeros_like(parameter)
                        for parameter, gradient in zip(
                            params,
                            retain_grads,
                            strict=True,
                        )
                    )
                    self.component_buffers.add(
                        "perturbed_retain",
                        named_params,
                        buffered_retain_grads,
                    )
                    del retain_grads, buffered_retain_grads
                    completed_batches += 1

                if not self.component_buffers.has_component("perturbed_retain"):
                    raise RuntimeError("UAM perturbed retain gradient buffer is empty.")
        except BaseException:
            self.component_buffers.clear_component("perturbed_retain")
            raise

        self.replay_calls += 1
        return stats, completed_batches

    @torch.no_grad()
    def _finalize_uam_gradients(self, named_params):
        perturbation = None
        uam_calls_at_entry = self.uam_calls
        replay_calls_at_entry = self.replay_calls
        last_uam_diagnostics_at_entry = self.last_uam_diagnostics
        last_surgery_diagnostics_at_entry = self.last_surgery_diagnostics
        try:
            named_params = list(named_params)
            if not self.component_buffers.has_component("forget"):
                raise RuntimeError("UAM forget gradient buffer is empty.")
            if not self.component_buffers.has_component("retain"):
                raise RuntimeError("UAM base retain gradient buffer is empty.")
            if not named_params:
                raise RuntimeError("UAM finalization requires selected parameters.")

            perturbation = self._build_uam_perturbation(named_params)
            perturbation_stats, replay_microsteps = (
                self._replay_perturbed_retain_gradients(
                    named_params,
                    perturbation.deltas,
                )
            )

            adapter = self._optimizer_geometry_adapter
            groups_by_parameter = adapter.groups_by_parameter()
            scalar_device = named_params[0][1].device
            forget_perturbed_retain_dot = torch.zeros(
                (),
                dtype=torch.float32,
                device=scalar_device,
            )
            optimizer_forget_sq = torch.zeros_like(forget_perturbed_retain_dot)
            reflection_forget_sq = torch.zeros_like(forget_perturbed_retain_dot)
            perturbed_retain_sq = torch.zeros_like(forget_perturbed_retain_dot)

            for name, parameter in named_params:
                group = groups_by_parameter[id(parameter)]
                sqrt_denominator = adapter.sqrt_denominator(parameter, group)
                forget = self._reflection_component_coordinate(
                    "forget",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                perturbed_retain = self._reflection_component_coordinate(
                    "perturbed_retain",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                optimizer_forget = self._mean_component_coordinate(
                    "forget",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                if forget is None:
                    forget = torch.zeros_like(parameter, dtype=torch.float32)
                if perturbed_retain is None:
                    perturbed_retain = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                if optimizer_forget is None:
                    optimizer_forget = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                forget_perturbed_retain_dot.add_((forget * perturbed_retain).sum())
                reflection_forget_sq.add_(forget.square().sum())
                optimizer_forget_sq.add_(optimizer_forget.square().sum())
                perturbed_retain_sq.add_(perturbed_retain.square().sum())

            decision = decide_uam(
                forget_perturbed_retain_dot,
                reflection_forget_sq,
                self.reflection_gamma,
                self.projection_eps,
            )

            scalar64 = torch.zeros((), dtype=torch.float64, device=scalar_device)
            residual_projection = None
            residual_decision = None
            diagnostic_retain_sq = torch.zeros_like(scalar64)
            residual_tangent_sq = torch.zeros_like(scalar64)
            residual_normal_sq = torch.zeros_like(scalar64)
            residual_normal_retain_dot = torch.zeros_like(scalar64)
            residual_normal_forget_dot = torch.zeros_like(scalar64)
            relative_residual_orthogonality = torch.zeros_like(scalar64)
            if self.uam_mode == "uam_gu":
                residual_retain_dot = torch.zeros_like(forget_perturbed_retain_dot)
                optimizer_retain_sq = torch.zeros_like(forget_perturbed_retain_dot)

                for name, parameter in named_params:
                    group = groups_by_parameter[id(parameter)]
                    sqrt_denominator = adapter.sqrt_denominator(parameter, group)
                    forget = self._reflection_component_coordinate(
                        "forget",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    retain = self._mean_component_coordinate(
                        "retain",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    perturbed_retain = self._reflection_component_coordinate(
                        "perturbed_retain",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    if forget is None:
                        forget = torch.zeros_like(parameter, dtype=torch.float32)
                    if retain is None:
                        retain = torch.zeros_like(parameter, dtype=torch.float32)
                    if perturbed_retain is None:
                        perturbed_retain = torch.zeros_like(
                            parameter,
                            dtype=torch.float32,
                        )
                    uam_reflection = apply_uam_tensor(
                        perturbed_retain,
                        forget,
                        decision,
                    )
                    uam = self._uam_in_optimizer_coordinates(
                        uam_reflection,
                        sqrt_denominator,
                    )
                    residual = uam - retain
                    residual_retain_dot.add_((residual * retain).sum())
                    optimizer_retain_sq.add_(retain.square().sum())

                residual_projection = decide_residual_gu(
                    residual_retain_dot,
                    optimizer_retain_sq,
                    self.residual_lambda,
                    self.projection_eps,
                )

                for name, parameter in named_params:
                    group = groups_by_parameter[id(parameter)]
                    sqrt_denominator = adapter.sqrt_denominator(parameter, group)
                    forget_reflection = self._reflection_component_coordinate(
                        "forget",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    forget = self._mean_component_coordinate(
                        "forget",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    retain = self._mean_component_coordinate(
                        "retain",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    perturbed_retain = self._reflection_component_coordinate(
                        "perturbed_retain",
                        name,
                        parameter,
                        sqrt_denominator,
                    )
                    if forget_reflection is None:
                        forget_reflection = torch.zeros_like(
                            parameter,
                            dtype=torch.float32,
                        )
                    if forget is None:
                        forget = torch.zeros_like(parameter, dtype=torch.float32)
                    if retain is None:
                        retain = torch.zeros_like(parameter, dtype=torch.float32)
                    if perturbed_retain is None:
                        perturbed_retain = torch.zeros_like(
                            parameter,
                            dtype=torch.float32,
                        )
                    uam_reflection = apply_uam_tensor(
                        perturbed_retain,
                        forget_reflection,
                        decision,
                    )
                    uam = self._uam_in_optimizer_coordinates(
                        uam_reflection,
                        sqrt_denominator,
                    )
                    normal = project_residual_gu_tensor(
                        uam,
                        retain,
                        residual_projection,
                    )
                    residual_normal_sq.add_(self._chunked_fp64_dot(normal))
                    residual_normal_forget_dot.add_(
                        self._chunked_fp64_dot(normal, forget)
                    )
                    residual_normal_retain_dot.add_(
                        self._chunked_fp64_dot(normal, retain)
                    )
                    diagnostic_retain_sq.add_(self._chunked_fp64_dot(retain))

                residual_tangent_sq.copy_(
                    residual_projection.projection_coefficient.to(
                        dtype=torch.float64
                    ).square()
                    * diagnostic_retain_sq
                )
                if residual_normal_sq.item() != 0.0:
                    relative_residual_orthogonality = (
                        residual_normal_retain_dot.abs()
                        / (diagnostic_retain_sq.sqrt() * residual_normal_sq.sqrt())
                    )
                residual_decision = decide_residual_gu_gate(
                    residual_projection,
                    residual_normal_forget_dot,
                    relative_residual_orthogonality,
                    self.sign_tau,
                )

            uam_sq = torch.zeros_like(scalar64)
            final_sq = torch.zeros_like(scalar64)
            forget_final_dot = torch.zeros_like(scalar64)
            retain_final_dot = torch.zeros_like(scalar64)

            for name, parameter in named_params:
                group = groups_by_parameter[id(parameter)]
                sqrt_denominator = adapter.sqrt_denominator(parameter, group)
                forget_reflection = self._reflection_component_coordinate(
                    "forget",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                perturbed_retain = self._reflection_component_coordinate(
                    "perturbed_retain",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                forget_optimizer = self._mean_component_coordinate(
                    "forget",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                retain_optimizer = self._mean_component_coordinate(
                    "retain",
                    name,
                    parameter,
                    sqrt_denominator,
                )
                if forget_reflection is None:
                    forget_reflection = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                if forget_optimizer is None:
                    forget_optimizer = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                if retain_optimizer is None:
                    retain_optimizer = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                if perturbed_retain is None:
                    perturbed_retain = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                uam_reflection = apply_uam_tensor(
                    perturbed_retain,
                    forget_reflection,
                    decision,
                )
                uam_optimizer = self._uam_in_optimizer_coordinates(
                    uam_reflection,
                    sqrt_denominator,
                )
                if residual_decision is None:
                    final_coordinates = uam_reflection
                    diagnostic_uam = uam_reflection
                    diagnostic_forget = forget_reflection
                    if self.reflection_geometry == "euclidean":
                        diagnostic_retain = self._mean_component_coordinate(
                            "retain",
                            name,
                            parameter,
                        )
                        if diagnostic_retain is None:
                            diagnostic_retain = torch.zeros_like(
                                parameter,
                                dtype=torch.float32,
                            )
                        raw = final_coordinates
                    else:
                        diagnostic_retain = retain_optimizer
                        raw = self._from_adam_coordinates(
                            final_coordinates,
                            sqrt_denominator,
                        )
                else:
                    final_coordinates, _ = apply_residual_gu_tensor(
                        uam_optimizer,
                        retain_optimizer,
                        residual_decision,
                    )
                    diagnostic_uam = uam_optimizer
                    diagnostic_forget = forget_optimizer
                    diagnostic_retain = retain_optimizer
                    raw = self._from_adam_coordinates(
                        final_coordinates,
                        sqrt_denominator,
                    )

                if residual_projection is None:
                    diagnostic_retain_sq.add_(
                        self._chunked_fp64_dot(diagnostic_retain)
                    )
                uam_sq.add_(self._chunked_fp64_dot(diagnostic_uam))
                final_sq.add_(self._chunked_fp64_dot(final_coordinates))
                forget_final_dot.add_(
                    self._chunked_fp64_dot(diagnostic_forget, final_coordinates)
                )
                retain_final_dot.add_(
                    self._chunked_fp64_dot(diagnostic_retain, final_coordinates)
                )
                if not torch.isfinite(raw).all().item():
                    raise RuntimeError(
                        f"UAM raw gradient for parameter {name!r} is non-finite."
                    )
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                converted = raw.to(
                    device=parameter.grad.device,
                    dtype=parameter.grad.dtype,
                )
                if not torch.isfinite(converted).all().item():
                    raise RuntimeError(
                        "UAM converted gradient for parameter "
                        f"{name!r} is non-finite."
                    )
                parameter.grad.copy_(converted)

            retain_norm = diagnostic_retain_sq.sqrt()
            residual_tangent_norm = residual_tangent_sq.sqrt()
            residual_normal_norm = residual_normal_sq.sqrt()
            residual_gate_kept = bool(
                residual_decision is not None and residual_decision.keep
            )
            residual_sign_gate_passed = bool(
                residual_decision is not None and residual_decision.sign_gate_passed
            )
            residual_orthogonality_safe = bool(
                residual_decision is None or residual_decision.orthogonality_safe
            )
            residual_projection_coefficient = (
                residual_decision.projection_coefficient
                if residual_decision is not None
                else torch.zeros_like(forget_perturbed_retain_dot)
            )
            residual_forget_gate_dot = (
                residual_normal_forget_dot
                if residual_decision is not None
                else torch.zeros_like(forget_perturbed_retain_dot)
            )
            predicted_forget_correction_derivative = (
                -self.residual_lambda * residual_normal_forget_dot
                if residual_gate_kept
                else torch.zeros_like(forget_perturbed_retain_dot)
            )
            update_step = uam_calls_at_entry + 1
            if self.replay_calls != update_step:
                raise RuntimeError(
                    "UAM replay counter must match the proposed update step."
                )
            diagnostics = {
                "record_type": "uam_geometry",
                "mode": self.uam_mode,
                "update_step": update_step,
                "forget_signal": self.forget_signal,
                "reflection_geometry": self.reflection_geometry,
                "perturbation_normalization": self.perturbation_normalization,
                "rho": self.rho,
                "requested_perturbation_norm": perturbation_stats.requested_norm,
                "effective_perturbation_norm": perturbation_stats.effective_norm,
                "effective_perturbation_ratio": (
                    perturbation_stats.effective_norm
                    / perturbation_stats.requested_norm
                    if perturbation_stats.requested_norm != 0.0
                    else 0.0
                ),
                "replay_microsteps": replay_microsteps,
                "perturbation_coefficient": float(
                    perturbation.decision.coefficient.item()
                ),
                "uam_projection_coefficient": float(decision.coefficient.item()),
                "residual_projection_coefficient": float(
                    residual_projection_coefficient.item()
                ),
                "optimizer_forget_norm": float(optimizer_forget_sq.sqrt().item()),
                "optimizer_retain_norm": float(retain_norm.item()),
                "reflection_forget_norm": float(
                    reflection_forget_sq.sqrt().item()
                ),
                "forget_norm": float(reflection_forget_sq.sqrt().item()),
                "retain_norm": float(retain_norm.item()),
                "perturbed_retain_norm": float(perturbed_retain_sq.sqrt().item()),
                "uam_coefficient": float(decision.coefficient.item()),
                "uam_coordinate_norm": float(uam_sq.sqrt().item()),
                "final_coordinate_norm": float(final_sq.sqrt().item()),
                "residual_tangent_norm": float(residual_tangent_norm.item()),
                "residual_normal_norm": float(residual_normal_norm.item()),
                "residual_sign_gate_passed": residual_sign_gate_passed,
                "residual_orthogonality_safe": residual_orthogonality_safe,
                "residual_gate_kept": residual_gate_kept,
                "relative_residual_orthogonality": float(
                    relative_residual_orthogonality.item()
                ),
                "residual_forget_gate_dot": float(residual_forget_gate_dot.item()),
                "predicted_retain_directional_derivative": float(
                    (-retain_final_dot).item()
                ),
                "predicted_forget_directional_derivative": float(
                    (-forget_final_dot).item()
                ),
                "predicted_forget_correction_derivative": float(
                    predicted_forget_correction_derivative.item()
                ),
                "identity_fallback_parameters": (
                    perturbation.identity_fallback_parameters
                ),
                "optimizer_geometry": adapter.name,
                "component_buffer_device": self.component_buffer_device,
                "uam_calls": update_step,
                "replay_calls": update_step,
            }
            string_keys = {
                "record_type",
                "mode",
                "forget_signal",
                "reflection_geometry",
                "perturbation_normalization",
                "optimizer_geometry",
                "component_buffer_device",
            }
            if not all(
                math.isfinite(float(value))
                for key, value in diagnostics.items()
                if key not in string_keys
            ):
                raise RuntimeError("UAM diagnostics contain non-finite scalars.")

            self.actual_delta_callback.prepare_step(
                update_step,
                named_params,
                self.component_buffers,
                component_scale=1.0 / self._uam_microsteps,
            )
            self._clear_uam_window()
            self.uam_calls = update_step
            self.replay_calls = update_step
            self.last_uam_diagnostics = diagnostics
            self.last_surgery_diagnostics = diagnostics.copy()
            if self.diagnostics_writer is not None:
                self.diagnostics_writer.write_step(diagnostics)
        except BaseException:
            self.actual_delta_callback.cancel_step()
            self.uam_calls = uam_calls_at_entry
            self.replay_calls = replay_calls_at_entry
            self.last_uam_diagnostics = last_uam_diagnostics_at_entry
            self.last_surgery_diagnostics = last_surgery_diagnostics_at_entry
            self._clear_uam_window(self.model, clear_grads=True)
            raise
        finally:
            if perturbation is not None:
                perturbation.deltas.clear()

    def _validate_uam_runtime(self):
        if self._uam_runtime_validated:
            return

        if self.uam_mode not in {"uam", "uam_gu"}:
            raise ValueError(f"Unsupported UAM mode: {self.uam_mode}")
        if self.forget_signal not in {"nll", "simnpo"}:
            raise ValueError(f"Unsupported UAM forget signal: {self.forget_signal}")
        if self.reflection_geometry not in {"optimizer", "euclidean"}:
            raise ValueError(
                "Unsupported UAM reflection geometry: "
                f"{self.reflection_geometry}"
            )
        if self.forget_signal == "simnpo":
            self._validate_simnpo_signal_config()
        if self.perturbation_normalization not in {"fixed_loss", "metric_trust"}:
            raise ValueError(
                "Unsupported UAM perturbation normalization: "
                f"{self.perturbation_normalization}"
            )
        if self.replay_device not in {"cpu", "parameter"}:
            raise ValueError(f"Unsupported UAM replay device: {self.replay_device}")
        if self.rho <= 0.0 or not math.isfinite(self.rho):
            raise ValueError("UAM rho must be positive and finite.")
        if not math.isfinite(self.reflection_gamma):
            raise ValueError("UAM reflection_gamma must be finite.")
        if self.residual_lambda < 0.0 or not math.isfinite(self.residual_lambda):
            raise ValueError("UAM residual_lambda must be non-negative and finite.")
        if self.sign_tau < 0.0 or not math.isfinite(self.sign_tau):
            raise ValueError("UAM sign_tau must be non-negative and finite.")
        if self.projection_eps <= 0.0 or not math.isfinite(self.projection_eps):
            raise ValueError("UAM projection_eps must be positive and finite.")
        if str(self.retain_loss_type).upper() != "NLL":
            raise NotImplementedError("UAM v1 requires NLL retain loss.")

        if self.is_deepspeed_enabled:
            raise NotImplementedError(
                "Approximate Adam UAM does not support DeepSpeed."
            )
        if self.is_fsdp_enabled:
            raise NotImplementedError("Approximate Adam UAM does not support FSDP.")
        if self.args.fp16:
            raise NotImplementedError("Approximate Adam UAM supports BF16/FP32 only.")
        if self.use_apex:
            raise NotImplementedError("Approximate Adam UAM does not support Apex.")
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise NotImplementedError(
                "Approximate Adam UAM supports one process and one GPU only."
            )
        if self.args.gradient_checkpointing:
            checkpointing_kwargs = self.args.gradient_checkpointing_kwargs or {}
            if checkpointing_kwargs.get("use_reentrant") is not False:
                raise NotImplementedError(
                    "Approximate Adam UAM requires use_reentrant=false."
                )

        named_params = self._selected_named_parameters()
        if not named_params:
            raise ValueError("UAM requires at least one selected trainable parameter.")

        adapter = make_optimizer_geometry_adapter(self.optimizer)
        adapter.validate(named_params)

        if self.component_buffer_device == "cpu":
            ComponentGradientBuffers.validate_host_memory(
                sum(parameter.numel() for _, parameter in named_params),
                component_count=3,
            )

        self._optimizer_geometry_adapter = adapter
        self._uam_runtime_validated = True
