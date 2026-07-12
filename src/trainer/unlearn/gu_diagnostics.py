import hashlib
import json
import math
import os
import tempfile
from builtins import BaseExceptionGroup, ExceptionGroup
from numbers import Real
from pathlib import Path

import torch
from transformers import TrainerCallback

from trainer.unlearn.component_buffers import ComponentGradientBuffers


def _present_values(records, key):
    return [record[key] for record in records if record.get(key) is not None]


def aggregate_surgery_records(records):
    geometry_records = [
        record
        for record in records
        if record.get("record_type", "geometry") in {"geometry", "uam_geometry"}
    ]
    conflicts = _present_values(geometry_records, "conflict")
    residuals = _present_values(
        geometry_records,
        "relative_orthogonality_residual",
    )
    magnitudes = _present_values(
        geometry_records,
        "relative_surgery_magnitude",
    )
    distances = _present_values(
        geometry_records,
        "relative_pcgrad_gu_distance",
    )
    perturbation_ratios = _present_values(
        geometry_records,
        "effective_perturbation_ratio",
    )
    residual_gates = _present_values(geometry_records, "residual_gate_kept")

    return {
        "surgery_count": len(geometry_records),
        "conflict_rate": (
            sum(bool(value) for value in conflicts) / len(conflicts)
            if conflicts
            else None
        ),
        "maximum_relative_orthogonality_residual": (
            max(residuals) if residuals else None
        ),
        "mean_relative_surgery_magnitude": (
            sum(magnitudes) / len(magnitudes) if magnitudes else None
        ),
        "mean_relative_pcgrad_gu_distance": (
            sum(distances) / len(distances) if distances else None
        ),
        "pcgrad_gu_distance_count": len(distances),
        "mean_effective_perturbation_ratio": (
            sum(perturbation_ratios) / len(perturbation_ratios)
            if perturbation_ratios
            else None
        ),
        "residual_gate_rate": (
            sum(bool(value) for value in residual_gates) / len(residual_gates)
            if residual_gates
            else None
        ),
    }


class SurgeryDiagnosticsWriter:
    def __init__(self, path, summary_path=None):
        self.path = Path(path)
        self.summary_path = (
            Path(summary_path)
            if summary_path is not None
            else self.path.with_suffix(".summary.json")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def write_step(self, record):
        encoded_line = (
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with self.path.open("a+b", buffering=0) as handle:
            handle.seek(0, os.SEEK_END)
            original_length = handle.tell()
            try:
                written = handle.write(encoded_line)
                if written != len(encoded_line):
                    raise OSError(
                        "Diagnostics JSONL append wrote fewer bytes than expected."
                    )
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException as write_error:
                try:
                    os.ftruncate(handle.fileno(), original_length)
                except BaseException as rollback_error:
                    message = "Diagnostics append and byte rollback both failed."
                    if isinstance(write_error, Exception) and isinstance(
                        rollback_error,
                        Exception,
                    ):
                        raise ExceptionGroup(
                            message,
                            [write_error, rollback_error],
                        )
                    raise BaseExceptionGroup(
                        message,
                        [write_error, rollback_error],
                    )
                raise

    def read_records(self):
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Malformed diagnostics JSONL at line {line_number}."
                    ) from error
        return records

    def write_summary(self):
        summary = aggregate_surgery_records(self.read_records())
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.summary_path.name}.",
            suffix=".tmp",
            dir=self.summary_path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.summary_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return summary


class ActualDeltaCallback(TrainerCallback):
    def __init__(
        self,
        mode="off",
        steps=(1, 10),
        sample_elements=1_000_000,
        writer=None,
    ):
        if mode not in {"off", "full", "sampled"}:
            raise ValueError(f"Unsupported actual delta mode: {mode}")
        if sample_elements <= 0:
            raise ValueError("actual delta sample_elements must be positive.")
        self.mode = mode
        self.steps = {int(step) for step in steps}
        self.sample_elements = int(sample_elements)
        self.writer = writer
        self._active_step = None
        self._active_parameters = {}
        self._component_probes = {}
        self._snapshots = {}
        self._pending_record = None
        self.last_record = None

    @staticmethod
    def _deterministic_global_indices(named_params, count):
        total = sum(parameter.numel() for _, parameter in named_params)
        count = min(int(count), total)
        if count == 0:
            return {}

        manifest = "|".join(
            f"{name}:{tuple(parameter.shape)}:{parameter.numel()}"
            for name, parameter in named_params
        )
        start_digest = hashlib.sha256(f"start:{manifest}".encode()).digest()
        step_digest = hashlib.sha256(f"step:{manifest}".encode()).digest()
        start = int.from_bytes(start_digest[:8], "big") % total
        stride = int.from_bytes(step_digest[:8], "big") % total
        stride = max(stride, 1)
        while math.gcd(stride, total) != 1:
            stride += 1
            if stride == total:
                stride = 1

        global_indices = (
            start + torch.arange(count, dtype=torch.int64) * stride
        ).remainder(total)
        global_indices = global_indices.sort().values
        result = {}
        offset = 0
        for name, parameter in named_params:
            end = offset + parameter.numel()
            selected = global_indices[
                (global_indices >= offset) & (global_indices < end)
            ]
            if selected.numel():
                result[name] = selected - offset
            offset = end
        return result

    @torch.no_grad()
    def prepare_step(
        self,
        update_step,
        named_params,
        component_buffers,
        component_scale=1.0,
    ):
        self.cancel_step()
        if self.mode == "off" or int(update_step) not in self.steps:
            return
        if (
            isinstance(component_scale, bool)
            or not isinstance(component_scale, Real)
            or not math.isfinite(component_scale)
            or component_scale <= 0.0
        ):
            raise ValueError(
                "actual delta component_scale must be a positive finite real."
            )
        component_scale = float(component_scale)

        try:
            named_params = list(named_params)
            if self.mode == "full":
                ComponentGradientBuffers.validate_host_memory(
                    sum(parameter.numel() for _, parameter in named_params),
                    component_count=3,
                )
            self._active_step = int(update_step)
            self._active_parameters = dict(named_params)
            indices_by_name = (
                self._deterministic_global_indices(
                    named_params,
                    self.sample_elements,
                )
                if self.mode == "sampled"
                else {name: None for name, _ in named_params}
            )

            for name, parameter in named_params:
                if name not in indices_by_name:
                    continue
                indices = indices_by_name[name]
                if indices is not None:
                    forget = self._sample_component_to_cpu(
                        component_buffers,
                        "forget",
                        name,
                        indices,
                    )
                    retain = self._sample_component_to_cpu(
                        component_buffers,
                        "retain",
                        name,
                        indices,
                    )
                else:
                    forget = component_buffers.tensor(
                        "forget",
                        name,
                        torch.device("cpu"),
                    )
                    retain = component_buffers.tensor(
                        "retain",
                        name,
                        torch.device("cpu"),
                    )
                    if forget is None:
                        forget = torch.zeros(parameter.numel(), dtype=torch.float32)
                    else:
                        forget = forget.reshape(-1).clone()
                    if retain is None:
                        retain = torch.zeros(parameter.numel(), dtype=torch.float32)
                    else:
                        retain = retain.reshape(-1).clone()
                self._component_probes[name] = {
                    "indices": indices,
                    "forget": forget.mul_(component_scale),
                    "retain": retain.mul_(component_scale),
                }
        except BaseException:
            self.cancel_step()
            raise

    @staticmethod
    def _to_cpu_probe(tensor):
        return tensor.detach().to(device="cpu", dtype=torch.float32).clone()

    def _sample_component_to_cpu(
        self,
        component_buffers,
        component,
        name,
        indices,
    ):
        storage = component_buffers.storage_tensor(component, name)
        if storage is None:
            return torch.zeros(indices.numel(), dtype=torch.float32)
        selected = storage.reshape(-1).index_select(
            0,
            indices.to(storage.device),
        )
        return self._to_cpu_probe(selected)

    @torch.no_grad()
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._active_step is None:
            return control
        if self._active_step != int(state.global_step) + 1:
            return control

        self._snapshots.clear()
        for name, probe in self._component_probes.items():
            parameter = self._active_parameters[name]
            current = parameter.detach().reshape(-1)
            indices = probe["indices"]
            if indices is not None:
                current = current.index_select(0, indices.to(current.device))
            self._snapshots[name] = current.float().cpu().clone()
        return control

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._active_step is None:
            return control
        if self._active_step != int(state.global_step) + 1:
            return control

        forget_dot = 0.0
        retain_dot = 0.0
        forget_sq = 0.0
        retain_sq = 0.0
        delta_sq = 0.0
        sampled_elements = 0
        try:
            for name, probe in self._component_probes.items():
                parameter = self._active_parameters[name]
                current = parameter.detach().reshape(-1)
                indices = probe["indices"]
                if indices is not None:
                    current = current.index_select(0, indices.to(current.device))
                delta = current.float().cpu() - self._snapshots[name]
                forget = probe["forget"]
                retain = probe["retain"]
                forget_dot += float((forget * delta).sum().item())
                retain_dot += float((retain * delta).sum().item())
                forget_sq += float(forget.square().sum().item())
                retain_sq += float(retain.square().sum().item())
                delta_sq += float(delta.square().sum().item())
                sampled_elements += delta.numel()

            record = {
                "record_type": "actual_delta",
                "update_step": self._active_step,
                "coverage": self.mode,
                "sampled_elements": sampled_elements,
                "forget_directional_derivative": forget_dot,
                "retain_directional_derivative": retain_dot,
                "forget_gradient_norm": math.sqrt(forget_sq),
                "retain_gradient_norm": math.sqrt(retain_sq),
                "parameter_delta_norm": math.sqrt(delta_sq),
            }
            self.last_record = record
            if self.writer is not None:
                self._pending_record = record
        finally:
            self._clear_active()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._pending_record is None:
            return control
        try:
            if self._pending_record["update_step"] != int(state.global_step):
                raise RuntimeError(
                    "Pending actual delta step does not match committed global step."
                )
            self.writer.write_step(self._pending_record)
        finally:
            self._pending_record = None
        return control

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if self.writer is not None:
                self.writer.write_summary()
        finally:
            self.cancel_step()
        return control

    def cancel_step(self):
        """Discard a prepared measurement without emitting a record."""
        self._clear_active()
        self._pending_record = None

    def _clear_active(self):
        self._active_step = None
        self._active_parameters.clear()
        self._component_probes.clear()
        self._snapshots.clear()
