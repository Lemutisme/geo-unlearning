import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch
from transformers import TrainerCallback


def _present_values(records, key):
    return [record[key] for record in records if record.get(key) is not None]


def aggregate_surgery_records(records):
    geometry_records = [
        record
        for record in records
        if record.get("record_type", "geometry") == "geometry"
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
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()

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
    def prepare_step(self, update_step, named_params, component_buffers):
        if self.mode == "off" or int(update_step) not in self.steps:
            return
        self._clear_active()
        named_params = list(named_params)
        self._active_step = int(update_step)
        self._active_parameters = dict(named_params)
        indices_by_name = (
            self._deterministic_global_indices(named_params, self.sample_elements)
            if self.mode == "sampled"
            else {name: None for name, _ in named_params}
        )

        for name, parameter in named_params:
            if name not in indices_by_name:
                continue
            indices = indices_by_name[name]
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
                forget = torch.zeros(parameter.shape, dtype=torch.float32)
            else:
                forget = forget.detach().float().cpu()
            if retain is None:
                retain = torch.zeros(parameter.shape, dtype=torch.float32)
            else:
                retain = retain.detach().float().cpu()

            if indices is not None:
                forget = forget.reshape(-1).index_select(0, indices)
                retain = retain.reshape(-1).index_select(0, indices)
            else:
                forget = forget.reshape(-1).clone()
                retain = retain.reshape(-1).clone()
            self._component_probes[name] = {
                "indices": indices,
                "forget": forget,
                "retain": retain,
            }

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
                self.writer.write_step(record)
        finally:
            self._clear_active()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.write_summary()
        self._clear_active()
        return control

    def _clear_active(self):
        self._active_step = None
        self._active_parameters.clear()
        self._component_probes.clear()
        self._snapshots.clear()
