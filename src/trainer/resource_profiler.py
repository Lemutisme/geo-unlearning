from collections import defaultdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

import torch
from transformers import TrainerCallback


def deterministic_quantile(values, probability):
    values = sorted(float(value) for value in values)
    probability = float(probability)
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    if probability < 0.0 or probability > 1.0 or not math.isfinite(probability):
        raise ValueError("Quantile probability must be finite and in [0, 1].")
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_resource_profile(
    step_ms,
    phase_ms,
    train_wall_seconds,
    cuda_max_allocated_bytes,
    cuda_max_reserved_bytes,
):
    step_ms = [float(value) for value in step_ms]
    if not step_ms:
        raise ValueError("Resource profile requires at least one optimizer update.")
    if cuda_max_reserved_bytes < cuda_max_allocated_bytes:
        raise ValueError("CUDA reserved bytes cannot be below allocated bytes.")
    steady = step_ms[1:] or step_ms
    phase_totals = {
        name: math.fsum(float(value) for value in values)
        for name, values in sorted(phase_ms.items())
    }
    return {
        "record_type": "training_resource_profile",
        "update_count": len(step_ms),
        "train_wall_seconds": float(train_wall_seconds),
        "first_update_ms": step_ms[0],
        "steady_update_mean_ms": statistics.fmean(steady),
        "steady_update_p50_ms": deterministic_quantile(steady, 0.50),
        "steady_update_p95_ms": deterministic_quantile(steady, 0.95),
        "cuda_max_allocated_bytes": int(cuda_max_allocated_bytes),
        "cuda_max_reserved_bytes": int(cuda_max_reserved_bytes),
        "phase_totals_ms": phase_totals,
        "phase_counts": {
            name: len(values) for name, values in sorted(phase_ms.items())
        },
    }


class ResourceProfileCallback(TrainerCallback):
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.enabled = self.path is not None
        self.step_ms = []
        self.phase_ms = defaultdict(list)
        self.train_wall_seconds = 0.0
        self.cuda_max_allocated_bytes = 0
        self.cuda_max_reserved_bytes = 0
        self._train_start_ns = None
        self._step_start_ns = None
        self._cuda_enabled = False
        self._pending_cuda_phases = []

    @staticmethod
    def _synchronize(enabled):
        if enabled:
            torch.cuda.synchronize()

    def on_train_begin(self, args, state, control, **kwargs):
        del state, kwargs
        if not self.enabled:
            return control
        self.step_ms.clear()
        self.phase_ms.clear()
        self._pending_cuda_phases.clear()
        self._cuda_enabled = torch.cuda.is_available() and not bool(args.use_cpu)
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats()
        self._synchronize(self._cuda_enabled)
        self._train_start_ns = time.perf_counter_ns()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        if not self.enabled:
            return control
        self._synchronize(self._cuda_enabled)
        self._step_start_ns = time.perf_counter_ns()
        return control

    def _resolve_cuda_phases(self):
        for name, start, end in self._pending_cuda_phases:
            self.phase_ms[name].append(float(start.elapsed_time(end)))
        self._pending_cuda_phases.clear()

    def on_step_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        if not self.enabled:
            return control
        self._synchronize(self._cuda_enabled)
        if self._cuda_enabled:
            self._resolve_cuda_phases()
        if self._step_start_ns is None:
            raise RuntimeError("Resource profiler step ended without a start.")
        elapsed_ns = time.perf_counter_ns() - self._step_start_ns
        self.step_ms.append(elapsed_ns / 1_000_000.0)
        self._step_start_ns = None
        return control

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self._cuda_enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending_cuda_phases.append((str(name), start, end))
            return

        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            self.phase_ms[str(name)].append(elapsed_ms)

    def on_train_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        if not self.enabled:
            return control
        self._synchronize(self._cuda_enabled)
        if self._cuda_enabled:
            self._resolve_cuda_phases()
            self.cuda_max_allocated_bytes = torch.cuda.max_memory_allocated()
            self.cuda_max_reserved_bytes = torch.cuda.max_memory_reserved()
        if self._train_start_ns is None:
            raise RuntimeError("Resource profiler training ended without a start.")
        self.train_wall_seconds = (
            time.perf_counter_ns() - self._train_start_ns
        ) / 1_000_000_000.0
        self.write_profile()
        return control

    def write_profile(self):
        record = summarize_resource_profile(
            self.step_ms,
            self.phase_ms,
            self.train_wall_seconds,
            self.cuda_max_allocated_bytes,
            self.cuda_max_reserved_bytes,
        )
        if self.path is None:
            return record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return record
