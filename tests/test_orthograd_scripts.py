import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESS_PROFILER = ROOT / "scripts/profile_process_tree.py"


def load_process_profiler():
    spec = importlib.util.spec_from_file_location(
        "profile_process_tree",
        PROCESS_PROFILER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_profiler_records_real_child_rss_and_injected_gpu_peak(tmp_path):
    profiler = load_process_profiler()
    gpu_values = iter([100, 120, 160, 140, 130])

    def gpu_query(_gpu):
        return next(gpu_values, 130) * 1024 * 1024

    output = tmp_path / "profile.json"
    command = [
        sys.executable,
        "-c",
        "import time; payload=bytearray(16*1024*1024); time.sleep(0.25)",
    ]

    exit_code, record = profiler.profile_command(
        command,
        output,
        gpu_index=0,
        poll_ms=25,
        gpu_query=gpu_query,
    )

    assert exit_code == 0
    assert record["exit_code"] == 0
    assert record["wall_seconds"] >= 0.2
    assert record["peak_process_tree_rss_bytes"] >= 16 * 1024 * 1024
    assert record["gpu_baseline_used_bytes"] == 100 * 1024 * 1024
    assert record["gpu_peak_used_bytes"] == 160 * 1024 * 1024
    assert record["gpu_peak_delta_bytes"] == 60 * 1024 * 1024
    assert record["sample_count"] >= 2
    assert json.loads(output.read_text()) == record
    assert output.read_text().endswith("\n")


def test_process_profiler_persists_and_propagates_child_failure(tmp_path):
    profiler = load_process_profiler()
    output = tmp_path / "failed.json"

    exit_code, record = profiler.profile_command(
        [sys.executable, "-c", "raise SystemExit(7)"],
        output,
        gpu_index=0,
        poll_ms=10,
        gpu_query=lambda _gpu: 0,
    )

    assert exit_code == 7
    assert record["exit_code"] == 7
    assert json.loads(output.read_text())["exit_code"] == 7


def test_gpu_query_parses_nvidia_smi_mib(monkeypatch):
    profiler = load_process_profiler()

    class Result:
        returncode = 0
        stdout = "1234\n"
        stderr = ""

    monkeypatch.setattr(profiler.subprocess, "run", lambda *args, **kwargs: Result())

    assert profiler.query_gpu_used_bytes(1) == 1234 * 1024 * 1024


def test_gpu_query_rejects_malformed_output(monkeypatch):
    profiler = load_process_profiler()

    class Result:
        returncode = 0
        stdout = "not-a-number\n"
        stderr = ""

    monkeypatch.setattr(profiler.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="memory.used"):
        profiler.query_gpu_used_bytes(0)
