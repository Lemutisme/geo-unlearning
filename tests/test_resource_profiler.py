import json

import pytest

from trainer.resource_profiler import (
    ResourceProfileCallback,
    deterministic_quantile,
    summarize_resource_profile,
)


def test_deterministic_quantile_uses_linear_interpolation():
    values = [10.0, 20.0, 40.0, 100.0]

    assert deterministic_quantile(values, 0.0) == 10.0
    assert deterministic_quantile(values, 0.5) == 30.0
    assert deterministic_quantile(values, 0.95) == pytest.approx(91.0)
    assert deterministic_quantile(values, 1.0) == 100.0


def test_resource_summary_separates_first_and_steady_updates():
    summary = summarize_resource_profile(
        step_ms=[100.0, 20.0, 30.0],
        phase_ms={"projection": [3.0, 4.0], "gradients": [10.0]},
        train_wall_seconds=0.2,
        cuda_max_allocated_bytes=100,
        cuda_max_reserved_bytes=120,
    )

    assert summary["update_count"] == 3
    assert summary["first_update_ms"] == 100.0
    assert summary["steady_update_mean_ms"] == 25.0
    assert summary["steady_update_p50_ms"] == 25.0
    assert summary["steady_update_p95_ms"] == pytest.approx(29.5)
    assert summary["phase_totals_ms"] == {
        "gradients": 10.0,
        "projection": 7.0,
    }
    assert summary["cuda_max_reserved_bytes"] >= summary["cuda_max_allocated_bytes"]


def test_resource_callback_atomically_writes_sorted_json(tmp_path):
    path = tmp_path / "resource.json"
    callback = ResourceProfileCallback(path)
    callback.step_ms.extend([8.0, 5.0])
    callback.phase_ms["projection"].append(2.0)
    callback.train_wall_seconds = 0.1
    callback.cuda_max_allocated_bytes = 10
    callback.cuda_max_reserved_bytes = 12

    record = callback.write_profile()

    persisted = json.loads(path.read_text())
    assert persisted == record
    assert path.read_text().endswith("\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_resource_summary_rejects_invalid_memory_order():
    with pytest.raises(ValueError, match="reserved"):
        summarize_resource_profile(
            step_ms=[1.0],
            phase_ms={},
            train_wall_seconds=0.1,
            cuda_max_allocated_bytes=2,
            cuda_max_reserved_bytes=1,
        )
