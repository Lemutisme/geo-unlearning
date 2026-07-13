import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts/analyze_wmdp_uam_w2.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_wmdp_uam_w2", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def geometry_records(arm, count=80, residual=1e-8):
    records = []
    for step in range(1, count + 1):
        if arm == "rmu":
            records.append(
                {
                    "record_type": "rmu_geometry",
                    "update_step": step,
                    "finalizer_calls": step,
                }
            )
        elif arm == "rmu_gu":
            records.append(
                {
                    "record_type": "geometry",
                    "update_step": step,
                    "projection_calls": step,
                    "relative_orthogonality_residual": residual,
                }
            )
        else:
            records.append(
                {
                    "record_type": "uam_geometry",
                    "update_step": step,
                    "uam_calls": step,
                    "replay_calls": step,
                    "residual_gate_kept": arm == "uam_gu",
                    "relative_residual_orthogonality": residual,
                }
            )
    return records


def make_complete_matrix(tmp_path):
    metrics = {
        "base": (0.440, 0.581),
        "rmu": (0.350, 0.570),
        "rmu_gu": (0.330, 0.575),
        "uam": (0.310, 0.568),
        "uam_gu": (0.290, 0.572),
    }
    provenance = {
        "model": "HuggingFaceH4/zephyr-7b-beta",
        "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
        "lm_eval_version": "0.4.8",
        "cyber_sha256": "a" * 64,
        "retain_dataset": "wikitext/wikitext-2-raw-v1/train",
    }
    for index, arm in enumerate(("base", "rmu", "rmu_gu", "uam", "uam_gu")):
        cyber, mmlu = metrics[arm]
        arm_dir = tmp_path / arm
        write_json(
            arm_dir / "evals/LMEval_SUMMARY.json",
            {
                "wmdp_cyber/acc": cyber,
                "wmdp_cyber/acc_stderr": 0.01,
                "mmlu/acc": mmlu,
                "mmlu/acc_stderr": 0.004,
            },
        )
        write_json(
            arm_dir / "resource.json",
            {
                "wall_seconds": 100.0 + index * 10,
                "training_wall_seconds": None if arm == "base" else 40.0 + index,
                "peak_nvml_mib": 20000 + index * 100,
                "peak_cuda_allocated_mib": None if arm == "base" else 18000 + index,
                "max_rss_kib": 1_000_000 + index,
            },
        )
        write_json(arm_dir / "provenance.json", provenance)
        if arm != "base":
            write_jsonl(
                arm_dir / "geometry.jsonl",
                geometry_records(arm),
            )
    return tmp_path


def test_base_gate_uses_fixed_inclusive_tolerances():
    analyzer = load_analyzer()

    assert analyzer.base_gate(0.410, 0.561)[0] is True
    assert analyzer.base_gate(0.470, 0.601)[0] is True
    assert analyzer.base_gate(0.4099, 0.581)[0] is False
    assert analyzer.base_gate(0.440, 0.6011)[0] is False


def test_complete_matrix_recomputes_metrics_runtime_and_complementarity(tmp_path):
    analyzer = load_analyzer()
    root = make_complete_matrix(tmp_path)

    result = analyzer.analyze_matrix(root)

    assert result["base_gate"]["passed"] is True
    assert result["arms"]["uam_gu"]["wmdp_cyber"] == pytest.approx(0.290)
    assert result["arms"]["rmu"]["relative_training_runtime"] == 1.0
    assert result["arms"]["uam_gu"]["relative_training_runtime"] == pytest.approx(
        44.0 / 41.0
    )
    assert result["conclusion"] == "complementarity"


def test_analyzer_rejects_missing_metric(tmp_path):
    analyzer = load_analyzer()
    root = make_complete_matrix(tmp_path)
    summary = root / "uam/evals/LMEval_SUMMARY.json"
    value = json.loads(summary.read_text())
    del value["mmlu/acc"]
    write_json(summary, value)

    with pytest.raises(ValueError, match="mmlu/acc"):
        analyzer.analyze_matrix(root)


def test_analyzer_rejects_79_of_80_updates(tmp_path):
    analyzer = load_analyzer()
    root = make_complete_matrix(tmp_path)
    write_jsonl(root / "uam/geometry.jsonl", geometry_records("uam", count=79))

    with pytest.raises(ValueError, match="80"):
        analyzer.analyze_matrix(root)


def test_analyzer_rejects_unsafe_kept_residual(tmp_path):
    analyzer = load_analyzer()
    root = make_complete_matrix(tmp_path)
    write_jsonl(
        root / "uam_gu/geometry.jsonl",
        geometry_records("uam_gu", residual=2e-6),
    )

    with pytest.raises(ValueError, match="orthogonality"):
        analyzer.analyze_matrix(root)


def test_analyzer_rejects_provenance_drift(tmp_path):
    analyzer = load_analyzer()
    root = make_complete_matrix(tmp_path)
    provenance = json.loads((root / "rmu/provenance.json").read_text())
    provenance["lm_eval_version"] = "different"
    write_json(root / "rmu/provenance.json", provenance)

    with pytest.raises(ValueError, match="provenance"):
        analyzer.analyze_matrix(root)


def test_analyzer_rejects_checkpoint_payload_or_symlink(tmp_path):
    analyzer = load_analyzer()
    root = make_complete_matrix(tmp_path)
    (root / "uam/checkpoint-80").mkdir()

    with pytest.raises(ValueError, match="checkpoint"):
        analyzer.analyze_matrix(root)

    (root / "uam/checkpoint-80").rmdir()
    (root / "uam/model-link").symlink_to(root / "base")
    with pytest.raises(ValueError, match="symlink"):
        analyzer.analyze_matrix(root)
