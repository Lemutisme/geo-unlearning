import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_pcgrad_smoke.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_pcgrad_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_arm(root, dataset, method, *, conflict_rate=1.0, distance=0.0):
    arm_name = f"{dataset}_{method}_production"
    arm_dir = root / arm_name
    eval_dir = arm_dir / "checkpoint-10" / "evals"
    eval_dir.mkdir(parents=True)
    summary_name = (
        "MUSE_SUMMARY.json" if dataset.startswith("muse") else "TOFU_SUMMARY.json"
    )
    metric_offset = {"control": 0.0, "gu": 1.0, "pcgrad": 2.0}[method]
    (eval_dir / summary_name).write_text(
        json.dumps(
            {
                "forget_metric": 10.0 + metric_offset,
                "retain_metric": 20.0 - metric_offset,
            }
        )
    )
    config = {
        "task_name": f"task_{dataset}_{method}",
        "paths": {"output_dir": str(arm_dir)},
        "model": {"name": dataset},
        "trainer": {
            "args": {"learning_rate": 1e-5},
            "method_args": {
                "geometric_config": {
                    "gu_enabled": method != "control",
                    "gradient_surgery": method if method != "control" else "gu",
                    "diagnostics_path": (
                        str(arm_dir / "gu_diagnostics.jsonl")
                        if method != "control"
                        else None
                    ),
                    "actual_delta_mode": "off",
                }
            },
        },
    }
    hydra_dir = arm_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text(yaml.safe_dump(config))
    if method != "control":
        conflicts = [True, conflict_rate == 1.0]
        if conflict_rate == 0.0:
            conflicts = [False, False]
        records = []
        for index, conflict in enumerate(conflicts, start=1):
            records.append(
                {
                    "record_type": "geometry",
                    "update_step": index,
                    "conflict": conflict,
                    "relative_orthogonality_residual": 1e-8 * index,
                    "relative_surgery_magnitude": 0.1 * index,
                    "relative_pcgrad_gu_distance": (
                        distance if method == "pcgrad" else None
                    ),
                }
            )
        (arm_dir / "gu_diagnostics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    return arm_name


def make_matrix(tmp_path, *, one_non_degenerate=True):
    root = tmp_path / "matrix"
    root.mkdir()
    datasets = ["tofu01", "tofu05", "tofu10", "muse_news", "muse_books"]
    rows = []
    pid = 100
    for dataset in datasets:
        for method in ("control", "gu", "pcgrad"):
            conflict_rate = 0.5 if one_non_degenerate and dataset == "tofu01" else 1.0
            distance = 0.2 if one_non_degenerate and dataset == "tofu01" else 0.0
            arm_name = write_arm(
                root,
                dataset,
                method,
                conflict_rate=conflict_rate,
                distance=distance,
            )
            rows.append(
                [
                    str(pid),
                    str(pid % 2),
                    dataset,
                    method,
                    "production",
                    "start",
                    "end",
                    "0",
                    f"run {arm_name}",
                ]
            )
            pid += 1
    header = [
        "pid",
        "gpu",
        "dataset",
        "method",
        "system_mode",
        "start_utc",
        "end_utc",
        "exit_code",
        "command",
    ]
    (root / "RUN_MANIFEST.tsv").write_text(
        "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows)
    )
    return root


def test_analyzer_emits_raw_deltas_geometry_and_advancement(tmp_path):
    analyzer = load_analyzer()
    root = make_matrix(tmp_path, one_non_degenerate=True)

    result = analyzer.analyze_matrix(root)
    tofu = result["datasets"]["tofu01"]

    assert tofu["raw"]["control"]["forget_metric"] == 10.0
    assert tofu["deltas"]["gu_minus_control"]["forget_metric"] == 1.0
    assert tofu["deltas"]["pcgrad_minus_control"]["forget_metric"] == 2.0
    assert tofu["deltas"]["pcgrad_minus_gu"]["retain_metric"] == -1.0
    assert tofu["conflict_rate"] == pytest.approx(0.5)
    assert tofu["mean_relative_pcgrad_gu_distance"] == pytest.approx(0.2)
    assert tofu["degenerate_to_gu"] is False
    assert tofu["advance"] is True
    assert result["global_stop"] is False


def test_cross_dataset_degeneracy_triggers_global_stop(tmp_path):
    analyzer = load_analyzer()
    root = make_matrix(tmp_path, one_non_degenerate=False)

    result = analyzer.analyze_matrix(root)

    assert result["global_stop"] is True
    assert all(
        row["degenerate_to_gu"] and not row["advance"]
        for row in result["datasets"].values()
    )


def test_analyzer_raises_instead_of_filling_missing_metrics(tmp_path):
    analyzer = load_analyzer()
    root = make_matrix(tmp_path)
    summary = next((root / "tofu01_pcgrad_production").rglob("TOFU_SUMMARY.json"))
    summary.write_text(json.dumps({"forget_metric": 12.0}))

    with pytest.raises(ValueError, match="Metric keys differ"):
        analyzer.analyze_matrix(root)


def test_analyzer_rejects_non_method_configuration_drift(tmp_path):
    analyzer = load_analyzer()
    root = make_matrix(tmp_path)
    config_path = root / "tofu05_pcgrad_production" / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["trainer"]["args"]["learning_rate"] = 2e-5
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="Matched configuration drift"):
        analyzer.analyze_matrix(root)


def test_markdown_and_json_outputs_are_written(tmp_path):
    analyzer = load_analyzer()
    root = make_matrix(tmp_path)
    result = analyzer.analyze_matrix(root)
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "analysis.json"

    analyzer.write_outputs(result, markdown_path, json_path)

    assert "# PCGrad Multi-Dataset Smoke Analysis" in markdown_path.read_text()
    assert "tofu01" in markdown_path.read_text()
    assert json.loads(json_path.read_text())["datasets"]["muse_books"]
