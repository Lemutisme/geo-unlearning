import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts/analyze_wmdp_uam_strong_pair.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "analyze_wmdp_uam_strong_pair",
        ANALYZER_PATH,
    )
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


def provenance():
    return {
        "model": "HuggingFaceH4/zephyr-7b-beta",
        "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
        "lm_eval_version": "0.4.8",
        "cyber_sha256": "a" * 64,
        "retain_dataset": "wikitext/wikitext-2-raw-v1/train",
    }


def strong_geometry(arm, count=160, residual=1e-8):
    records = []
    for step in range(1, count + 1):
        records.append(
            {
                "record_type": "uam_geometry",
                "update_step": step,
                "uam_calls": step,
                "replay_calls": step,
                "residual_gate_kept": arm == "uam_gu",
                "relative_residual_orthogonality": residual,
                "residual_orthogonality_safe": residual <= 1e-6,
                "predicted_forget_directional_derivative": float(step),
            }
        )
    for step in (1, count):
        records.append(
            {
                "record_type": "actual_delta",
                "update_step": step,
                "coverage": "full",
            }
        )
    return records


def strong_config(arm):
    return {
        "trainer": {
            "args": {
                "learning_rate": 1.25e-4,
                "max_steps": 160,
                "lr_scheduler_type": "constant",
            },
            "method_args": {
                "uam_config": {
                    "mode": arm,
                    "rho": 5e-5,
                    "noise_std": 0.01,
                }
            },
        }
    }


def make_primary_summary(tmp_path):
    path = tmp_path / "primary.json"
    write_json(
        path,
        {
            "provenance": provenance(),
            "arms": {
                "uam": {
                    "wmdp_cyber": 0.4282838449924509,
                    "mmlu": 0.5829653895456488,
                    "training_wall_seconds": 731.288,
                    "wall_seconds": 1591.982006072998,
                    "peak_nvml_mib": 34926.0,
                },
                "uam_gu": {
                    "wmdp_cyber": 0.42979365878208353,
                    "mmlu": 0.583677538812135,
                    "training_wall_seconds": 929.7619,
                    "wall_seconds": 1799.4482028484344,
                    "peak_nvml_mib": 37166.0,
                },
            },
        },
    )
    return path


def make_strong_pair(tmp_path, uam_cyber=0.29, uam_gu_cyber=0.28):
    metrics = {
        "uam": (uam_cyber, 0.560),
        "uam_gu": (uam_gu_cyber, 0.565),
    }
    for index, arm in enumerate(("uam", "uam_gu")):
        arm_dir = tmp_path / arm
        cyber, mmlu = metrics[arm]
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
                "wall_seconds": 1800.0 + index * 200,
                "training_wall_seconds": 1200.0 + index * 200,
                "peak_nvml_mib": 35000.0 + index * 1000,
                "peak_cuda_allocated_mib": None,
                "max_rss_kib": 6_000_000.0 + index,
            },
        )
        write_json(arm_dir / "provenance.json", provenance())
        write_jsonl(arm_dir / "geometry.jsonl", strong_geometry(arm))
        config_path = arm_dir / ".hydra/config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(strong_config(arm), sort_keys=True),
            encoding="utf-8",
        )
    return tmp_path


@pytest.mark.parametrize(
    ("cyber", "expected"),
    [(0.2699, "below"), (0.27, "inside"), (0.29, "inside"), (0.31, "inside"), (0.3101, "above")],
)
def test_target_band_is_inclusive(cyber, expected):
    analyzer = load_analyzer()

    assert analyzer.calibration_status(cyber) == expected


def test_complete_pair_computes_primary_deltas_and_complementarity(tmp_path):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong")
    primary = make_primary_summary(tmp_path)

    result = analyzer.analyze_pair(root, primary)

    assert result["calibration"]["status"] == "inside"
    assert result["conclusion"] == "complementarity"
    assert result["arms"]["uam_gu"]["relative_training_runtime"] == pytest.approx(
        1400.0 / 1200.0
    )
    assert result["deltas_vs_primary"]["uam"]["wmdp_cyber"] == pytest.approx(
        0.29 - 0.4282838449924509
    )
    assert result["pair_delta"]["mmlu"] == pytest.approx(0.005)
    assert result["diagnostics"]["uam_gu"]["actual_delta_records"] == 2


def test_pair_classifies_one_better_endpoint_as_trade_off(tmp_path):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong", uam_gu_cyber=0.30)
    primary = make_primary_summary(tmp_path)

    result = analyzer.analyze_pair(root, primary)

    assert result["conclusion"] == "trade_off"


def test_analyzer_rejects_159_updates(tmp_path):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong")
    primary = make_primary_summary(tmp_path)
    write_jsonl(root / "uam/geometry.jsonl", strong_geometry("uam", count=159))

    with pytest.raises(ValueError, match="160"):
        analyzer.analyze_pair(root, primary)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("args", "learning_rate"), 1e-4, "learning rate"),
        (("args", "max_steps"), 80, "max steps"),
        (("args", "lr_scheduler_type"), "linear", "scheduler"),
        (("method_args", "uam_config", "rho"), 5e-4, "rho"),
    ],
)
def test_analyzer_rejects_config_drift(tmp_path, path, value, message):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong")
    primary = make_primary_summary(tmp_path)
    config_path = root / "uam/.hydra/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = config["trainer"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyzer.analyze_pair(root, primary)


def test_analyzer_rejects_provenance_drift(tmp_path):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong")
    primary = make_primary_summary(tmp_path)
    changed = provenance()
    changed["lm_eval_version"] = "different"
    write_json(root / "uam_gu/provenance.json", changed)

    with pytest.raises(ValueError, match="provenance"):
        analyzer.analyze_pair(root, primary)


def test_analyzer_rejects_unsafe_kept_residual(tmp_path):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong")
    primary = make_primary_summary(tmp_path)
    write_jsonl(
        root / "uam_gu/geometry.jsonl",
        strong_geometry("uam_gu", residual=2e-6),
    )

    with pytest.raises(ValueError, match="orthogonality"):
        analyzer.analyze_pair(root, primary)


def test_analyzer_rejects_checkpoint_payload(tmp_path):
    analyzer = load_analyzer()
    root = make_strong_pair(tmp_path / "strong")
    primary = make_primary_summary(tmp_path)
    (root / "uam/checkpoint-160").mkdir()

    with pytest.raises(ValueError, match="checkpoint"):
        analyzer.analyze_pair(root, primary)
