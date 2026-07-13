#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import yaml

try:
    from analyze_wmdp_uam_w2 import (
        audit_artifacts,
        choose_conclusion,
        finite_number,
        load_json,
        load_metrics,
        load_resource,
        read_jsonl,
        validate_geometry,
    )
except ModuleNotFoundError:
    from scripts.analyze_wmdp_uam_w2 import (
        audit_artifacts,
        choose_conclusion,
        finite_number,
        load_json,
        load_metrics,
        load_resource,
        read_jsonl,
        validate_geometry,
    )


ARMS = ("uam", "uam_gu")
EXPECTED_UPDATES = 160
EXPECTED_LR = 1.25e-4
EXPECTED_SCHEDULER = "constant"
EXPECTED_RHO = 5e-5
EXPECTED_NOISE_STD = 0.01
TARGET_LOW = 0.27
TARGET_HIGH = 0.31


def load_yaml(path):
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Missing YAML artifact: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Malformed YAML artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"YAML artifact must contain a mapping: {path}")
    return value


def nested_mapping(value, key, label):
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"Strong-pair config is missing {label}")
    return nested


def validate_config(arm, arm_dir):
    config = load_yaml(Path(arm_dir) / ".hydra/config.yaml")
    trainer = nested_mapping(config, "trainer", "trainer")
    args = nested_mapping(trainer, "args", "trainer.args")
    method_args = nested_mapping(
        trainer,
        "method_args",
        "trainer.method_args",
    )
    uam = nested_mapping(
        method_args,
        "uam_config",
        "trainer.method_args.uam_config",
    )

    learning_rate = finite_number(
        args.get("learning_rate"),
        "strong-pair learning rate",
    )
    if not math.isclose(learning_rate, EXPECTED_LR, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Strong-pair learning rate must equal {EXPECTED_LR}, got {learning_rate}"
        )
    max_steps = args.get("max_steps")
    if isinstance(max_steps, bool) or max_steps != EXPECTED_UPDATES:
        raise ValueError(
            f"Strong-pair max steps must equal {EXPECTED_UPDATES}, got {max_steps!r}"
        )
    scheduler = args.get("lr_scheduler_type")
    if scheduler != EXPECTED_SCHEDULER:
        raise ValueError(
            f"Strong-pair scheduler must be {EXPECTED_SCHEDULER}, got {scheduler!r}"
        )
    rho = finite_number(uam.get("rho"), "strong-pair rho")
    if not math.isclose(rho, EXPECTED_RHO, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Strong-pair rho must equal {EXPECTED_RHO}, got {rho}")
    noise_std = finite_number(
        uam.get("noise_std"),
        "strong-pair noise std",
    )
    if not math.isclose(
        noise_std,
        EXPECTED_NOISE_STD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Strong-pair noise std must equal "
            f"{EXPECTED_NOISE_STD}, got {noise_std}"
        )
    if uam.get("mode") != arm:
        raise ValueError(
            f"Strong-pair {arm} config has mode {uam.get('mode')!r}"
        )
    return {
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "lr_scheduler_type": scheduler,
        "rho": rho,
        "noise_std": noise_std,
        "mode": arm,
    }


def calibration_status(cyber):
    cyber = finite_number(cyber, "UAM WMDP-Cyber accuracy")
    if cyber < TARGET_LOW:
        return "below"
    if cyber > TARGET_HIGH:
        return "above"
    return "inside"


def summarize_geometry(arm, arm_dir):
    geometry = validate_geometry(
        arm,
        arm_dir,
        expected_updates=EXPECTED_UPDATES,
    )
    records = read_jsonl(Path(arm_dir) / "geometry.jsonl")
    actual_delta = [
        record for record in records if record.get("record_type") == "actual_delta"
    ]
    if [record.get("update_step") for record in actual_delta] != [1, EXPECTED_UPDATES]:
        raise ValueError(
            "Strong-pair actual-delta probes must occur at steps "
            f"1 and {EXPECTED_UPDATES}"
        )
    if any(record.get("coverage") != "full" for record in actual_delta):
        raise ValueError("Strong-pair actual-delta probes must have full coverage")

    residuals = [
        finite_number(
            record.get("relative_residual_orthogonality"),
            f"{arm} orthogonality",
        )
        for record in geometry
    ]
    predicted = [
        finite_number(
            record.get("predicted_forget_directional_derivative"),
            f"{arm} predicted forget derivative",
        )
        for record in geometry
    ]
    return {
        "geometry_updates": len(geometry),
        "uam_calls": geometry[-1]["uam_calls"],
        "replay_calls": geometry[-1]["replay_calls"],
        "gate_kept": sum(
            bool(record.get("residual_gate_kept", False)) for record in geometry
        ),
        "max_relative_orthogonality": max(residuals),
        "positive_forget_derivative_steps": sum(value > 0.0 for value in predicted),
        "actual_delta_records": len(actual_delta),
    }


def metric_delta(current, reference):
    keys = (
        "wmdp_cyber",
        "mmlu",
        "training_wall_seconds",
        "wall_seconds",
        "peak_nvml_mib",
    )
    result = {}
    for key in keys:
        current_value = current.get(key)
        reference_value = reference.get(key)
        if current_value is None or reference_value is None:
            result[key] = None
        else:
            result[key] = current_value - reference_value
    return result


def analyze_pair(root, primary_summary):
    root = Path(root)
    audit_artifacts(root)
    primary = load_json(primary_summary)
    primary_arms = primary.get("arms")
    primary_provenance = primary.get("provenance")
    if not isinstance(primary_arms, dict) or not isinstance(primary_provenance, dict):
        raise ValueError("Primary summary is missing arms or provenance")

    arms = {}
    configs = {}
    diagnostics = {}
    for arm in ARMS:
        if not isinstance(primary_arms.get(arm), dict):
            raise ValueError(f"Primary summary is missing arm {arm}")
        arm_dir = root / arm
        current_provenance = load_json(arm_dir / "provenance.json")
        if current_provenance != primary_provenance:
            raise ValueError(f"Experiment provenance drift detected for {arm}")
        configs[arm] = validate_config(arm, arm_dir)
        diagnostics[arm] = summarize_geometry(arm, arm_dir)
        arms[arm] = {
            **load_metrics(arm_dir),
            **load_resource(arm_dir),
        }

    uam_time = arms["uam"]["training_wall_seconds"]
    if uam_time is None or uam_time <= 0.0:
        raise ValueError("Strong UAM training wall time must be positive")
    for arm in ARMS:
        training_time = arms[arm]["training_wall_seconds"]
        if training_time is None or training_time <= 0.0:
            raise ValueError(f"Strong {arm} training wall time must be positive")
        arms[arm]["relative_training_runtime"] = training_time / uam_time

    deltas_vs_primary = {
        arm: metric_delta(arms[arm], primary_arms[arm]) for arm in ARMS
    }
    pair_delta = metric_delta(arms["uam_gu"], arms["uam"])
    return {
        "scope": "WMDP-Cyber strong UAM calibration pair",
        "target_is_calibration_only": True,
        "calibration": {
            "low": TARGET_LOW,
            "high": TARGET_HIGH,
            "center": 0.29,
            "status": calibration_status(arms["uam"]["wmdp_cyber"]),
        },
        "provenance": primary_provenance,
        "config": configs,
        "primary_arms": {arm: primary_arms[arm] for arm in ARMS},
        "arms": arms,
        "diagnostics": diagnostics,
        "deltas_vs_primary": deltas_vs_primary,
        "pair_delta": pair_delta,
        "conclusion": choose_conclusion(arms),
    }


def markdown_table(result):
    lines = [
        "| Method | WMDP-Cyber ↓ | MMLU ↑ | train runtime | wall-clock | peak mem |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"uam": "UAM", "uam_gu": "UAM + GU"}
    for arm in ARMS:
        primary = result["primary_arms"][arm]
        lines.append(
            f"| Primary {labels[arm]} | {primary['wmdp_cyber']:.4f} | "
            f"{primary['mmlu']:.4f} | {primary['training_wall_seconds']:.1f}s | "
            f"{primary['wall_seconds']:.1f}s | {primary['peak_nvml_mib']:.0f} MiB |"
        )
        strong = result["arms"][arm]
        lines.append(
            f"| Strong {labels[arm]} | {strong['wmdp_cyber']:.4f} | "
            f"{strong['mmlu']:.4f} | {strong['training_wall_seconds']:.1f}s | "
            f"{strong['wall_seconds']:.1f}s | {strong['peak_nvml_mib']:.0f} MiB |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--primary-summary", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    args = parser.parse_args()
    root = Path(args.matrix_root)
    result = analyze_pair(root, args.primary_summary)
    if args.audit_only:
        return

    output_json = Path(args.output_json) if args.output_json else root / "STRONG_SUMMARY.json"
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown
        else root / "STRONG_TABLE.md"
    )
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(markdown_table(result), encoding="utf-8")


if __name__ == "__main__":
    main()
