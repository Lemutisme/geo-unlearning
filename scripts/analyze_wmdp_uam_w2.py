#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


ARMS = ("base", "rmu", "rmu_gu", "uam", "uam_gu")
EXPECTED_UPDATES = 80
MAX_ORTHOGONALITY = 1e-6
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def load_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Missing JSON artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return value


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def base_gate(cyber, mmlu):
    cyber_delta = abs(float(cyber) - 0.4400)
    mmlu_delta = abs(float(mmlu) - 0.5810)
    passed = cyber_delta <= 0.0300 + 1e-12 and mmlu_delta <= 0.0200 + 1e-12
    return passed, {"cyber_delta": cyber_delta, "mmlu_delta": mmlu_delta}


def audit_artifacts(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Matrix root is not a directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink artifact: {path}")
        lower = path.name.lower()
        if path.is_dir() and lower.startswith("checkpoint-"):
            raise ValueError(f"Unexpected checkpoint directory: {path}")
        if not path.is_file():
            continue
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or lower == "trainer_state.json"
            or lower.startswith(("optimizer", "scheduler", "rng_state"))
        ):
            raise ValueError(f"Unexpected checkpoint payload: {path}")


def load_metrics(arm_dir):
    path = Path(arm_dir) / "evals/LMEval_SUMMARY.json"
    summary = load_json(path)
    required = (
        "wmdp_cyber/acc",
        "wmdp_cyber/acc_stderr",
        "mmlu/acc",
        "mmlu/acc_stderr",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"Missing evaluation metric(s) {missing} in {path}")
    return {
        "wmdp_cyber": finite_number(summary[required[0]], required[0]),
        "wmdp_cyber_stderr": finite_number(summary[required[1]], required[1]),
        "mmlu": finite_number(summary[required[2]], required[2]),
        "mmlu_stderr": finite_number(summary[required[3]], required[3]),
    }


def load_resource(arm_dir):
    resource = load_json(Path(arm_dir) / "resource.json")
    required = (
        "wall_seconds",
        "training_wall_seconds",
        "peak_nvml_mib",
        "peak_cuda_allocated_mib",
        "max_rss_kib",
    )
    missing = [key for key in required if key not in resource]
    if missing:
        raise ValueError(f"Missing resource field(s): {missing}")
    result = {}
    for key in required:
        value = resource[key]
        result[key] = None if value is None else finite_number(value, key)
    return result


def read_jsonl(path):
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"Missing diagnostics artifact: {path}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed diagnostics at {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Diagnostics record is not a mapping at {path}:{line_number}")
        records.append(record)
    return records


def validate_geometry(arm, arm_dir, expected_updates=EXPECTED_UPDATES):
    records = read_jsonl(Path(arm_dir) / "geometry.jsonl")
    record_type = {
        "rmu": "rmu_geometry",
        "rmu_gu": "geometry",
        "uam": "uam_geometry",
        "uam_gu": "uam_geometry",
    }[arm]
    geometry = [record for record in records if record.get("record_type") == record_type]
    if len(geometry) != expected_updates:
        raise ValueError(
            f"{arm} requires {expected_updates} geometry updates, found {len(geometry)}"
        )
    if [record.get("update_step") for record in geometry] != list(
        range(1, expected_updates + 1)
    ):
        raise ValueError(f"{arm} geometry update steps are not contiguous")

    if arm == "rmu":
        if geometry[-1].get("finalizer_calls") != expected_updates:
            raise ValueError(
                f"RMU finalizer count does not equal {expected_updates}"
            )
    elif arm == "rmu_gu":
        if geometry[-1].get("projection_calls") != expected_updates:
            raise ValueError(
                f"RMU+GU projection count does not equal {expected_updates}"
            )
        for record in geometry:
            residual = finite_number(
                record.get("relative_orthogonality_residual"),
                "RMU+GU orthogonality",
            )
            if residual > MAX_ORTHOGONALITY:
                raise ValueError("RMU+GU orthogonality residual exceeds 1e-6")
    else:
        if geometry[-1].get("uam_calls") != expected_updates:
            raise ValueError(
                f"UAM finalizer count does not equal {expected_updates}"
            )
        if geometry[-1].get("replay_calls") != expected_updates:
            raise ValueError(f"UAM replay count does not equal {expected_updates}")
        if arm == "uam_gu":
            for record in geometry:
                if not record.get("residual_gate_kept", False):
                    continue
                residual = finite_number(
                    record.get("relative_residual_orthogonality"),
                    "UAM+GU orthogonality",
                )
                if residual > MAX_ORTHOGONALITY:
                    raise ValueError("UAM+GU orthogonality residual exceeds 1e-6")
    return geometry


def choose_conclusion(arms):
    uam = arms["uam"]
    combined = arms["uam_gu"]
    cyber_better = combined["wmdp_cyber"] <= uam["wmdp_cyber"]
    mmlu_better = combined["mmlu"] >= uam["mmlu"]
    if cyber_better and mmlu_better:
        return "complementarity"
    if cyber_better != mmlu_better:
        return "trade_off"
    return "dominated"


def analyze_matrix(root):
    root = Path(root)
    audit_artifacts(root)
    arms = {}
    provenance = None
    for arm in ARMS:
        arm_dir = root / arm
        metrics = load_metrics(arm_dir)
        resources = load_resource(arm_dir)
        current_provenance = load_json(arm_dir / "provenance.json")
        if provenance is None:
            provenance = current_provenance
        elif current_provenance != provenance:
            raise ValueError(f"Experiment provenance drift detected for {arm}")
        if arm != "base":
            validate_geometry(arm, arm_dir)
        arms[arm] = {**metrics, **resources}

    passed, deltas = base_gate(arms["base"]["wmdp_cyber"], arms["base"]["mmlu"])
    if not passed:
        raise ValueError(f"Base compatibility gate failed: {deltas}")
    rmu_time = arms["rmu"]["training_wall_seconds"]
    if rmu_time is None or rmu_time <= 0.0:
        raise ValueError("RMU training wall time must be positive")
    for arm in ARMS:
        training_time = arms[arm]["training_wall_seconds"]
        arms[arm]["relative_training_runtime"] = (
            None if training_time is None else training_time / rmu_time
        )
    return {
        "scope": "WMDP-Cyber paper-derived port",
        "base_gate": {"passed": passed, **deltas},
        "provenance": provenance,
        "arms": arms,
        "conclusion": choose_conclusion(arms),
    }


def markdown_table(result):
    labels = {
        "base": "Base",
        "rmu": "RMU",
        "rmu_gu": "RMU + GU",
        "uam": "UAM",
        "uam_gu": "UAM + GU",
    }
    lines = [
        "| Method | WMDP-Cyber ↓ | MMLU ↑ | Relative runtime | wall-clock | peak mem |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = result["arms"][arm]
        relative = values["relative_training_runtime"]
        relative_text = "—" if relative is None else f"{relative:.3f}×"
        lines.append(
            f"| {labels[arm]} | {values['wmdp_cyber']:.4f} | "
            f"{values['mmlu']:.4f} | {relative_text} | "
            f"{values['wall_seconds']:.1f}s | {values['peak_nvml_mib']:.0f} MiB |"
        )
    return "\n".join(lines) + "\n"


def check_base_only(root):
    root = Path(root)
    audit_artifacts(root / "base")
    metrics = load_metrics(root / "base")
    passed, deltas = base_gate(metrics["wmdp_cyber"], metrics["mmlu"])
    if not passed:
        raise ValueError(f"Base compatibility gate failed: {deltas}")
    return {"passed": True, **metrics, **deltas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--check-base-gate", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    args = parser.parse_args()
    root = Path(args.matrix_root)
    if args.audit_only:
        audit_artifacts(root)
        return
    if args.check_base_gate:
        print(json.dumps(check_base_only(root), indent=2, sort_keys=True))
        return
    result = analyze_matrix(root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    table = markdown_table(result)
    if args.output_json:
        Path(args.output_json).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.output_markdown:
        Path(args.output_markdown).write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
