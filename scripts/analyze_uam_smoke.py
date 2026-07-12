#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import tempfile
from numbers import Integral, Real
from pathlib import Path


METHODS = ("uam_nll", "uam_simnpo", "uam_gu_nll", "uam_gu_simnpo")
MANIFEST_COLUMNS = (
    "pid",
    "gpu",
    "method",
    "start_utc",
    "end_utc",
    "exit_code",
    "command",
)
METHOD_PARAMETERS = {
    "uam_nll": ("uam", "nll", "fixed_loss"),
    "uam_simnpo": ("uam", "simnpo", "fixed_loss"),
    "uam_gu_nll": ("uam_gu", "nll", "metric_trust"),
    "uam_gu_simnpo": ("uam_gu", "simnpo", "metric_trust"),
}
GEOMETRY_REQUIRED = {
    "update_step",
    "mode",
    "forget_signal",
    "perturbation_normalization",
    "rho",
    "requested_perturbation_norm",
    "effective_perturbation_norm",
    "effective_perturbation_ratio",
    "forget_norm",
    "retain_norm",
    "perturbed_retain_norm",
    "uam_coefficient",
    "residual_tangent_norm",
    "residual_normal_norm",
    "relative_residual_orthogonality",
    "residual_forget_gate_dot",
    "residual_gate_kept",
    "predicted_forget_directional_derivative",
    "predicted_retain_directional_derivative",
    "identity_fallback_parameters",
    "replay_microsteps",
    "uam_calls",
    "replay_calls",
}
GEOMETRY_STRING_FIELDS = {
    "mode",
    "forget_signal",
    "perturbation_normalization",
}
GEOMETRY_INTEGER_FIELDS = {
    "update_step",
    "identity_fallback_parameters",
    "replay_microsteps",
    "uam_calls",
    "replay_calls",
}
ACTUAL_DELTA_REQUIRED = {
    "update_step",
    "coverage",
    "sampled_elements",
    "forget_directional_derivative",
    "retain_directional_derivative",
    "forget_gradient_norm",
    "retain_gradient_norm",
    "parameter_delta_norm",
}
ACTUAL_DELTA_NUMERIC_FIELDS = ACTUAL_DELTA_REQUIRED - {"coverage"}
CHECKPOINT_PAYLOAD_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "trainer_state.json",
    "model_state*",
    "training_args*",
    "optimizer*",
    "scheduler*",
    "rng_state*",
)


def _load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON: {path}") from error
    return value


def _finite_number(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite(record, fields, label, path):
    for field in fields:
        if not _finite_number(record[field]):
            raise ValueError(f"{label} field {field!r} in {path} must be finite")


def read_manifest(root):
    root = Path(root)
    path = root / "RUN_MANIFEST.tsv"
    if not path.is_file():
        raise ValueError(f"Missing run manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"Invalid run manifest schema: {path}")
        rows = list(reader)

    if any(
        None in row or any(row[column] is None for column in MANIFEST_COLUMNS)
        for row in rows
    ):
        raise ValueError(f"Invalid run manifest row schema: {path}")

    if len(rows) != len(METHODS):
        raise ValueError(
            f"Expected {len(METHODS)} manifest rows in {path}, found {len(rows)}"
        )
    observed = [row["method"] for row in rows]
    if set(observed) != set(METHODS) or len(set(observed)) != len(observed):
        raise ValueError(
            "Manifest must contain exactly one row for each approved method; "
            f"found {observed}"
        )
    for row in rows:
        try:
            pid = int(row["pid"])
        except ValueError as error:
            raise ValueError(f"Manifest PID is invalid: {row['pid']!r}") from error
        if pid <= 0:
            raise ValueError(f"Manifest PID must be positive: {row['pid']!r}")
        if row["gpu"] not in {"0", "1"}:
            raise ValueError(f"Manifest GPU must be 0 or 1: {row['gpu']!r}")
        if not row["start_utc"].strip() or not row["end_utc"].strip():
            raise ValueError(
                f"Manifest timestamps must be nonempty for {row['method']}"
            )
        expected_command = (
            f"bash scripts/uam_smoke_arm.sh {row['method']} "
            f"{row['gpu']} {root.name}"
        )
        if row["command"] != expected_command:
            raise ValueError(
                f"Manifest command mismatch for {row['method']}: "
                f"expected {expected_command!r}, found {row['command']!r}"
            )
        try:
            exit_code = int(row["exit_code"])
        except ValueError as error:
            raise ValueError(
                f"Invalid exit code for {row['method']}: {row['exit_code']!r}"
            ) from error
        if exit_code != 0:
            raise ValueError(
                f"Cannot analyze failed arm {row['method']}: exit {exit_code}"
            )
    return rows


def find_summary(arm_dir):
    arm_dir = Path(arm_dir)
    matches = sorted(
        path for path in arm_dir.rglob("TOFU_SUMMARY.json") if path.is_file()
    )
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one TOFU_SUMMARY.json under "
            f"{arm_dir}, found {len(matches)}"
        )
    summary = _load_json(matches[0])
    if not isinstance(summary, dict) or not summary:
        raise ValueError(f"Metric summary must be a nonempty mapping: {matches[0]}")
    for metric, value in summary.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"Metric names must be nonempty strings: {matches[0]}")
        if not _finite_number(value):
            raise ValueError(
                f"Metric {metric!r} in {matches[0]} must be finite and numeric"
            )
    return summary, matches[0]


def _load_diagnostics_jsonl(path):
    records = []
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Missing diagnostics: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed diagnostics JSONL at {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Diagnostics record at {path}:{line_number} is not a mapping"
                )
            records.append(record)
    return records


def _validate_geometry(record, index, path):
    missing = GEOMETRY_REQUIRED - record.keys()
    if missing:
        raise ValueError(
            f"UAM geometry record {index} in {path} is missing {sorted(missing)}"
        )
    for field in GEOMETRY_STRING_FIELDS:
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"UAM geometry field {field!r} in {path} must be a string")
    if record["mode"] not in {"uam", "uam_gu"}:
        raise ValueError(f"Unsupported UAM mode in {path}: {record['mode']!r}")
    if record["forget_signal"] not in {"nll", "simnpo"}:
        raise ValueError(
            f"Unsupported forget signal in {path}: {record['forget_signal']!r}"
        )
    if record["perturbation_normalization"] not in {
        "fixed_loss",
        "metric_trust",
    }:
        raise ValueError(
            "Unsupported perturbation normalization in "
            f"{path}: {record['perturbation_normalization']!r}"
        )
    if not isinstance(record["residual_gate_kept"], bool):
        raise ValueError(f"residual_gate_kept in {path} must be boolean")

    numeric_fields = GEOMETRY_REQUIRED - GEOMETRY_STRING_FIELDS - {"residual_gate_kept"}
    _require_finite(record, numeric_fields, "UAM geometry", path)
    for field in GEOMETRY_INTEGER_FIELDS:
        if not isinstance(record[field], Integral) or isinstance(record[field], bool):
            raise ValueError(f"UAM geometry field {field!r} in {path} must be integral")
    if record["update_step"] <= 0 or record["replay_microsteps"] <= 0:
        raise ValueError(f"UAM geometry counters in {path} must be positive")
    if record["identity_fallback_parameters"] < 0:
        raise ValueError(f"identity_fallback_parameters in {path} must be nonnegative")
    if not (record["uam_calls"] == record["replay_calls"] == record["update_step"]):
        raise ValueError(
            "UAM finalizer/replay counter drift at step "
            f"{record['update_step']} in {path}"
        )


def _validate_actual_delta(record, index, path):
    missing = ACTUAL_DELTA_REQUIRED - record.keys()
    if missing:
        raise ValueError(
            f"Actual-delta record {index} in {path} is missing {sorted(missing)}"
        )
    if record["coverage"] != "full":
        raise ValueError(
            f"Actual-delta coverage in {path} must be 'full', "
            f"found {record['coverage']!r}"
        )
    _require_finite(record, ACTUAL_DELTA_NUMERIC_FIELDS, "Actual-delta", path)
    for field in ("update_step", "sampled_elements"):
        if not isinstance(record[field], Integral) or isinstance(record[field], bool):
            raise ValueError(f"Actual-delta field {field!r} in {path} must be integral")
    if record["sampled_elements"] <= 0:
        raise ValueError(f"Actual-delta sampled_elements in {path} must be positive")


def read_diagnostics(path):
    path = Path(path)
    geometry = []
    actual_delta = []
    for record in _load_diagnostics_jsonl(path):
        record_type = record.get("record_type")
        if record_type == "uam_geometry":
            geometry.append(record)
        elif record_type == "actual_delta":
            actual_delta.append(record)
        else:
            raise ValueError(
                f"Unexpected diagnostics record type in {path}: {record_type!r}"
            )

    if len(geometry) != 10:
        raise ValueError(
            f"Expected exactly 10 UAM geometry records in {path}, found {len(geometry)}"
        )
    for index, record in enumerate(geometry, start=1):
        _validate_geometry(record, index, path)
    geometry_steps = [record["update_step"] for record in geometry]
    if geometry_steps != list(range(1, 11)):
        raise ValueError(
            f"UAM geometry update steps in {path} must be 1..10, "
            f"found {geometry_steps}"
        )
    rhos = {float(record["rho"]) for record in geometry}
    if len(rhos) != 1:
        raise ValueError(f"UAM diagnostics rho must be constant in {path}")
    if next(iter(rhos)) <= 0.0:
        raise ValueError(f"UAM diagnostics rho in {path} must be positive")
    positive_fields = {
        "requested_perturbation_norm",
        "effective_perturbation_norm",
        "effective_perturbation_ratio",
    }
    for record in geometry:
        for field in positive_fields:
            if record[field] <= 0.0:
                raise ValueError(
                    f"UAM geometry field {field!r} in {path} must be positive"
                )
        if record["replay_microsteps"] != 8:
            raise ValueError(f"UAM geometry replay_microsteps in {path} must equal 8")

    if len(actual_delta) != 2:
        raise ValueError(
            "Expected exactly 2 actual-delta records in "
            f"{path}, found {len(actual_delta)}"
        )
    for index, record in enumerate(actual_delta, start=1):
        _validate_actual_delta(record, index, path)
    actual_steps = [record["update_step"] for record in actual_delta]
    if actual_steps != [1, 10]:
        raise ValueError(
            f"Expected actual-delta steps [1, 10] in {path}, found {actual_steps}"
        )
    return {"geometry": geometry, "actual_delta": actual_delta}


def checkpoint_payloads(root):
    root = Path(root)
    payloads = set()
    if root.is_symlink():
        payloads.add(str(root))
    payloads.update(str(path) for path in root.rglob("*") if path.is_symlink())
    payloads.update(
        str(path)
        for path in root.rglob("checkpoint-*")
        if not path.is_symlink() and path.is_dir()
    )
    for pattern in CHECKPOINT_PAYLOAD_PATTERNS:
        payloads.update(
            str(path)
            for path in root.rglob(pattern)
            if not path.is_symlink() and path.is_file()
        )
    return sorted(payloads)


def _mean(values):
    return sum(values) / len(values)


def _metric_delta(left, baseline):
    result = {}
    for metric in sorted(baseline):
        delta = left[metric] - baseline[metric]
        if not math.isfinite(delta):
            raise ValueError(f"Metric subtraction for {metric!r} must remain finite")
        result[metric] = delta
    return result


def _aggregate_diagnostics(records, path):
    geometry = records["geometry"]
    perturbation_ratios = [
        record["effective_perturbation_ratio"] for record in geometry
    ]
    mean_ratio = _mean(perturbation_ratios)
    if not math.isfinite(mean_ratio) or mean_ratio <= 0.0:
        raise ValueError(
            f"Mean effective perturbation ratio in {path} must be finite and positive"
        )
    gate_rate = _mean([float(record["residual_gate_kept"]) for record in geometry])
    kept_residuals = [
        record["relative_residual_orthogonality"]
        for record in geometry
        if record["residual_gate_kept"]
    ]
    maximum_kept_residual = max(kept_residuals) if kept_residuals else None
    finalizer_count = geometry[-1]["uam_calls"]
    replay_count = geometry[-1]["replay_calls"]
    if finalizer_count != 10 or replay_count != 10:
        raise ValueError(
            f"Expected 10 UAM finalizers and replays in {path}, "
            f"found {finalizer_count} and {replay_count}"
        )
    return {
        "actual_delta_probe_count": len(records["actual_delta"]),
        "finalizer_count": finalizer_count,
        "max_relative_residual_orthogonality_kept": maximum_kept_residual,
        "mean_effective_perturbation_ratio": mean_ratio,
        "mean_residual_gate_rate": gate_rate,
        "replay_count": replay_count,
        "update_count": len(geometry),
    }


def analyze_matrix(matrix_root):
    root = Path(matrix_root)
    rows = read_manifest(root)
    payloads = checkpoint_payloads(root)
    if payloads:
        raise ValueError(
            "Cannot analyze matrix with checkpoint payloads: " + ", ".join(payloads)
        )

    manifest_by_method = {row["method"]: row for row in rows}
    raw = {}
    summary_paths = {}
    diagnostics_paths = {}
    diagnostics = {}
    actual_deltas = {}
    for method in METHODS:
        arm_dir = root / method
        raw[method], summary_path = find_summary(arm_dir)
        summary_paths[method] = str(summary_path)
        diagnostics_path = arm_dir / "uam_diagnostics.jsonl"
        records = read_diagnostics(diagnostics_path)
        diagnostics_paths[method] = str(diagnostics_path)
        diagnostics[method] = _aggregate_diagnostics(records, diagnostics_path)
        actual_deltas[method] = records["actual_delta"]

        expected_parameters = METHOD_PARAMETERS[method]
        for record in records["geometry"]:
            observed_parameters = (
                record["mode"],
                record["forget_signal"],
                record["perturbation_normalization"],
            )
            if observed_parameters != expected_parameters:
                raise ValueError(
                    f"Diagnostics method mapping drift for {method}: "
                    f"expected {expected_parameters}, found {observed_parameters}"
                )

    metric_keys = {method: set(metrics) for method, metrics in raw.items()}
    if len({frozenset(keys) for keys in metric_keys.values()}) != 1:
        detail = "; ".join(
            f"{method}={sorted(metric_keys[method])}" for method in METHODS
        )
        raise ValueError(f"Metric keys differ across UAM arms: {detail}")

    baseline = raw["uam_nll"]
    deltas = {
        f"{method}_minus_uam_nll": _metric_delta(raw[method], baseline)
        for method in METHODS
    }
    return {
        "actual_deltas": actual_deltas,
        "deltas": deltas,
        "diagnostics": diagnostics,
        "diagnostics_paths": diagnostics_paths,
        "matrix_root": str(root),
        "provenance": {
            "checkpoint_payload_count": 0,
            "manifest": manifest_by_method,
            "manifest_path": str(root / "RUN_MANIFEST.tsv"),
            "manifest_rows": len(rows),
            "methods": list(METHODS),
            "successful_rows": len(rows),
        },
        "raw": raw,
        "seed_count": 1,
        "summary_paths": summary_paths,
        "updates_per_arm": 10,
    }


def _format_number(value):
    if value is None:
        return "NA"
    return f"{value:.8g}"


def render_markdown(result):
    lines = [
        "# UAM Smoke Matrix Analysis",
        "",
        "Each arm uses one seed and ten updates. This is a mechanism check; "
        "it supports no superiority claim.",
        "",
        "## Provenance",
        "",
        f"Matrix root: `{result['matrix_root']}`",
        "",
        f"Manifest: **{result['provenance']['successful_rows']}/"
        f"{result['provenance']['manifest_rows']}** successful arms.",
        "",
        f"Checkpoint payloads: **{result['provenance']['checkpoint_payload_count']}**.",
        "",
        "## Method diagnostics",
        "",
        "| Method | Mean effective ratio | Mean residual gate rate | "
        "Max kept residual orthogonality | Finalizers | Replays | Delta probes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        diagnostics = result["diagnostics"][method]
        lines.append(
            f"| {method} | "
            f"{_format_number(diagnostics['mean_effective_perturbation_ratio'])} | "
            f"{_format_number(diagnostics['mean_residual_gate_rate'])} | "
            f"{_format_number(diagnostics['max_relative_residual_orthogonality_kept'])} | "
            f"{diagnostics['finalizer_count']} | {diagnostics['replay_count']} | "
            f"{diagnostics['actual_delta_probe_count']} |"
        )

    lines.extend(
        [
            "",
            "## Metric and actual-delta probes",
            "",
            "| Method | Metric | Raw value | Delta vs uam_nll |",
            "|---|---|---:|---:|",
        ]
    )
    for method in METHODS:
        delta = result["deltas"][f"{method}_minus_uam_nll"]
        for metric in sorted(result["raw"][method]):
            lines.append(
                f"| {method} | {metric} | "
                f"{_format_number(result['raw'][method][metric])} | "
                f"{_format_number(delta[metric])} |"
            )

    lines.extend(
        [
            "",
            "| Method | Step | Coverage | Delta norm | Forget derivative | "
            "Retain derivative |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for record in result["actual_deltas"][method]:
            lines.append(
                f"| {method} | {record['update_step']} | {record['coverage']} | "
                f"{_format_number(record['parameter_delta_norm'])} | "
                f"{_format_number(record['forget_directional_derivative'])} | "
                f"{_format_number(record['retain_directional_derivative'])} |"
            )
    return "\n".join(lines)


def _write_output_temp(destination, content):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return Path(temporary_name)


def write_outputs(result, markdown_path, json_path):
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    markdown_content = render_markdown(result).rstrip("\n") + "\n"
    json_content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        temporary_paths.append(_write_output_temp(markdown_path, markdown_content))
        temporary_paths.append(_write_output_temp(json_path, json_content))
        os.replace(temporary_paths[0], markdown_path)
        temporary_paths.pop(0)
        os.replace(temporary_paths[0], json_path)
        temporary_paths.clear()
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    result = analyze_matrix(args.matrix_root)
    write_outputs(result, args.markdown, args.json)
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
