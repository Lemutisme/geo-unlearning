#!/usr/bin/env python3

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import yaml


DATASETS = ("tofu01", "tofu05", "tofu10", "muse_news", "muse_books")
METHODS = ("control", "gu", "pcgrad")
ALLOWED_METHOD_DIFFERENCES = {
    "task_name",
    "paths.output_dir",
    "trainer.method_args.geometric_config.gu_enabled",
    "trainer.method_args.geometric_config.gradient_surgery",
    "trainer.method_args.geometric_config.diagnostics_path",
    "trainer.method_args.geometric_config.actual_delta_mode",
}
CHECKPOINT_PAYLOAD_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON: {path}") from error


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed diagnostics JSONL at {path}:{line_number}"
                ) from error
    return records


def mean(values):
    return sum(values) / len(values) if values else None


def aggregate_geometry(path):
    records = [
        record
        for record in load_jsonl(path)
        if record.get("record_type", "geometry") == "geometry"
    ]
    if not records:
        raise ValueError(f"No geometry records found in {path}")
    required = {
        "conflict",
        "relative_orthogonality_residual",
        "relative_surgery_magnitude",
    }
    for index, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"Geometry record {index} in {path} is missing {sorted(missing)}"
            )
    distances = [
        record["relative_pcgrad_gu_distance"]
        for record in records
        if record.get("relative_pcgrad_gu_distance") is not None
    ]
    return {
        "surgery_count": len(records),
        "conflict_rate": mean([float(record["conflict"]) for record in records]),
        "maximum_relative_orthogonality_residual": max(
            record["relative_orthogonality_residual"] for record in records
        ),
        "mean_relative_surgery_magnitude": mean(
            [record["relative_surgery_magnitude"] for record in records]
        ),
        "mean_relative_pcgrad_gu_distance": mean(distances),
        "pcgrad_gu_distance_count": len(distances),
    }


def actual_delta_records(path):
    records = [
        record
        for record in load_jsonl(path)
        if record.get("record_type") == "actual_delta"
    ]
    required = {
        "update_step",
        "coverage",
        "parameter_delta_norm",
        "forget_directional_derivative",
        "retain_directional_derivative",
    }
    for index, record in enumerate(records, start=1):
        missing = required - record.keys()
        if missing:
            raise ValueError(
                f"Actual-delta record {index} in {path} is missing {sorted(missing)}"
            )
    return records


def flatten_dict(value, prefix=""):
    flattened = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_dict(value[key], child_prefix))
    elif isinstance(value, list):
        flattened[prefix] = tuple(value)
    else:
        flattened[prefix] = value
    return flattened


def normalized_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Hydra config is not a mapping: {path}")
    flattened = flatten_dict(deepcopy(config))
    for key in ALLOWED_METHOD_DIFFERENCES:
        flattened.pop(key, None)
    return flattened


def validate_matched_configs(dataset, arm_dirs):
    normalized = {}
    for method, arm_dir in arm_dirs.items():
        config_path = arm_dir / ".hydra" / "config.yaml"
        if not config_path.is_file():
            raise ValueError(
                f"Missing Hydra config for {dataset}/{method}: {config_path}"
            )
        normalized[method] = normalized_config(config_path)

    reference = normalized["control"]
    for method in ("gu", "pcgrad"):
        if normalized[method] != reference:
            keys = sorted(
                key
                for key in reference.keys() | normalized[method].keys()
                if reference.get(key) != normalized[method].get(key)
            )
            raise ValueError(
                f"Matched configuration drift for {dataset}/{method}: "
                + ", ".join(keys[:10])
            )


def read_manifest(root):
    path = root / "RUN_MANIFEST.tsv"
    if not path.is_file():
        raise ValueError(f"Missing run manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"dataset", "method", "system_mode", "exit_code"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid run manifest schema: {path}")
    failed = [row for row in rows if int(row["exit_code"]) != 0]
    if failed:
        raise ValueError(
            "Cannot analyze failed arms: "
            + ", ".join(
                f"{row['dataset']}/{row['method']}/{row['system_mode']}"
                for row in failed
            )
        )
    return rows


def checkpoint_payloads(root):
    return sorted(
        {
            str(path)
            for pattern in CHECKPOINT_PAYLOAD_PATTERNS
            for path in root.rglob(pattern)
            if path.is_file()
        }
    )


def arm_directory(root, row):
    return root / f"{row['dataset']}_{row['method']}_{row['system_mode']}"


def find_summary(arm_dir, dataset):
    name = "MUSE_SUMMARY.json" if dataset.startswith("muse") else "TOFU_SUMMARY.json"
    matches = sorted(arm_dir.rglob(name))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {name} under {arm_dir}, found {len(matches)}"
        )
    summary = load_json(matches[0])
    if not summary:
        raise ValueError(f"Empty metric summary: {matches[0]}")
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric {key} in {matches[0]} is not numeric")
    return summary, matches[0]


def metric_delta(left, right):
    return {metric: left[metric] - right[metric] for metric in sorted(left)}


def analyze_matrix(matrix_root):
    root = Path(matrix_root)
    rows = read_manifest(root)
    production = {
        (row["dataset"], row["method"]): row
        for row in rows
        if row["system_mode"] == "production"
    }
    result = {
        "matrix_root": str(root),
        "provenance": {
            "manifest_path": str(root / "RUN_MANIFEST.tsv"),
            "manifest_rows": len(rows),
            "successful_rows": len(rows),
            "matched_hydra_configs": True,
            "checkpoint_payload_count": len(checkpoint_payloads(root)),
        },
        "datasets": {},
        "systems": {},
    }

    for dataset in DATASETS:
        missing = [method for method in METHODS if (dataset, method) not in production]
        if missing:
            raise ValueError(f"Missing production arms for {dataset}: {missing}")
        arm_dirs = {
            method: arm_directory(root, production[(dataset, method)])
            for method in METHODS
        }
        validate_matched_configs(dataset, arm_dirs)
        raw = {}
        summary_paths = {}
        geometry = {}
        actual_deltas = {}
        for method in METHODS:
            raw[method], path = find_summary(arm_dirs[method], dataset)
            summary_paths[method] = str(path)
            if method != "control":
                diagnostics_path = arm_dirs[method] / "gu_diagnostics.jsonl"
                if not diagnostics_path.is_file():
                    raise ValueError(f"Missing diagnostics: {diagnostics_path}")
                geometry[method] = aggregate_geometry(diagnostics_path)
                actual_deltas[method] = actual_delta_records(diagnostics_path)

        metric_keys = {method: set(values) for method, values in raw.items()}
        if len({frozenset(keys) for keys in metric_keys.values()}) != 1:
            raise ValueError(
                f"Metric keys differ for {dataset}: "
                + "; ".join(
                    f"{method}={sorted(keys)}" for method, keys in metric_keys.items()
                )
            )

        pcgrad_geometry = geometry["pcgrad"]
        conflict_rate = pcgrad_geometry["conflict_rate"]
        distance = pcgrad_geometry["mean_relative_pcgrad_gu_distance"]
        if conflict_rate is None or distance is None:
            raise ValueError(f"Missing PCGrad decision diagnostics for {dataset}")
        advance = conflict_rate < 0.95 or distance > 1e-4
        degenerate = conflict_rate == 1.0 and distance <= 1e-6
        result["datasets"][dataset] = {
            "raw": raw,
            "summary_paths": summary_paths,
            "deltas": {
                "gu_minus_control": metric_delta(raw["gu"], raw["control"]),
                "pcgrad_minus_control": metric_delta(raw["pcgrad"], raw["control"]),
                "pcgrad_minus_gu": metric_delta(raw["pcgrad"], raw["gu"]),
            },
            "geometry": geometry,
            "actual_deltas": actual_deltas,
            "conflict_rate": conflict_rate,
            "mean_relative_pcgrad_gu_distance": distance,
            "degenerate_to_gu": degenerate,
            "advance": advance,
            "seed_count": 1,
        }

    result["global_stop"] = all(
        row["conflict_rate"] == 1.0 and row["mean_relative_pcgrad_gu_distance"] <= 1e-6
        for row in result["datasets"].values()
    )

    for row in rows:
        if row["system_mode"] == "production":
            continue
        arm_dir = arm_directory(root, row)
        summary, summary_path = find_summary(arm_dir, row["dataset"])
        diagnostics_path = arm_dir / "gu_diagnostics.jsonl"
        system_key = f"{row['dataset']}/{row['method']}/{row['system_mode']}"
        production_reference = result["datasets"][row["dataset"]]["raw"][row["method"]]
        if set(summary) != set(production_reference):
            raise ValueError(
                f"Metric keys differ for system arm {system_key}: "
                f"system={sorted(summary)}; production={sorted(production_reference)}"
            )
        result["systems"][system_key] = {
            "raw": summary,
            "delta_from_production": metric_delta(summary, production_reference),
            "summary_path": str(summary_path),
            "geometry": (
                aggregate_geometry(diagnostics_path)
                if diagnostics_path.is_file()
                else None
            ),
            "actual_deltas": (
                actual_delta_records(diagnostics_path)
                if diagnostics_path.is_file()
                else []
            ),
        }
    return result


def format_number(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.8g}"


def render_markdown(result):
    lines = [
        "# PCGrad Multi-Dataset Smoke Analysis",
        "",
        "Each arm uses one seed and ten optimizer updates. These are mechanism "
        "checks, not final performance claims.",
        "",
        "## Provenance",
        "",
        f"Matrix root: `{result['matrix_root']}`",
        "",
        f"Manifest: **{result['provenance']['successful_rows']}/"
        f"{result['provenance']['manifest_rows']}** successful arms.",
        "",
        "Hydra configs: matched across control, GU, and PCGrad after excluding "
        "method-only fields.",
        "",
        f"Checkpoint payloads: **{result['provenance']['checkpoint_payload_count']}**.",
        "",
        "| Dataset | Conflict rate | Mean PCGrad–GU distance | Degenerate | Advance |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for dataset, row in result["datasets"].items():
        lines.append(
            f"| {dataset} | {format_number(row['conflict_rate'])} | "
            f"{format_number(row['mean_relative_pcgrad_gu_distance'])} | "
            f"{format_number(row['degenerate_to_gu'])} | "
            f"{format_number(row['advance'])} |"
        )
    lines.extend(
        [
            "",
            f"Cross-dataset stop: **{format_number(result['global_stop'])}**.",
            "",
        ]
    )

    for dataset, row in result["datasets"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Metric | Control | GU | PCGrad | GU−Control | PCGrad−Control | PCGrad−GU |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in sorted(row["raw"]["control"]):
            lines.append(
                f"| {metric} | {format_number(row['raw']['control'][metric])} | "
                f"{format_number(row['raw']['gu'][metric])} | "
                f"{format_number(row['raw']['pcgrad'][metric])} | "
                f"{format_number(row['deltas']['gu_minus_control'][metric])} | "
                f"{format_number(row['deltas']['pcgrad_minus_control'][metric])} | "
                f"{format_number(row['deltas']['pcgrad_minus_gu'][metric])} |"
            )
        lines.append("")

    if result["systems"]:
        lines.extend(
            [
                "## Systems isolation",
                "",
                "| Arm | Conflict rate | Mean PCGrad–GU distance |",
                "|---|---:|---:|",
            ]
        )
        for name, row in sorted(result["systems"].items()):
            geometry = row["geometry"] or {}
            lines.append(
                f"| {name} | {format_number(geometry.get('conflict_rate'))} | "
                f"{format_number(geometry.get('mean_relative_pcgrad_gu_distance'))} |"
            )
        lines.extend(
            [
                "",
                "### System metric comparison",
                "",
                "| Arm | Metric | Value | Delta vs production |",
                "|---|---|---:|---:|",
            ]
        )
        for name, row in sorted(result["systems"].items()):
            for metric in sorted(row["raw"]):
                lines.append(
                    f"| {name} | {metric} | {format_number(row['raw'][metric])} | "
                    f"{format_number(row['delta_from_production'][metric])} |"
                )

        lines.append("")

    lines.extend(
        [
            "## Actual update probes",
            "",
            "| Arm | Step | Coverage | Delta norm | Forget dot | Retain dot |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for dataset, row in result["datasets"].items():
        for method, records in row["actual_deltas"].items():
            for record in records:
                lines.append(
                    f"| {dataset}/{method}/production | "
                    f"{record['update_step']} | {record['coverage']} | "
                    f"{format_number(record['parameter_delta_norm'])} | "
                    f"{format_number(record['forget_directional_derivative'])} | "
                    f"{format_number(record['retain_directional_derivative'])} |"
                )
    for name, row in sorted(result["systems"].items()):
        for record in row["actual_deltas"]:
            lines.append(
                f"| {name} | {record['update_step']} | {record['coverage']} | "
                f"{format_number(record['parameter_delta_norm'])} | "
                f"{format_number(record['forget_directional_derivative'])} | "
                f"{format_number(record['retain_directional_derivative'])} |"
            )
    return "\n".join(lines)


def write_outputs(result, markdown_path, json_path):
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(result).rstrip() + "\n")
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_matrix(args.matrix_root)
    write_outputs(result, args.markdown, args.json)
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
