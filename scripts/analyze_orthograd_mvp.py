#!/usr/bin/env python3

import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

import yaml


BENCHMARKS = (
    "tofu01",
    "tofu05",
    "tofu10",
    "muse_news",
    "muse_books",
    "wmdp_cyber",
    "wmdp_bio",
)
METHODS = ("control", "orthograd", "gu_euclidean", "gu_adam")
METHOD_LABELS = {
    "control": "SimNPO (base)",
    "orthograd": "+ OrthoGrad-style (per-sample QR, Euclidean)",
    "gu_euclidean": "+ GU, Euclidean metric (Table 5)",
    "gu_adam": "+ GU, Adam metric (ours)",
}
PAYLOAD_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")
ALLOWED_CONFIG_DIFFERENCES = {
    "task_name",
    "paths.output_dir",
    "trainer.args.logging_dir",
    "trainer.handler",
    "trainer.method_args.geometric_config.gu_enabled",
    "trainer.method_args.geometric_config.optimizer_geometry",
    "trainer.method_args.geometric_config.diagnostics_path",
    "trainer.method_args.geometric_config.resource_profile_path",
}
ALLOWED_CONFIG_PREFIXES = ("trainer.method_args.orthograd_config",)
SUMMARY_NAMES = {
    "tofu": "TOFU_SUMMARY.json",
    "muse": "MUSE_SUMMARY.json",
    "wmdp": "LMEval_SUMMARY.json",
}


def family(benchmark):
    return benchmark.split("_", 1)[0].rstrip("0123456789")


def load_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Non-object JSONL record at {path}:{line_number}")
            records.append(record)
    return records


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def stable_mean(values):
    values = [finite_number(value, "aggregate value") for value in values]
    if not values:
        return None
    return math.fsum(values) / len(values)


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value[key], child))
    elif isinstance(value, list):
        result[prefix] = tuple(value)
    else:
        result[prefix] = value
    return result


def normalized_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Hydra config is not a mapping: {path}")
    result = flatten(deepcopy(value))
    for key in list(result):
        if key in ALLOWED_CONFIG_DIFFERENCES or any(
            key == prefix or key.startswith(prefix + ".")
            for prefix in ALLOWED_CONFIG_PREFIXES
        ):
            result.pop(key)
    return result


def validate_method_config(benchmark, method, path):
    value = yaml.safe_load(Path(path).read_text())
    trainer = value.get("trainer", {})
    handler = trainer.get("handler")
    geometry = trainer.get("method_args", {}).get("geometric_config", {})
    expected_handler = (
        "OrthogradUnlearn" if method == "orthograd" else "GeometricUnlearn"
    )
    if handler != expected_handler:
        raise ValueError(f"Invalid trainer mapping for {benchmark}/{method}: {handler}")
    if method == "control" and geometry.get("gu_enabled") is not False:
        raise ValueError(f"Control GU must be disabled for {benchmark}")
    if method in {"gu_euclidean", "gu_adam"} and geometry.get("gu_enabled") is not True:
        raise ValueError(f"GU must be enabled for {benchmark}/{method}")
    expected_geometry = {
        "orthograd": "euclidean",
        "gu_euclidean": "euclidean",
        "gu_adam": "adam",
    }.get(method)
    if expected_geometry and geometry.get("optimizer_geometry") != expected_geometry:
        raise ValueError(f"Invalid optimizer geometry for {benchmark}/{method}")


def read_manifest(root):
    path = root / "RUN_MANIFEST.tsv"
    if not path.is_file():
        raise ValueError(f"Missing run manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "benchmark",
        "method",
        "gpu",
        "start_utc",
        "end_utc",
        "exit_code",
        "command",
        "arm_dir",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid run manifest schema: {path}")
    expected = {(benchmark, method) for benchmark in BENCHMARKS for method in METHODS}
    observed = [(row["benchmark"], row["method"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate benchmark/method rows in run manifest")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"Incomplete run manifest; missing={missing}, extra={extra}")
    failed = [row for row in rows if int(row["exit_code"]) != 0]
    if failed:
        names = ", ".join(f"{row['benchmark']}/{row['method']}" for row in failed)
        raise ValueError(f"Cannot analyze failed arms: {names}")
    return rows


def checkpoint_payloads(root):
    payloads = {
        path
        for pattern in PAYLOAD_PATTERNS
        for path in root.rglob(pattern)
        if path.is_file()
    }
    for name in ("trainer_state.json", "optimizer", "scheduler", "rng_state"):
        payloads.update(path for path in root.rglob(f"{name}*") if path.is_file())
    return sorted(str(path) for path in payloads)


def symlinks(root):
    return sorted(str(path) for path in root.rglob("*") if path.is_symlink())


def validate_summary(path):
    summary = load_json(path)
    if not summary:
        raise ValueError(f"Empty metric summary: {path}")
    return {
        key: finite_number(value, f"metric {key} in {path}")
        for key, value in sorted(summary.items())
    }


def table_metrics(benchmark, summary):
    current_family = family(benchmark)
    if current_family == "tofu":
        mapping = {
            "ES Re.": "retain_extraction_strength",
            "ES Un.": "extraction_strength",
            "Priv.": "privleak",
            "MU": "model_utility",
        }
    elif current_family == "muse":
        mapping = {
            "Forget KnowMem": "forget_knowmem_ROUGE",
            "Forget VerbMem": "forget_verbmem_ROUGE",
            "Priv.": "privleak",
            "Retain KnowMem": "retain_knowmem_ROUGE",
        }
    else:
        task = "wmdp_cyber" if benchmark.endswith("cyber") else "wmdp_bio"
        mapping = {"WMDP accuracy": f"{task}/acc", "MMLU accuracy": "mmlu/acc"}
    missing = sorted(set(mapping.values()) - summary.keys())
    if missing:
        raise ValueError(f"Missing table metrics for {benchmark}: {missing}")
    return {label: summary[key] for label, key in mapping.items()}


def validate_training_resource(path):
    value = load_json(path)
    if value.get("record_type") != "training_resource_profile":
        raise ValueError(f"Invalid training resource record: {path}")
    if int(value.get("update_count", -1)) != 10:
        raise ValueError(f"Expected ten optimizer updates: {path}")
    numeric = (
        "train_wall_seconds",
        "first_update_ms",
        "steady_update_mean_ms",
        "steady_update_p50_ms",
        "steady_update_p95_ms",
        "cuda_max_allocated_bytes",
        "cuda_max_reserved_bytes",
    )
    for key in numeric:
        finite_number(value.get(key), f"{key} in {path}")
    if value["cuda_max_reserved_bytes"] < value["cuda_max_allocated_bytes"]:
        raise ValueError(f"CUDA reserved memory below allocated memory: {path}")
    for key, item in value.get("phase_totals_ms", {}).items():
        finite_number(item, f"phase {key} in {path}")
    return value


def validate_process_resource(path):
    value = load_json(path)
    if value.get("record_type") != "process_tree_resource_profile":
        raise ValueError(f"Invalid process resource record: {path}")
    if int(value.get("exit_code", -1)) != 0:
        raise ValueError(f"Nonzero profiled process exit: {path}")
    for key in (
        "wall_seconds",
        "peak_process_tree_rss_bytes",
        "gpu_baseline_used_bytes",
        "gpu_peak_used_bytes",
        "gpu_peak_delta_bytes",
        "sample_count",
        "poll_ms",
    ):
        finite_number(value.get(key), f"{key} in {path}")
    if value["gpu_peak_used_bytes"] < value["gpu_baseline_used_bytes"]:
        raise ValueError(f"GPU peak below baseline: {path}")
    if value["sample_count"] <= 0 or value["poll_ms"] <= 0:
        raise ValueError(f"Invalid sampler coverage: {path}")
    return value


def aggregate_diagnostics(method, path):
    records = load_jsonl(path)
    record_type = "orthograd_geometry" if method == "orthograd" else "geometry"
    geometry = [
        record for record in records if record.get("record_type") == record_type
    ]
    if len(geometry) != 10:
        raise ValueError(f"Expected ten {record_type} records in {path}")
    steps = [int(record.get("update_step", -1)) for record in geometry]
    if steps != list(range(1, 11)):
        raise ValueError(f"Non-contiguous geometry update steps in {path}")
    if method == "orthograd":
        for record in geometry:
            requested = int(record.get("requested_rank", 0))
            effective = int(record.get("effective_rank", 0))
            if requested <= 0 or effective <= 0 or effective > requested:
                raise ValueError(f"Invalid Orthograd rank in {path}")
        residual = max(
            finite_number(record.get("maximum_relative_residual"), "Orthograd residual")
            for record in geometry
        )
        if residual > 1e-4:
            raise ValueError(f"Orthograd residual exceeds tolerance in {path}")
        return {
            "geometry_count": len(geometry),
            "mean_effective_rank": stable_mean(
                record["effective_rank"] for record in geometry
            ),
            "mean_requested_rank": stable_mean(
                record["requested_rank"] for record in geometry
            ),
            "mean_preserved_signal_ratio": stable_mean(
                record["preserved_signal_ratio"] for record in geometry
            ),
            "maximum_relative_residual": residual,
            "maximum_basis_bytes": max(
                int(record["basis_bytes"]) for record in geometry
            ),
            "total_estimated_projection_flops": int(
                math.fsum(
                    finite_number(
                        record["estimated_projection_flops"], "projection FLOPs"
                    )
                    for record in geometry
                )
            ),
        }
    residual = max(
        finite_number(record.get("relative_orthogonality_residual"), "GU residual")
        for record in geometry
    )
    if residual > 1e-4:
        raise ValueError(f"GU residual exceeds tolerance in {path}")
    return {
        "geometry_count": len(geometry),
        "maximum_relative_residual": residual,
        "maximum_basis_bytes": 0,
        "total_estimated_projection_flops": None,
    }


def validate_wmdp_provenance(benchmark, path):
    value = load_json(path)
    if value.get("dataset") != "cais/wmdp-corpora":
        raise ValueError(f"Unexpected WMDP source in {path}")
    if value.get("revision") != "daf89fa9b618b63a624228061a9cebacca88009c":
        raise ValueError(f"Unpinned WMDP revision in {path}")
    files = value.get("files", {})
    split = benchmark.removeprefix("wmdp_")
    for kind in ("forget", "retain"):
        record = files.get(f"{split}-{kind}", files.get(kind))
        digest = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Missing WMDP {kind} SHA-256 in {path}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"Malformed WMDP {kind} SHA-256 in {path}") from error
    return value


def safe_overhead(value, baseline, label):
    value = finite_number(value, label)
    baseline = finite_number(baseline, f"baseline {label}")
    if baseline <= 0:
        raise ValueError(f"Baseline {label} must be positive")
    return value / baseline - 1.0


def analyze_matrix(matrix_root):
    root = Path(matrix_root)
    rows = read_manifest(root)
    payloads = checkpoint_payloads(root)
    if payloads:
        raise ValueError(f"Found checkpoint payload: {payloads[0]}")
    links = symlinks(root)
    if links:
        raise ValueError(f"Found symbolic link: {links[0]}")
    row_map = {(row["benchmark"], row["method"]): row for row in rows}
    result = {
        "provenance": {
            "matrix_root": str(root),
            "manifest_path": str(root / "RUN_MANIFEST.tsv"),
            "manifest_rows": len(rows),
            "successful_rows": len(rows),
            "checkpoint_payload_count": 0,
            "symbolic_link_count": 0,
            "single_seed": 0,
            "optimizer_updates": 10,
        },
        "benchmarks": {},
    }

    for benchmark in BENCHMARKS:
        arms = {}
        normalized = {}
        summary_keys = None
        for method in METHODS:
            row = row_map[(benchmark, method)]
            arm = Path(row["arm_dir"])
            if not arm.is_absolute():
                arm = root / arm
            config_path = arm / ".hydra" / "config.yaml"
            if not config_path.is_file():
                raise ValueError(f"Missing Hydra config: {config_path}")
            validate_method_config(benchmark, method, config_path)
            normalized[method] = normalized_config(config_path)
            summary_path = arm / SUMMARY_NAMES[family(benchmark)]
            if not summary_path.is_file():
                raise ValueError(f"Missing metric summary: {summary_path}")
            summary = validate_summary(summary_path)
            if summary_keys is None:
                summary_keys = set(summary)
            elif set(summary) != summary_keys:
                raise ValueError(f"Summary metric keys differ for {benchmark}/{method}")
            training = validate_training_resource(
                arm / "training_resource_profile.json"
            )
            process = validate_process_resource(arm / "process_resource_profile.json")
            diagnostics = None
            if method != "control":
                diagnostics = aggregate_diagnostics(method, arm / "diagnostics.jsonl")
            provenance = None
            if family(benchmark) == "wmdp":
                provenance = validate_wmdp_provenance(
                    benchmark, arm / "wmdp_provenance.json"
                )
            arms[method] = {
                "arm_dir": str(arm),
                "summary_path": str(summary_path),
                "raw_summary": summary,
                "metrics": table_metrics(benchmark, summary),
                "training_resource": training,
                "process_resource": process,
                "diagnostics": diagnostics,
                "wmdp_provenance": provenance,
            }

        reference = normalized["control"]
        for method in METHODS[1:]:
            if normalized[method] != reference:
                differing = sorted(
                    key
                    for key in reference.keys() | normalized[method].keys()
                    if reference.get(key) != normalized[method].get(key)
                )
                raise ValueError(
                    f"Matched configuration drift for {benchmark}/{method}: "
                    + ", ".join(differing[:12])
                )

        baseline = arms["control"]
        for method, arm in arms.items():
            arm["metric_delta_from_control"] = {
                key: arm["metrics"][key] - baseline["metrics"][key]
                for key in arm["metrics"]
            }
            arm["overhead"] = {
                "wall_clock": safe_overhead(
                    arm["process_resource"]["wall_seconds"],
                    baseline["process_resource"]["wall_seconds"],
                    "wall clock",
                ),
                "train_runtime": safe_overhead(
                    arm["training_resource"]["train_wall_seconds"],
                    baseline["training_resource"]["train_wall_seconds"],
                    "train runtime",
                ),
                "allocated": safe_overhead(
                    arm["training_resource"]["cuda_max_allocated_bytes"],
                    baseline["training_resource"]["cuda_max_allocated_bytes"],
                    "allocated memory",
                ),
                "nvml": safe_overhead(
                    arm["process_resource"]["gpu_peak_delta_bytes"],
                    baseline["process_resource"]["gpu_peak_delta_bytes"],
                    "NVML memory",
                ),
            }
        result["benchmarks"][benchmark] = {"methods": arms}
    return result


def format_float(value):
    return f"{float(value):.6f}"


def gib(value):
    return f"{float(value) / (1024**3):.3f} GiB"


def render_primary_table(benchmark, methods):
    current_family = family(benchmark)
    if current_family == "tofu":
        headers = ("ES Re. ↑", "ES Un. ↓", "Priv. ↑", "MU ↑")
        labels = ("ES Re.", "ES Un.", "Priv.", "MU")
    elif current_family == "muse":
        headers = (
            "Forget KnowMem ↓",
            "Forget VerbMem ↓",
            "Priv. ↑",
            "Retain KnowMem ↑",
        )
        labels = ("Forget KnowMem", "Forget VerbMem", "Priv.", "Retain KnowMem")
    else:
        headers = ("WMDP accuracy ↓", "MMLU accuracy ↑")
        labels = ("WMDP accuracy", "MMLU accuracy")
    header = ["Method", *headers, "wall-clock", "peak mem"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method in METHODS:
        arm = methods[method]
        values = [format_float(arm["metrics"][label]) for label in labels]
        values.extend(
            [
                f"{arm['process_resource']['wall_seconds']:.3f}s",
                gib(arm["process_resource"]["gpu_peak_delta_bytes"]),
            ]
        )
        lines.append("| " + " | ".join([METHOD_LABELS[method], *values]) + " |")
    return "\n".join(lines)


def render_resource_table(methods):
    header = (
        "Method",
        "train wall",
        "wall overhead",
        "first update",
        "steady mean",
        "peak allocated",
        "peak reserved",
        "CPU RSS",
        "basis bytes",
        "max residual",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method in METHODS:
        arm = methods[method]
        train = arm["training_resource"]
        process = arm["process_resource"]
        diagnostic = arm["diagnostics"] or {}
        residual = diagnostic.get("maximum_relative_residual")
        lines.append(
            "| "
            + " | ".join(
                (
                    METHOD_LABELS[method],
                    f"{train['train_wall_seconds']:.3f}s",
                    f"{arm['overhead']['wall_clock'] * 100:.2f}%",
                    f"{train['first_update_ms']:.3f}ms",
                    f"{train['steady_update_mean_ms']:.3f}ms",
                    gib(train["cuda_max_allocated_bytes"]),
                    gib(train["cuda_max_reserved_bytes"]),
                    gib(process["peak_process_tree_rss_bytes"]),
                    str(diagnostic.get("maximum_basis_bytes", 0)),
                    "N/A" if residual is None else f"{residual:.3e}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown(result):
    lines = [
        "# OrthoGrad and GU MVP comparison",
        "",
        "Single seed (0), ten optimizer updates per arm. These results support only feasibility and overhead comparisons, not convergence or state-of-the-art claims.",
        "",
        f"Matrix root: `{result['provenance']['matrix_root']}`",
        "",
        "Checkpoint payloads: **0**. Peak memory in the primary tables is sampled device-level NVML delta above the pre-launch baseline; allocator and CPU RSS measurements are reported separately.",
    ]
    for benchmark in BENCHMARKS:
        methods = result["benchmarks"][benchmark]["methods"]
        lines.extend(
            [
                "",
                f"## {benchmark}",
                "",
                render_primary_table(benchmark, methods),
                "",
                "Resource detail:",
                "",
                render_resource_table(methods),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_outputs(result, markdown_path, json_path):
    atomic_write(markdown_path, render_markdown(result))
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(json_path, serialized)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root")
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_matrix(args.matrix_root)
    write_outputs(result, args.markdown, args.json)


if __name__ == "__main__":
    main()
