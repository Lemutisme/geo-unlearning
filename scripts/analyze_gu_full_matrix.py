#!/usr/bin/env python3
"""Reconcile and aggregate the registered GU full-matrix evidence."""

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

from lm_eval.api.metrics import pooled_sample_stderr

try:
    from scripts import run_gu_full_matrix as registry
except ModuleNotFoundError:
    import run_gu_full_matrix as registry


TOFU_FIELDS = (
    "retain_extraction_strength",
    "extraction_strength",
    "privleak",
    "model_utility",
)
MUSE_FIELDS = (
    "forget_verbmem_ROUGE",
    "forget_knowmem_ROUGE",
    "extraction_strength",
    "privleak",
    "retain_knowmem_ROUGE",
)
WMDP_FIELDS = ("wmdp_cyber/acc", "mmlu/acc")
METRIC_FIELDS = {"tofu": TOFU_FIELDS, "muse": MUSE_FIELDS, "lm_eval": WMDP_FIELDS}
TOFU_SUMMARY_FIELDS = frozenset(
    {
        "exact_memorization",
        "extraction_strength",
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "mia_gradnorm",
        "mia_loss",
        "mia_min_k",
        "mia_min_k_plus_plus",
        "mia_zlib",
        "model_utility",
        "privleak",
        "retain_extraction_strength",
    }
)
TOFU_RAW_FIELDS = frozenset(
    {
        *TOFU_SUMMARY_FIELDS,
        "ra_Q_A_PERT_Prob",
        "ra_Q_A_Prob",
        "ra_Q_A_Prob_normalised",
        "ra_Q_A_ROUGE",
        "ra_Truth_Ratio",
        "retain_Q_A_PARA_Prob",
        "retain_Q_A_PERT_Prob",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_Truth_Ratio",
        "wf_Q_A_PERT_Prob",
        "wf_Q_A_Prob",
        "wf_Q_A_Prob_normalised",
        "wf_Q_A_ROUGE",
        "wf_Truth_Ratio",
    }
)
MUSE_ENDPOINT_FIELDS = frozenset(
    {
        "exact_memorization",
        "extraction_strength",
        "forget_knowmem_ROUGE",
        "forget_verbmem_ROUGE",
        "mia_gradnorm",
        "mia_loss",
        "mia_min_k",
        "mia_min_k_plus_plus",
        "mia_reference",
        "mia_zlib",
        "privleak",
        "retain_extraction_strength",
        "retain_knowmem_ROUGE",
    }
)
ENDPOINT_FIELDS = {
    "tofu": (TOFU_SUMMARY_FIELDS, TOFU_RAW_FIELDS),
    "muse": (MUSE_ENDPOINT_FIELDS, MUSE_ENDPOINT_FIELDS),
}

TOFU_HEADER = (
    "| Method | ES Re. ↑ | ES Un. ↓ | Priv. ↑ | MU ↑ | wall-clock | peak mem |"
)
MUSE_HEADER = (
    "| Method | VerbMem ↓ | KnowMem ↓ | Extraction ↓ | Privacy/MIA | "
    "Retain utility ↑ | wall-clock | peak mem |"
)
WMDP_HEADER = "| Method | WMDP-Cyber ↓ | MMLU ↑ | wall-clock | peak mem |"
TABLES = {
    "tofu_forget01": ("TOFU forget01", TOFU_HEADER, TOFU_FIELDS),
    "tofu_forget05": ("TOFU forget05", TOFU_HEADER, TOFU_FIELDS),
    "tofu_forget10": ("TOFU forget10", TOFU_HEADER, TOFU_FIELDS),
    "muse_news": ("MUSE News", MUSE_HEADER, MUSE_FIELDS),
    "muse_books": ("MUSE Books", MUSE_HEADER, MUSE_FIELDS),
    "wmdp_cyber": ("WMDP Cyber", WMDP_HEADER, WMDP_FIELDS),
}
FORBIDDEN_WMDP_TASK = re.compile(r"wmdp[\s_./-]*bio", re.IGNORECASE)


def _read_json(path):
    payload = json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON constant: {value}")
        ),
    )
    if FORBIDDEN_WMDP_TASK.search(json.dumps(payload, sort_keys=True)):
        raise ValueError("forbidden WMDP task token in registered evidence")
    return payload


def analyze_matrix(matrix_root):
    """Validate one run root and return deterministic raw and aggregate records."""
    root = Path(matrix_root).resolve()
    if not root.is_dir():
        raise ValueError(f"matrix root is not a directory: {root}")
    for path in sorted(root.rglob("*")):
        if FORBIDDEN_WMDP_TASK.search(path.relative_to(root).as_posix()):
            raise ValueError("forbidden WMDP task token in an evidence path")

    manifest_path = root / "manifest.json"
    state_path = root / "queue_state.json"
    manifest = registry.validate_manifest(_read_json(manifest_path))
    state = registry.validate_queue_state(_read_json(state_path))
    if Path(manifest.get("output_root", "")).resolve() != root:
        raise ValueError("manifest output_root does not match the analysis root")
    if Path(state["output_root"]).resolve() != root:
        raise ValueError("queue state output_root does not match the analysis root")
    if [job["job_id"] for job in state["jobs"][:60]] != [
        job["job_id"] for job in manifest["jobs"]
    ]:
        raise ValueError("manifest and queue state Stage-1 registries differ")

    completed_by_pair = {}
    invalid = []
    incomplete = []
    for job in state["jobs"]:
        status = job["status"]
        identity = {
            "benchmark": job["benchmark"],
            "method": job["method"],
            "seed": job["seed"],
        }
        if status in {"invalid_scientific", "failed_infrastructure"}:
            invalid.append(
                {
                    **identity,
                    "status": status,
                    "reason": job["attempt_history"][-1]["evidence"]["failure_kind"],
                }
            )
            continue
        if status in {"pending", "running"}:
            incomplete.append(
                {
                    **identity,
                    "status": status,
                    "reason": "not_started" if status == "pending" else "in_progress",
                }
            )
            continue
        if status != "completed":
            raise ValueError(f"unsupported queue status for {job['job_id']}")

        output = (root / job["output_dir"]).resolve()
        if not output.is_relative_to(root) or not output.is_dir():
            raise ValueError(f"completed output directory is missing: {job['job_id']}")
        result_path = output / "JOB_RESULT.json"
        result = _read_json(result_path)
        embedded = job["attempt_history"][-1]["evidence"]
        if result != embedded:
            raise ValueError(
                f"JOB_RESULT sidecar disagrees with queue state: {job['job_id']}"
            )
        registry.validate_job_result(result, job, output, expected_status="completed")

        command = _read_json(output / result["command_path"])
        if command != registry._command_record(job, output):
            raise ValueError(f"command sidecar identity mismatch: {job['job_id']}")
        for field in ("run_log_path", "diagnostics_path"):
            if not (output / result[field]).is_file():
                raise ValueError(
                    f"completed sidecar is missing: {job['job_id']}/{field}"
                )

        summary_path = output / result["endpoint_summary_path"]
        raw_path = output / result["endpoint_raw_path"]
        expected_endpoints = {summary_path.resolve(), raw_path.resolve()}
        observed_endpoints = {
            path.resolve()
            for path in output.rglob("*.json")
            if path.name.endswith("_SUMMARY.json") or path.name.endswith("_EVAL.json")
        }
        if observed_endpoints != expected_endpoints:
            raise ValueError(f"extra or mismatched endpoints for {job['job_id']}")
        summary = _read_json(summary_path)
        raw = _read_json(raw_path)
        fields = METRIC_FIELDS[job["evaluator_kind"]]
        metrics = {}

        if job["evaluator_kind"] in ENDPOINT_FIELDS:
            summary_fields, raw_fields = ENDPOINT_FIELDS[job["evaluator_kind"]]
            if set(summary) != summary_fields or set(raw) != raw_fields:
                raise ValueError(f"wrong endpoint schema for {job['job_id']}")
            raw_aggregates = {}
            for field in raw_fields:
                entry = raw[field]
                value = entry.get("agg_value") if isinstance(entry, dict) else None
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"malformed raw aggregate metric {field} for {job['job_id']}"
                    )
                raw_aggregates[field] = value
            for field in summary_fields:
                summary_value = summary[field]
                if (
                    isinstance(summary_value, bool)
                    or not isinstance(summary_value, (int, float))
                    or not math.isfinite(summary_value)
                ):
                    raise ValueError(
                        f"malformed summary metric {field} for {job['job_id']}"
                    )
                if summary_value != raw_aggregates[field]:
                    raise ValueError(
                        f"summary/raw disagreement for {job['job_id']}/{field}"
                    )
            metrics = {field: summary[field] for field in fields}
        else:
            expected_summary = {
                "mmlu/acc",
                "mmlu/acc_stderr",
                "wmdp_cyber/acc",
                "wmdp_cyber/acc_stderr",
            }
            if (
                not isinstance(summary, dict)
                or set(summary) != expected_summary
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in summary.values()
                )
            ):
                raise ValueError(f"wrong LMEval summary schema for {job['job_id']}")
            if not isinstance(raw, dict) or set(raw) != {"mmlu", "wmdp_cyber"}:
                raise ValueError(f"wrong LMEval raw schema for {job['job_id']}")
            if (
                not isinstance(raw["mmlu"], dict)
                or not raw["mmlu"]
                or any(not name.startswith("mmlu_") for name in raw["mmlu"])
                or any(
                    not isinstance(samples, list) or len(samples) < 2
                    for samples in raw["mmlu"].values()
                )
                or not isinstance(raw["wmdp_cyber"], dict)
                or set(raw["wmdp_cyber"]) != {"wmdp_cyber"}
                or any(
                    not isinstance(samples, list) or len(samples) < 2
                    for samples in raw["wmdp_cyber"].values()
                )
            ):
                raise ValueError(f"wrong LMEval task schema for {job['job_id']}")
            for task in ("wmdp_cyber", "mmlu"):
                values = []
                sizes = []
                stderrs = []
                for samples in raw[task].values():
                    if any(not isinstance(sample, dict) for sample in samples):
                        raise ValueError(
                            f"malformed LMEval samples for {job['job_id']}/{task}"
                        )
                    task_values = [sample.get("acc") for sample in samples]
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in task_values
                    ):
                        raise ValueError(
                            f"malformed LMEval acc for {job['job_id']}/{task}"
                        )
                    values.extend(task_values)
                    sizes.append(len(task_values))
                    stderrs.append(
                        statistics.stdev(task_values) / math.sqrt(len(task_values))
                    )
                recomputed = {
                    f"{task}/acc": statistics.fmean(values),
                    f"{task}/acc_stderr": float(pooled_sample_stderr(stderrs, sizes)),
                }
                for field, value in recomputed.items():
                    if not math.isclose(
                        summary[field], value, rel_tol=1.0e-12, abs_tol=1.0e-12
                    ):
                        raise ValueError(
                            f"summary/raw disagreement for {job['job_id']}/{field}"
                        )
                metrics[f"{task}/acc"] = summary[f"{task}/acc"]

        pair = (job["benchmark"], job["method"])
        records = completed_by_pair.setdefault(pair, {})
        if job["seed"] in records:
            raise ValueError(
                f"duplicate completed seed for {job['method']}/{job['benchmark']}"
            )
        records[job["seed"]] = {
            **identity,
            "job_id": job["job_id"],
            "stage": job["stage"],
            "status": status,
            "provenance": result["provenance"],
            "job_result_path": str(result_path),
            "endpoint_summary_path": str(summary_path),
            "endpoint_raw_path": str(raw_path),
            "metrics": metrics,
            "wall_clock_seconds": result["wall_clock_seconds"],
            "peak_nvml_mib": result["peak_nvml_mib"],
        }

    complete = []
    require_complete_failures = []
    for benchmark in registry.BENCHMARKS:
        for method in registry.STAGE_ONE_METHODS:
            pair = (benchmark, method)
            records = completed_by_pair.get(pair, {})
            if set(records) != {0, 1, 2}:
                require_complete_failures.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "seeds": sorted(records),
                    }
                )
                if 0 in records:
                    observed = {
                        row["seed"]
                        for row in incomplete + invalid
                        if (row["benchmark"], row["method"]) == pair
                    }
                    for seed in sorted({1, 2} - records.keys()):
                        if seed not in observed:
                            incomplete.append(
                                {
                                    "benchmark": benchmark,
                                    "method": method,
                                    "seed": seed,
                                    "status": "missing",
                                    "reason": "missing_required_replication",
                                }
                            )
                continue
            per_seed = [records[seed] for seed in (0, 1, 2)]
            aggregate = {}
            for field in (
                *METRIC_FIELDS[registry.BENCHMARKS[benchmark]["evaluator_kind"]],
                "wall_clock_seconds",
                "peak_nvml_mib",
            ):
                values = [
                    row[field] if field in row else row["metrics"][field]
                    for row in per_seed
                ]
                aggregate[field] = {
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values),
                }
            complete.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "seeds": [0, 1, 2],
                    "per_seed": per_seed,
                    "aggregate": aggregate,
                }
            )

    benchmark_order = {name: index for index, name in enumerate(registry.BENCHMARKS)}
    method_order = {name: index for index, name in enumerate(registry.METHODS)}
    invalid.sort(
        key=lambda row: (
            benchmark_order[row["benchmark"]],
            method_order[row["method"]],
            row["seed"],
        )
    )
    incomplete.sort(
        key=lambda row: (
            benchmark_order[row["benchmark"]],
            method_order[row["method"]],
            row["seed"],
        )
    )
    report = {
        "schema_version": registry.SCHEMA_VERSION,
        "protocol": registry.PROTOCOL,
        "provenance_scope": {
            "matrix_root": str(root),
            "manifest_path": str(manifest_path),
            "queue_state_path": str(state_path),
            "registered_benchmarks": list(registry.BENCHMARKS),
            "benchmarks": {
                name: config["provenance"]
                for name, config in registry.BENCHMARKS.items()
            },
            "metric_fields": {
                kind: list(fields) for kind, fields in METRIC_FIELDS.items()
            },
            "resource_fields": ["wall_clock_seconds", "peak_nvml_mib"],
        },
        "complete": complete,
        "invalid": invalid,
        "incomplete": incomplete,
        "not_applicable": manifest["not_applicable"],
        "require_complete_failures": require_complete_failures,
    }
    if FORBIDDEN_WMDP_TASK.search(json.dumps(report, sort_keys=True)):
        raise ValueError("forbidden WMDP task token in report")
    return report


def _format_stat(aggregate, field, unit=""):
    stat = aggregate[field]
    suffix = f" {unit}" if unit else ""
    return f"{stat['mean']:.4f} ± {stat['sample_std']:.4f}{suffix}"


def render_tables(report):
    """Render the six exact benchmark tables and their factual statuses."""
    complete = {(row["benchmark"], row["method"]): row for row in report["complete"]}
    invalid = {}
    incomplete = {}
    for row in report["invalid"]:
        invalid.setdefault((row["benchmark"], row["method"]), []).append(row)
    for row in report["incomplete"]:
        incomplete.setdefault((row["benchmark"], row["method"]), []).append(row)
    not_applicable = {
        (row["benchmark"], row["method"]): row for row in report["not_applicable"]
    }
    lines = ["# GU Full Matrix Tables"]
    for benchmark, (title, header, fields) in TABLES.items():
        lines.extend(["", f"## {title}", "", header])
        lines.append("|---|" + "---:|" * (len(fields) + 2))
        for method in registry.METHODS:
            pair = (benchmark, method)
            if pair in complete:
                aggregate = complete[pair]["aggregate"]
                cells = [_format_stat(aggregate, field) for field in fields]
                cells.extend(
                    [
                        _format_stat(aggregate, "wall_clock_seconds", "s"),
                        _format_stat(aggregate, "peak_nvml_mib", "NVML MiB"),
                    ]
                )
            elif pair in not_applicable:
                text = f"N/A ({not_applicable[pair]['reason']})"
                cells = [text] * (len(fields) + 2)
            else:
                cells = ["not aggregated"] * (len(fields) + 2)
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        status_rows = [
            row
            for method in registry.STAGE_ONE_METHODS
            for row in invalid.get((benchmark, method), [])
            + incomplete.get((benchmark, method), [])
        ]
        if status_rows:
            lines.extend(["", "Status:"])
            for row in status_rows:
                lines.append(
                    f"- {row['method']} seed {row['seed']}: "
                    f"{row['status']} ({row['reason']})"
                )
    return "\n".join(lines) + "\n"


def render_report(report):
    """Render provenance and complete/invalid/incomplete/N/A inventories."""
    scope = report["provenance_scope"]
    lines = [
        "# GU Full Matrix Report",
        "",
        "## Provenance scope",
        "",
        f"- Protocol: `{report['protocol']}`",
        f"- Matrix root: `{scope['matrix_root']}`",
        f"- Manifest: `{scope['manifest_path']}`",
        f"- Queue state: `{scope['queue_state_path']}`",
        "- Resources: `wall_clock_seconds`; `peak_nvml_mib` (NVML MiB)",
        "",
        "## Complete",
        "",
    ]
    lines.extend(
        f"- {row['benchmark']}/{row['method']}: seeds 0, 1, 2"
        for row in report["complete"]
    )
    if not report["complete"]:
        lines.append("- None")
    for heading, key in (("Invalid", "invalid"), ("Incomplete", "incomplete")):
        lines.extend(["", f"## {heading}", ""])
        lines.extend(
            f"- {row['benchmark']}/{row['method']} seed {row['seed']}: "
            f"{row['status']} ({row['reason']})"
            for row in report[key]
        )
        if not report[key]:
            lines.append("- None")
    lines.extend(["", "## N/A", ""])
    lines.extend(
        f"- {row['benchmark']}/{row['method']}: {row['reason']}"
        for row in report["not_applicable"]
    )
    return "\n".join(lines) + "\n"


def _atomic_write(path, contents):
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(report, matrix_root):
    """Atomically replace the deterministic raw, table, and report outputs."""
    root = Path(matrix_root).resolve()
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(root / "GU_FULL_RAW.json", raw)
    _atomic_write(root / "GU_FULL_TABLES.md", render_tables(report))
    _atomic_write(root / "GU_FULL_REPORT.md", render_report(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = analyze_matrix(args.root)
        write_outputs(report, args.root)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.require_complete and report["require_complete_failures"]:
        print(
            "error: valid Stage-1 methods lack exact completed seeds 1 and 2",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
