#!/usr/bin/env python3
"""Run checkpoint-free method-native baseline references on two GPUs."""

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from scripts import run_gu_full_matrix as gu


SCHEMA_VERSION = 1
PROTOCOL = "baseline_reference_20260729"
GU_ARGUMENT_PREFIX = "+trainer.method_args.gu."
TERMINAL = {"completed", "invalid_scientific", "failed_infrastructure"}
METRIC_FIELDS = {
    "tofu": (
        "retain_extraction_strength",
        "extraction_strength",
        "privleak",
        "model_utility",
    ),
    "muse": (
        "forget_verbmem_ROUGE",
        "forget_knowmem_ROUGE",
        "extraction_strength",
        "privleak",
        "retain_knowmem_ROUGE",
    ),
    "lm_eval": ("wmdp_cyber/acc", "mmlu/acc"),
}
HIGHER_IS_BETTER = {
    "retain_extraction_strength",
    "privleak",
    "model_utility",
    "retain_knowmem_ROUGE",
    "mmlu/acc",
}
NVML_COMMAND = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader",
]


def build_manifest():
    source = gu.build_manifest(seed=0)
    jobs = []
    for registered in source["jobs"]:
        job = deepcopy(registered)
        job["arm"] = "baseline"
        job["status"] = "pending"
        job["output_dir"] = f'jobs/{job["job_id"]}'
        jobs.append(job)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "seed": 0,
        "jobs": jobs,
        "not_applicable": deepcopy(source["not_applicable"]),
    }


def build_command(job, output_dir):
    command = gu.build_command(job, Path(output_dir).resolve())
    return [
        argument
        for argument in command
        if not argument.startswith(GU_ARGUMENT_PREFIX)
    ]


def build_smoke_command(job, output_dir):
    command = [
        argument
        for argument in build_command(job, output_dir)
        if not argument.startswith("trainer.args.max_steps=")
        and not argument.startswith("+trainer.args.max_steps=")
        and argument != "trainer.args.do_eval=true"
    ]
    command.extend(("+trainer.args.max_steps=1", "trainer.args.do_eval=false"))
    return command


def validate_job(job):
    if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
        raise ValueError("baseline job has no valid job_id")
    expected = {
        candidate["job_id"]: candidate for candidate in build_manifest()["jobs"]
    }.get(job["job_id"])
    if expected is None:
        raise ValueError("baseline job is not registered")
    fields = (*gu.IMMUTABLE_JOB_FIELDS, "job_id", "arm")
    if any(job.get(field) != expected.get(field) for field in fields):
        raise ValueError("baseline job identity mismatch")
    return expected


def child_environment(job, physical_gpu):
    if physical_gpu not in {0, 1}:
        raise ValueError("physical_gpu must be 0 or 1")
    environment = gu.build_environment(job)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    return environment


def _command_record(job, output_dir, physical_gpu, smoke=False):
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "arm": "baseline",
        "argv": (
            build_smoke_command(job, output_dir)
            if smoke
            else build_command(job, output_dir)
        ),
        "environment_overrides": child_environment(job, physical_gpu),
        "provenance": deepcopy(job["provenance"]),
    }


def _command_identity(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_job(job, output_dir, physical_gpu, launched=None, smoke=False):
    validate_job(job)
    output = Path(output_dir).resolve()
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ValueError("baseline output must be a real directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    command_path = output / "command.json"
    run_log_path = output / "run.log"
    result_path = output / "BASELINE_JOB_RESULT.json"
    command_record = _command_record(job, output, physical_gpu, smoke=smoke)
    gu._atomic_write_json(command_path, command_record)

    monitor_warnings = []
    peak_nvml_mib = 0
    gpu_memory_sample_count = 0
    launch_error = None
    returncode = None
    started = time.monotonic()
    with run_log_path.open("w", encoding="utf-8") as run_log:
        try:
            process = subprocess.Popen(
                command_record["argv"],
                cwd=gu.CODE_ROOT,
                env={**os.environ, **command_record["environment_overrides"]},
                stdout=run_log,
                stderr=subprocess.STDOUT,
            )
            if launched is not None:
                launched(process.pid)
        except OSError as error:
            launch_error = str(error)
            run_log.write(f"subprocess launch failed: {error}\n")
        else:
            while True:
                observed = False
                try:
                    sample = subprocess.run(
                        NVML_COMMAND,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=5.0,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    monitor_warnings.append(f"NVML sampling failed: {error}")
                else:
                    if sample.returncode != 0:
                        monitor_warnings.append(
                            f"NVML sampling exited {sample.returncode}: {sample.stderr}"
                        )
                    else:
                        for line in sample.stdout.splitlines():
                            fields = [field.strip() for field in line.split(",", 1)]
                            if len(fields) != 2:
                                continue
                            try:
                                pid = int(fields[0])
                                memory = int(re.sub(r"\s*MiB\s*$", "", fields[1]))
                            except ValueError:
                                continue
                            if pid == process.pid:
                                observed = True
                                peak_nvml_mib = max(peak_nvml_mib, memory)
                        gpu_memory_sample_count += int(observed)
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(1.0)
    wall_clock_seconds = max(0.0, time.monotonic() - started)
    log_text = run_log_path.read_text(errors="replace")

    issues = []
    updates = 0
    for line in log_text.splitlines():
        opening, closing = line.find("{"), line.rfind("}")
        if opening < 0 or closing < opening:
            continue
        candidate = line[opening : closing + 1]
        try:
            record = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                record = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(record, dict) and "loss" in record:
            loss = record["loss"]
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
                issues.append("Trainer loss is nonfinite")
            updates += 1
    if updates == 0:
        issues.append("Trainer optimizer update logs are missing")

    prefix = gu.ENDPOINT_PREFIXES[job["evaluator_kind"]]
    checkpoint_pattern = re.compile(r"checkpoint-[0-9]+")
    summaries = [] if smoke else sorted(
        path
        for path in output.rglob(f"{prefix}_SUMMARY.json")
        if path.parent.name == "evals"
        and checkpoint_pattern.fullmatch(path.parent.parent.name)
    )
    raw_files = [] if smoke else sorted(
        path
        for path in output.rglob(f"{prefix}_EVAL.json")
        if path.parent.name == "evals"
        and checkpoint_pattern.fullmatch(path.parent.parent.name)
    )
    endpoint_summary_path = endpoint_raw_path = None
    if smoke:
        pass
    elif len(summaries) != 1 or len(raw_files) != 1 or (
        summaries and raw_files and summaries[0].parent != raw_files[0].parent
    ):
        issues.append(
            f"endpoint pair count mismatch: summary={len(summaries)}, raw={len(raw_files)}"
        )
    else:
        for label, path in (("summary", summaries[0]), ("raw", raw_files[0])):
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                issues.append(f"{label} endpoint is invalid: {error}")
                continue
            if not isinstance(payload, dict) or not payload or not gu._finite_tree(payload):
                issues.append(f"{label} endpoint is nonfinite or empty")
        endpoint_summary_path = summaries[0].relative_to(output).as_posix()
        endpoint_raw_path = raw_files[0].relative_to(output).as_posix()

    allowed_root = {
        "command.json",
        "run.log",
        "BASELINE_JOB_RESULT.json",
        f'{job["method"]}.log',
    }
    allowed_hydra = {
        ".hydra/config.yaml",
        ".hydra/hydra.yaml",
        ".hydra/overrides.yaml",
    }
    allowed_endpoints = {f"{prefix}_SUMMARY.json", f"{prefix}_EVAL.json"}
    forbidden = []
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        parts = relative.parts
        allowed = (
            len(parts) == 1
            and parts[0] in allowed_root
            or relative.as_posix() in allowed_hydra
            or len(parts) == 3
            and checkpoint_pattern.fullmatch(parts[0]) is not None
            and parts[1] == "evals"
            and parts[2] in allowed_endpoints
        )
        if not allowed:
            forbidden.append(relative.as_posix())
    if forbidden:
        issues.append("forbidden persistence artifact is present")

    failure_kind = None
    if launch_error is not None:
        status, failure_kind = "failed_infrastructure", "subprocess_launch"
        issues.append(f"subprocess launch failed: {launch_error}")
    elif returncode != 0:
        lowered = log_text.lower()
        if any(marker in lowered for marker in ("out of memory", "outofmemory", "cuda oom")):
            failure_kind = "oom"
        elif any(marker in lowered for marker in ("nan", "nonfinite", "non-finite", "infinite")):
            failure_kind = "nonfinite"
        else:
            failure_kind = "subprocess_exit"
        status = "invalid_scientific"
        issues.append(f"subprocess exited with code {returncode}")
    elif issues:
        status, failure_kind = "invalid_scientific", "evidence_validation"
    elif gpu_memory_sample_count == 0 or peak_nvml_mib <= 0:
        status, failure_kind = "failed_infrastructure", "resource_monitor"
        issues.append("GPU monitor never observed the direct child")
    else:
        status = "completed"

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "arm": "baseline",
        "status": status,
        "failure_kind": failure_kind,
        "returncode": returncode,
        "command_identity": _command_identity(command_record),
        "command_path": "command.json",
        "run_log_path": "run.log",
        "environment_overrides": command_record["environment_overrides"],
        "provenance": deepcopy(job["provenance"]),
        "wall_clock_seconds": wall_clock_seconds,
        "peak_nvml_mib": peak_nvml_mib,
        "gpu_memory_sample_count": gpu_memory_sample_count,
        "gpu_monitor_warning_count": len(monitor_warnings),
        "optimizer_update_count": updates,
        "endpoint_summary_path": endpoint_summary_path,
        "endpoint_raw_path": endpoint_raw_path,
        "forbidden_artifacts": sorted(forbidden),
        "issues": issues,
    }
    gu._atomic_write_json(result_path, result)
    return result


def partition(jobs, worker, workers=2):
    if type(worker) is not int or worker not in range(workers):
        raise ValueError("worker must be 0 or 1")
    return [job for index, job in enumerate(jobs) if index % workers == worker]


def admit_gpu(physical_gpu):
    if physical_gpu not in {0, 1}:
        raise ValueError("physical_gpu must be 0 or 1")
    sample = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5.0,
    )
    if sample.returncode != 0:
        raise RuntimeError(f"GPU admission failed: {sample.stderr}")
    values = []
    for line in sample.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2 and fields[0] == str(physical_gpu):
            try:
                values.append(int(fields[1]))
            except ValueError:
                continue
    if len(values) != 1:
        raise RuntimeError("GPU admission did not return one physical device")
    if values[0] >= 500:
        raise RuntimeError(f"GPU{physical_gpu} is using {values[0]} MiB")
    return values[0]


def run_worker(manifest_path, worker, physical_gpu):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["output_root"]).resolve()
    jobs = partition(manifest["jobs"], worker=worker, workers=2)
    state_path = root / f"worker{worker}_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "worker": worker,
            "physical_gpu": physical_gpu,
            "jobs": [],
        }
        for registered in jobs:
            job = deepcopy(registered)
            job.update(
                status="pending",
                attempt_count=0,
                attempt_history=[],
                pid=None,
            )
            state["jobs"].append(job)
        gu._atomic_write_json(state_path, state)

    lock_path = Path(f"/tmp/baseline_reference_gpu{physical_gpu}.lock")
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"GPU{physical_gpu} baseline worker is already active") from error

    try:
        running = [job for job in state["jobs"] if job["status"] == "running"]
        if len(running) > 1:
            raise ValueError("worker state has multiple running jobs")
        if running:
            job = running[0]
            pid = job.get("pid")
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"baseline child {pid} is still running")
            output_dir = root / job["output_dir"]
            result_path = output_dir / "BASELINE_JOB_RESULT.json"
            result = None
            if result_path.is_file():
                try:
                    candidate = json.loads(result_path.read_text())
                    expected_identity = job["attempt_history"][-1]["command_identity"]
                    if (
                        candidate.get("job_id") == job["job_id"]
                        and candidate.get("command_identity") == expected_identity
                        and candidate.get("status") in TERMINAL
                    ):
                        result = candidate
                except (OSError, json.JSONDecodeError):
                    result = None
            if result is None:
                result = {
                    "job_id": job["job_id"],
                    "status": "failed_infrastructure",
                    "failure_kind": "stale_running",
                    "command_identity": job["attempt_history"][-1][
                        "command_identity"
                    ],
                }
            job["attempt_history"][-1].update(
                status=result["status"],
                evidence=result,
            )
            job["pid"] = None
            job["status"] = (
                "pending"
                if result["status"] == "failed_infrastructure"
                and job["attempt_count"] < 2
                else result["status"]
            )
            gu._atomic_write_json(state_path, state)

        for job in state["jobs"]:
            if job["status"] in TERMINAL:
                continue
            output_dir = root / job["output_dir"]
            while job["status"] == "pending":
                admit_gpu(physical_gpu)
                job["status"] = "running"
                job["attempt_count"] += 1
                command = _command_record(job, output_dir, physical_gpu)
                attempt = {
                    "attempt": job["attempt_count"],
                    "status": "running",
                    "pid": None,
                    "argv": command["argv"],
                    "command_identity": _command_identity(command),
                }
                job["attempt_history"].append(attempt)
                gu._atomic_write_json(state_path, state)

                def record_pid(pid):
                    job["pid"] = pid
                    attempt["pid"] = pid
                    gu._atomic_write_json(state_path, state)

                result = run_job(
                    job,
                    output_dir,
                    physical_gpu,
                    launched=record_pid,
                )
                if (
                    result.get("job_id") != job["job_id"]
                    or result.get("command_identity") != attempt["command_identity"]
                    or result.get("status") not in TERMINAL
                ):
                    raise ValueError("baseline result identity mismatch")
                attempt.update(status=result["status"], evidence=result)
                job["pid"] = None
                if (
                    result["status"] == "failed_infrastructure"
                    and job["attempt_count"] < 2
                ):
                    job["status"] = "pending"
                    gu._atomic_write_json(state_path, state)
                    continue
                job["status"] = result["status"]
                gu._atomic_write_json(state_path, state)
        return state
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def improvement(field, baseline_value, gu_value):
    values = (baseline_value, gu_value)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"comparison metric {field} must be finite")
    if field in HIGHER_IS_BETTER:
        return gu_value - baseline_value
    return baseline_value - gu_value


def compare(baseline_root, gu_root):
    baseline_root = Path(baseline_root).resolve()
    gu_root = Path(gu_root).resolve()
    manifest = json.loads((baseline_root / "manifest.json").read_text())
    baseline_jobs = {}
    baseline_gpus = {}
    for worker in (0, 1):
        state = json.loads(
            (baseline_root / f"worker{worker}_state.json").read_text()
        )
        if state.get("physical_gpu") != worker:
            raise ValueError("baseline worker/GPU identity mismatch")
        for job in state["jobs"]:
            if job["job_id"] in baseline_jobs:
                raise ValueError("duplicate baseline job identity")
            if job["status"] != "completed":
                raise ValueError(f'incomplete baseline job: {job["job_id"]}')
            baseline_jobs[job["job_id"]] = job
            baseline_gpus[job["job_id"]] = worker
    if set(baseline_jobs) != {job["job_id"] for job in manifest["jobs"]}:
        raise ValueError("baseline state does not contain all registered jobs")

    gu_state = json.loads((gu_root / "queue_state.json").read_text())
    gu_jobs = {
        job["job_id"]: job
        for job in gu_state["jobs"]
        if job["seed"] == 0
    }
    if len(gu_jobs) != 60 or any(
        job["status"] != "completed" for job in gu_jobs.values()
    ):
        raise ValueError("GU seed0 matrix is not complete")

    report = {
        "scope": "seed0 reference; no variance estimate",
        "baseline_root": str(baseline_root),
        "gu_root": str(gu_root),
        "benchmarks": {benchmark: [] for benchmark in gu.BENCHMARKS},
    }
    for registered in manifest["jobs"]:
        job_id = registered["job_id"]
        baseline_job = baseline_jobs[job_id]
        gu_job = gu_jobs[job_id]
        validate_job(baseline_job)
        gu.validate_job_identity(gu_job)
        baseline_output = baseline_root / baseline_job["output_dir"]
        gu_output = gu_root / gu_job["output_dir"]
        baseline_result = json.loads(
            (baseline_output / "BASELINE_JOB_RESULT.json").read_text()
        )
        gu_result = json.loads((gu_output / "JOB_RESULT.json").read_text())
        expected_baseline_identity = _command_identity(
            _command_record(
                baseline_job,
                baseline_output,
                baseline_gpus[job_id],
            )
        )
        if (
            baseline_result.get("status") != "completed"
            or baseline_result.get("command_identity") != expected_baseline_identity
            or baseline_result.get("provenance") != baseline_job["provenance"]
        ):
            raise ValueError(f"invalid baseline evidence: {job_id}")
        gu.validate_job_result(
            gu_result,
            gu_job,
            gu_output,
            expected_status="completed",
        )
        if baseline_result["provenance"] != gu_result["provenance"]:
            raise ValueError(f"baseline/GU provenance mismatch: {job_id}")

        arm_metrics = {}
        for arm, output, result in (
            ("baseline", baseline_output, baseline_result),
            ("gu", gu_output, gu_result),
        ):
            summary = json.loads(
                (output / result["endpoint_summary_path"]).read_text()
            )
            raw = json.loads((output / result["endpoint_raw_path"]).read_text())
            fields = METRIC_FIELDS[registered["evaluator_kind"]]
            metrics = {}
            if registered["evaluator_kind"] in {"tofu", "muse"}:
                for field in fields:
                    raw_metric = raw.get(field)
                    raw_value = (
                        raw_metric.get("agg_value")
                        if isinstance(raw_metric, dict)
                        else None
                    )
                    if summary.get(field) != raw_value:
                        raise ValueError(
                            f"{arm} summary/raw mismatch: {job_id}/{field}"
                        )
                    improvement(field, raw_value, raw_value)
                    metrics[field] = raw_value
            else:
                for field, task in (
                    ("wmdp_cyber/acc", "wmdp_cyber"),
                    ("mmlu/acc", "mmlu"),
                ):
                    if not isinstance(raw.get(task), dict):
                        raise ValueError(f"wrong WMDP raw task: {task}")
                    values = [
                        sample.get("acc")
                        for samples in raw[task].values()
                        for sample in samples
                    ]
                    if not values:
                        raise ValueError(f"empty WMDP raw task: {task}")
                    mean = math.fsum(values) / len(values)
                    if not math.isclose(summary.get(field), mean, abs_tol=1e-12):
                        raise ValueError(f"{arm} summary/raw mismatch: {field}")
                    improvement(field, mean, mean)
                    metrics[field] = mean
            arm_metrics[arm] = metrics

        rows = report["benchmarks"][registered["benchmark"]]
        for field in METRIC_FIELDS[registered["evaluator_kind"]]:
            baseline_value = arm_metrics["baseline"][field]
            gu_value = arm_metrics["gu"][field]
            rows.append(
                {
                    "method": registered["method"],
                    "metric": field,
                    "baseline": baseline_value,
                    "gu": gu_value,
                    "improvement": improvement(field, baseline_value, gu_value),
                    "baseline_path": str(
                        baseline_output / baseline_result["endpoint_summary_path"]
                    ),
                    "gu_path": str(gu_output / gu_result["endpoint_summary_path"]),
                }
            )
        for field in ("wall_clock_seconds", "peak_nvml_mib"):
            baseline_value = baseline_result[field]
            gu_value = gu_result[field]
            rows.append(
                {
                    "method": registered["method"],
                    "metric": field,
                    "baseline": baseline_value,
                    "gu": gu_value,
                    "improvement": baseline_value - gu_value,
                    "baseline_path": str(
                        baseline_output / "BASELINE_JOB_RESULT.json"
                    ),
                    "gu_path": str(gu_output / "JOB_RESULT.json"),
                }
            )
    return report


def render_comparison(report):
    lines = [
        "# Baseline vs GU Seed-0 Reference",
        "",
        report["scope"],
    ]
    for benchmark, rows in report["benchmarks"].items():
        lines.extend(
            [
                "",
                f"## {benchmark}",
                "",
                "| Method | Metric | Baseline | GU | Δ improvement |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f'| {row["method"]} | {row["metric"]} | '
                f'{row["baseline"]:.6g} | {row["gu"]:.6g} | '
                f'{row["improvement"]:+.6g} |'
            )
    return "\n".join(lines) + "\n"


def write_comparison(report, baseline_root):
    root = Path(baseline_root).resolve()
    gu._atomic_write_json(root / "BASELINE_GU_RAW.json", report)
    outputs = {
        "BASELINE_GU_TABLES.md": render_comparison(report),
        "BASELINE_GU_REPORT.md": (
            "# Baseline vs GU Report\n\n"
            "Seed-0 reference only; no variance or significance estimate.\n"
        ),
    }
    for name, contents in outputs.items():
        destination = root / name
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(contents)
        os.replace(temporary, destination)


def run_smoke(manifest_path, physical_gpu, families):
    requested = tuple(families)
    if not requested or any(family not in {"tofu", "muse", "wmdp"} for family in requested):
        raise ValueError("smoke families must be tofu, muse, or wmdp")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["output_root"]).resolve()
    selected = []
    for family in requested:
        selected.append(
            next(
                job
                for job in manifest["jobs"]
                if job["benchmark"].split("_", 1)[0] == family
            )
        )

    lock = Path(f"/tmp/baseline_reference_gpu{physical_gpu}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"GPU{physical_gpu} baseline worker is already active") from error
    try:
        results = []
        for job in selected:
            admit_gpu(physical_gpu)
            output = root / "smoke" / f"gpu{physical_gpu}" / job["job_id"]
            results.append(run_job(job, output, physical_gpu, smoke=True))
        return results
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = commands.add_parser("manifest")
    manifest_parser.add_argument("--output-root", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    worker_parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline-root", type=Path, required=True)
    compare_parser.add_argument("--gu-root", type=Path, required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--manifest", type=Path, required=True)
    smoke_parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    smoke_parser.add_argument("--benchmarks", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "manifest":
            root = args.output_root.resolve()
            root.mkdir(parents=True, exist_ok=True)
            manifest = build_manifest()
            manifest["output_root"] = str(root)
            gu._atomic_write_json(root / "manifest.json", manifest)
            print(root / "manifest.json")
            return 0
        if args.command == "worker":
            state = run_worker(args.manifest, args.worker, args.gpu)
            counts = dict(Counter(job["status"] for job in state["jobs"]))
            print(json.dumps(counts, sort_keys=True))
            return int(any(job["status"] != "completed" for job in state["jobs"]))
        if args.command == "status":
            counts = Counter()
            for worker in (0, 1):
                path = args.root.resolve() / f"worker{worker}_state.json"
                if path.is_file():
                    state = json.loads(path.read_text())
                    counts.update(job["status"] for job in state["jobs"])
            print(json.dumps(dict(counts), sort_keys=True))
            return 0
        if args.command == "compare":
            report = compare(args.baseline_root, args.gu_root)
            write_comparison(report, args.baseline_root)
            print(args.baseline_root.resolve() / "BASELINE_GU_TABLES.md")
            return 0
        if args.command == "smoke":
            results = run_smoke(
                args.manifest,
                args.gpu,
                tuple(item for item in args.benchmarks.split(",") if item),
            )
            print(json.dumps(results, indent=2, sort_keys=True))
            return int(any(result["status"] != "completed" for result in results))
        manifest = json.loads(args.manifest.read_text())
        fingerprints = {}
        memo = {}
        for job in manifest["jobs"]:
            gu.validate_job_identity(job)
            fingerprints[job["job_id"]] = gu.validate_sources(job, memo)
        print(json.dumps(fingerprints, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        print(f"{args.command}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
