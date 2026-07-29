#!/usr/bin/env python3
"""Run checkpoint-free method-native baseline references on two GPUs."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from scripts import run_gu_full_matrix as gu


SCHEMA_VERSION = 1
PROTOCOL = "baseline_reference_20260729"
GU_ARGUMENT_PREFIX = "+trainer.method_args.gu."
TERMINAL = {"completed", "invalid_scientific", "failed_infrastructure"}
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


def _command_record(job, output_dir, physical_gpu):
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "benchmark": job["benchmark"],
        "seed": job["seed"],
        "arm": "baseline",
        "argv": build_command(job, output_dir),
        "environment_overrides": child_environment(job, physical_gpu),
        "provenance": deepcopy(job["provenance"]),
    }


def _command_identity(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_job(job, output_dir, physical_gpu):
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
    command_record = _command_record(job, output, physical_gpu)
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
    summaries = sorted(
        path
        for path in output.rglob(f"{prefix}_SUMMARY.json")
        if path.parent.name == "evals"
        and checkpoint_pattern.fullmatch(path.parent.parent.name)
    )
    raw_files = sorted(
        path
        for path in output.rglob(f"{prefix}_EVAL.json")
        if path.parent.name == "evals"
        and checkpoint_pattern.fullmatch(path.parent.parent.name)
    )
    endpoint_summary_path = endpoint_raw_path = None
    if len(summaries) != 1 or len(raw_files) != 1 or (
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = commands.add_parser("manifest")
    manifest_parser.add_argument("--output-root", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
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
