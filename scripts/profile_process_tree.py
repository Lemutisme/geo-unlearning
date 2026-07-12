#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import psutil


_ACTIVE_PROCESS = None


def query_gpu_used_bytes(gpu_index):
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(gpu_index)}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi memory.used query failed: " + result.stderr.strip()
        )
    try:
        used_mib = int(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError(
            "nvidia-smi memory.used query returned malformed output: "
            + result.stdout.strip()
        ) from error
    return used_mib * 1024 * 1024


def process_tree_rss_bytes(pid):
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            continue
    return total


def _atomic_write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def profile_command(
    command,
    output_path,
    gpu_index,
    poll_ms=100,
    gpu_query=query_gpu_used_bytes,
):
    global _ACTIVE_PROCESS
    command = [str(part) for part in command]
    if not command:
        raise ValueError("Profiled command must not be empty.")
    poll_ms = int(poll_ms)
    if poll_ms <= 0:
        raise ValueError("poll_ms must be positive.")

    gpu_baseline = int(gpu_query(gpu_index))
    gpu_peak = gpu_baseline
    peak_rss = 0
    sample_count = 0
    start_ns = time.perf_counter_ns()
    process = subprocess.Popen(command, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        while True:
            peak_rss = max(peak_rss, process_tree_rss_bytes(process.pid))
            gpu_peak = max(gpu_peak, int(gpu_query(gpu_index)))
            sample_count += 1
            exit_code = process.poll()
            if exit_code is not None:
                break
            time.sleep(poll_ms / 1000.0)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        _ACTIVE_PROCESS = None

    wall_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000.0
    record = {
        "record_type": "process_tree_resource_profile",
        "command": command,
        "exit_code": int(exit_code),
        "gpu_index": int(gpu_index),
        "poll_ms": poll_ms,
        "sample_count": sample_count,
        "wall_seconds": wall_seconds,
        "peak_process_tree_rss_bytes": int(peak_rss),
        "gpu_baseline_used_bytes": gpu_baseline,
        "gpu_peak_used_bytes": gpu_peak,
        "gpu_peak_delta_bytes": max(0, gpu_peak - gpu_baseline),
    }
    _atomic_write_json(output_path, record)
    return int(exit_code), record


def _forward_signal(signum, _frame):
    process = _ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signum)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--poll-ms", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main():
    args = parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _forward_signal)
    exit_code, _ = profile_command(
        args.command,
        args.output,
        gpu_index=args.gpu,
        poll_ms=args.poll_ms,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
