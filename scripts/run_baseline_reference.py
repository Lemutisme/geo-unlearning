#!/usr/bin/env python3
"""Run checkpoint-free method-native baseline references on two GPUs."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from scripts import run_gu_full_matrix as gu


SCHEMA_VERSION = 1
PROTOCOL = "baseline_reference_20260729"
GU_ARGUMENT_PREFIX = "+trainer.method_args.gu."


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
