# TOFU GU Stage-A Causal Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the 20-job TOFU-forget10 Stage-A matrix that separately measures parameter scope, BF16→FP32 storage, current GU projection, and retain-history effects for CEU, GradDiff, SimNPO, and UNDIAL.

**Architecture:** Add an inert-by-default `scope_control` branch to the common trainer for matched no-projection controls. A dedicated checkpoint-free GPU0 runner builds exact commands from the pinned GU registry, executes a durable sequential queue, and a separate analyzer validates endpoints and reports adjacent causal deltas.

**Tech Stack:** Python 3.11, PyTorch 2.4.1, Transformers 4.45.1, Hydra/OmegaConf, pytest, NVML through `nvidia-smi`, JSONL diagnostics, tmux.

---

### Task 1: Add scope-only optimizer control with TDD

**Files:**
- Modify: `src/trainer/unlearn/base.py:38-282`
- Create: `tests/test_scope_control.py`

- [ ] **Step 1: Write failing scope and dtype tests**

Create `tests/test_scope_control.py`:

```python
import copy

import pytest
import torch

from tests.test_minimal_gu import (
    TinyCausalLM,
    gu_config,
    make_trainer,
    run_objective_update,
)


def scope_config(**overrides):
    config = {
        "enabled": True,
        "parameter_regex": ["protected[.]weight"],
        "selected_storage": "native",
    }
    config.update(overrides)
    return config


def test_scope_control_freezes_unselected_parameters_and_optimizer_is_exact(tmp_path):
    model = TinyCausalLM()
    trainer = make_trainer(model, tmp_path, scope_control=scope_config())

    trainer.create_optimizer()

    assert [name for name, _ in trainer._scope_control_selected] == [
        "protected.weight"
    ]
    assert model.protected.weight.requires_grad
    assert not model.embed.weight.requires_grad
    assert not model.unmatched.requires_grad
    assert [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ] == [model.protected.weight]


@pytest.mark.parametrize(
    ("storage", "expected"),
    [("native", torch.bfloat16), ("float32", torch.float32)],
)
def test_scope_control_selected_storage_is_explicit(
    tmp_path,
    storage,
    expected,
):
    model = TinyCausalLM().to(dtype=torch.bfloat16)
    trainer = make_trainer(
        model,
        tmp_path,
        scope_control=scope_config(selected_storage=storage),
    )

    trainer.create_optimizer()

    assert model.protected.weight.dtype == expected
    assert model.embed.weight.dtype == torch.bfloat16
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_scope_control.py -k 'freezes or selected_storage'
```

Expected: FAIL because `UnlearnTrainer.__init__` forwards the unknown
`scope_control` argument to Transformers Trainer.

- [ ] **Step 3: Write failing contract, conflict, and inertness tests**

Append:

```python
@pytest.mark.parametrize(
    "config",
    [
        {"enabled": True, "parameter_regex": ["protected"]},
        scope_config(parameter_regex=[]),
        scope_config(parameter_regex="protected"),
        scope_config(parameter_regex=["("]),
        scope_config(parameter_regex=["missing"]),
        scope_config(selected_storage="bf16"),
    ],
)
def test_scope_control_rejects_invalid_contract(tmp_path, config):
    trainer = make_trainer(TinyCausalLM(), tmp_path, scope_control=config)

    with pytest.raises(ValueError, match="scope_control"):
        trainer.create_optimizer()


def test_scope_control_rejects_common_gu_combination(tmp_path):
    with pytest.raises(ValueError, match="scope_control.*GU|GU.*scope_control"):
        make_trainer(
            TinyCausalLM(),
            tmp_path,
            scope_control=scope_config(),
            gu=gu_config(),
        )


def test_disabled_scope_control_matches_native_update(tmp_path):
    torch.manual_seed(37)
    native_model = TinyCausalLM()
    controlled_model = copy.deepcopy(native_model)
    batch = {
        "forget": {
            "input_ids": torch.tensor([[1, 4, 2, 8, 3]]),
            "labels": torch.tensor([[1, 4, 2, 8, 3]]),
        }
    }
    native = make_trainer(native_model, tmp_path / "native")
    disabled = make_trainer(
        controlled_model,
        tmp_path / "disabled",
        scope_control=scope_config(enabled=False),
    )

    native_loss = run_objective_update(native, copy.deepcopy(batch))
    disabled_loss = run_objective_update(disabled, copy.deepcopy(batch))

    torch.testing.assert_close(native_loss, disabled_loss, rtol=0, atol=0)
    for left, right in zip(native.model.parameters(), disabled.model.parameters()):
        assert torch.equal(left, right)
    assert not hasattr(disabled, "_scope_control_selected")


def test_scope_control_registers_no_gu_hooks_or_transient_state(tmp_path):
    trainer = make_trainer(
        TinyCausalLM(),
        tmp_path,
        scope_control=scope_config(),
    )
    trainer.create_optimizer()

    assert not trainer.optimizer._optimizer_step_pre_hooks
    assert not trainer.optimizer._optimizer_step_post_hooks
    assert not hasattr(trainer, "_gu_selected")
    assert not hasattr(trainer, "_gu_constraint_history")
```

- [ ] **Step 4: Run the expanded tests and verify RED**

Run:

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_scope_control.py
```

Expected: all tests fail before the scope-control branch exists.

- [ ] **Step 5: Implement the minimal scope-control branch**

In `UnlearnTrainer.__init__`, consume the mapping before calling the parent:

```python
def __init__(self, *args, gu=None, scope_control=None, **kwargs):
    self.gu_config = gu
    self.gu_enabled = gu is not None and gu.get("enabled") is True
    self.scope_control_config = scope_control
    self.scope_control_enabled = (
        scope_control is not None and scope_control.get("enabled") is True
    )
    if self.gu_enabled and self.scope_control_enabled:
        raise ValueError("scope_control cannot be combined with common GU")
    self.gu_projection_calls = 0
    self.gu_last_diagnostics = None
    super().__init__(*args, **kwargs)
```

At the beginning of `create_optimizer`, route a supplied scope mapping before
the existing GU branch:

```python
if self.scope_control_config is not None:
    config = self.scope_control_config
    required = {"enabled", "parameter_regex", "selected_storage"}
    if set(config) != required or not isinstance(config["enabled"], bool):
        raise ValueError("scope_control configuration keys are invalid")
    if not config["enabled"]:
        return super().create_optimizer()
    patterns = config["parameter_regex"]
    if (
        not isinstance(patterns, Sequence)
        or isinstance(patterns, (str, bytes))
        or not patterns
        or any(not isinstance(pattern, str) or not pattern for pattern in patterns)
    ):
        raise ValueError("scope_control parameter_regex is invalid")
    try:
        compiled = tuple(re.compile(pattern) for pattern in patterns)
    except re.error as error:
        raise ValueError("scope_control parameter_regex is invalid") from error
    storage = config["selected_storage"]
    if storage not in {"native", "float32"}:
        raise ValueError("scope_control selected_storage is invalid")
    selected = tuple(
        (name, parameter)
        for name, parameter in self.model.named_parameters()
        if any(pattern.search(name) for pattern in compiled)
    )
    if not selected:
        raise ValueError("scope_control parameter_regex selected no parameters")
    original = tuple(
        (parameter, parameter.requires_grad, parameter.dtype)
        for parameter in self.model.parameters()
    )
    try:
        selected_ids = {id(parameter) for _, parameter in selected}
        for _, parameter in selected:
            if storage == "float32" and parameter.dtype == torch.bfloat16:
                parameter.data = parameter.detach().float()
        for parameter in self.model.parameters():
            if id(parameter) not in selected_ids:
                parameter.requires_grad_(False)
        optimizer = super().create_optimizer()
        optimizer_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if (
            len(optimizer_ids) != len(set(optimizer_ids))
            or set(optimizer_ids) != selected_ids
        ):
            raise ValueError(
                "scope_control optimizer parameters do not match selected scope"
            )
    except Exception:
        for parameter, requires_grad, dtype in original:
            parameter.requires_grad_(requires_grad)
            if parameter.dtype != dtype:
                parameter.data = parameter.detach().to(dtype)
        raise
    self._scope_control_patterns = compiled
    self._scope_control_selected = selected
    return optimizer
```

- [ ] **Step 6: Run scope-control and GU regression tests**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_scope_control.py tests/test_minimal_gu.py
```

Expected: scope-control tests pass; existing GU tests remain green.

- [ ] **Step 7: Commit scope control**

```bash
git add src/trainer/unlearn/base.py tests/test_scope_control.py
git commit -m "feat: add matched optimizer scope control"
```

### Task 2: Build the deterministic Stage-A registry

**Files:**
- Create: `scripts/run_tofu_gu_ablation.py`
- Create: `tests/test_tofu_gu_ablation.py`

- [ ] **Step 1: Write failing manifest and command tests**

Create `tests/test_tofu_gu_ablation.py`:

```python
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_tofu_gu_ablation.py"
METHODS = ("CEU", "GradDiff", "SimNPO", "UNDIAL")
ARMS = ("native", "scope_bf16", "scope_fp32", "gu_rank0", "gu_rank8")


def load_runner():
    spec = importlib.util.spec_from_file_location("tofu_gu_ablation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_a_manifest_has_exact_twenty_jobs():
    runner = load_runner()
    manifest = runner.build_manifest()

    assert manifest["protocol"] == "tofu_gu_stage_a_20260809"
    assert manifest["stage"] == "stage_a"
    assert manifest["seed"] == 0
    assert len(manifest["jobs"]) == 20
    assert {(job["method"], job["arm"]) for job in manifest["jobs"]} == {
        (method, arm) for method in METHODS for arm in ARMS
    }
    assert [job["job_id"] for job in manifest["jobs"]] == sorted(
        job["job_id"] for job in manifest["jobs"]
    )


def test_stage_a_commands_change_exactly_one_intervention(tmp_path):
    runner = load_runner()
    jobs = runner.build_manifest()["jobs"]
    by_method = {
        method: {job["arm"]: job for job in jobs if job["method"] == method}
        for method in METHODS
    }
    for method, arms in by_method.items():
        commands = {
            arm: runner.build_command(job, tmp_path / job["job_id"])
            for arm, job in arms.items()
        }
        assert not any("method_args.gu" in value for value in commands["native"])
        assert not any("scope_control" in value for value in commands["native"])
        assert any("scope_control" in value for value in commands["scope_bf16"])
        assert any("selected_storage:native" in value for value in commands["scope_bf16"])
        assert any("selected_storage:float32" in value for value in commands["scope_fp32"])
        assert any("retain_history_rank=0" in value for value in commands["gu_rank0"])
        assert any("retain_history_rank=8" in value for value in commands["gu_rank8"])
        for command in commands.values():
            assert "save_model_after_train=false" in command
            assert "trainer.args.save_strategy=no" in command
            assert "trainer.args.save_only_model=false" in command
            assert "trainer.args.seed=0" in command
            assert "forget_split=forget10" in command
```

- [ ] **Step 2: Run registry tests and verify RED**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_tofu_gu_ablation.py -k 'manifest or commands'
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement manifest and command construction**

Create the runner with these public constants and builders:

```python
#!/usr/bin/env python3
import argparse
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

from scripts import run_baseline_reference as baseline
from scripts import run_gu_full_matrix as gu

SCHEMA_VERSION = 1
PROTOCOL = "tofu_gu_stage_a_20260809"
STAGE = "stage_a"
METHODS = ("CEU", "GradDiff", "SimNPO", "UNDIAL")
ARMS = ("native", "scope_bf16", "scope_fp32", "gu_rank0", "gu_rank8")
PARAMETER_REGEX = r"model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight"
TERMINAL = {"completed", "invalid_scientific", "failed_infrastructure"}


def _base_job(method):
    return next(
        deepcopy(job)
        for job in gu.build_manifest(seed=0)["jobs"]
        if job["method"] == method and job["benchmark"] == "tofu_forget10"
    )


def build_manifest():
    jobs = []
    for method in METHODS:
        for arm in ARMS:
            base = _base_job(method)
            job_id = f"{method}__{arm}__tofu_forget10__seed0"
            jobs.append(
                {
                    **base,
                    "job_id": job_id,
                    "method": method,
                    "arm": arm,
                    "stage": STAGE,
                    "seed": 0,
                    "status": "pending",
                    "output_dir": f"jobs/{job_id}",
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "stage": STAGE,
        "seed": 0,
        "jobs": sorted(jobs, key=lambda job: job["job_id"]),
    }


def _replace(command, prefix, replacement):
    indices = [index for index, value in enumerate(command) if value.startswith(prefix)]
    if len(indices) != 1:
        raise ValueError(f"expected one command argument with prefix {prefix}")
    command[indices[0]] = replacement


def build_command(job, output_dir):
    base = _base_job(job["method"])
    if job["arm"] in {"native", "scope_bf16", "scope_fp32"}:
        command = baseline.build_command(base, output_dir)
    else:
        command = gu.build_command(base, output_dir)
    _replace(command, "task_name=", f"task_name={job['job_id']}")
    if job["arm"].startswith("scope_"):
        storage = "native" if job["arm"] == "scope_bf16" else "float32"
        command.append(
            "+trainer.method_args.scope_control="
            f"{{enabled:true,parameter_regex:[\"{PARAMETER_REGEX}\"],"
            f"selected_storage:{storage}}}"
        )
    if job["arm"] in {"gu_rank0", "gu_rank8"}:
        rank = 0 if job["arm"] == "gu_rank0" else 8
        _replace(
            command,
            "+trainer.method_args.gu.retain_history_rank=",
            f"+trainer.method_args.gu.retain_history_rank={rank}",
        )
    return command
```

- [ ] **Step 4: Run manifest/command tests and verify GREEN**

Run the Step 2 command again.

Expected: both tests pass.

- [ ] **Step 5: Add Hydra-composition and GPU0-environment tests**

Append tests proving all 20 commands compose and every child environment binds
physical GPU0 with the registered offline caches. Use Hydra exactly as existing
`tests/test_gu_full_matrix.py::compose_command` does:

```python
def test_every_stage_a_command_composes_and_binds_gpu0(tmp_path, monkeypatch):
    monkeypatch.setenv("GU_MATRIX_GPU", "0")
    runner = load_runner()
    from tests.test_gu_full_matrix import compose_command

    for job in runner.build_manifest()["jobs"]:
        command = runner.build_command(job, tmp_path / job["job_id"])
        config = compose_command(command)
        assert config.task_name == job["job_id"]
        assert config.forget_split == "forget10"
        environment = runner.build_environment(job)
        assert environment["CUDA_VISIBLE_DEVICES"] == "0"
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["HF_DATASETS_OFFLINE"] == "1"
```

- [ ] **Step 6: Implement `build_environment` and run composition tests**

```python
def build_environment(job):
    environment = os.environ.copy()
    environment.update(gu.environment_overrides(_base_job(job["method"])))
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    return environment


def command_identity(record):
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
```

Run:

```bash
GU_MATRIX_GPU=0 /root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_tofu_gu_ablation.py
```

Expected: all registry and composition tests pass.

- [ ] **Step 7: Commit the Stage-A registry**

```bash
git add scripts/run_tofu_gu_ablation.py tests/test_tofu_gu_ablation.py
git commit -m "feat: register TOFU GU causal ablations"
```

### Task 3: Add a durable checkpoint-free executor

**Files:**
- Modify: `scripts/run_tofu_gu_ablation.py`
- Modify: `tests/test_tofu_gu_ablation.py`

- [ ] **Step 1: Write failing executor evidence tests**

Use a subprocess stub that creates one exact TOFU summary/raw pair, a Hydra
directory, and GU diagnostics only for GU arms. Assert:

```python
def test_completed_ablation_requires_endpoint_resource_and_arm_evidence(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    job = next(
        item for item in runner.build_manifest()["jobs"] if item["arm"] == "gu_rank0"
    )
    install_successful_ablation_child(monkeypatch, runner, job, tmp_path)

    result = runner.run_job(job, tmp_path / job["output_dir"])

    assert result["status"] == "completed"
    assert result["optimizer_update_count"] > 0
    assert result["projection_count"] == result["optimizer_update_count"]
    assert result["peak_nvml_mib"] > 0
    assert result["forbidden_artifacts"] == []
```

Add parametrized corruptions for nonzero child exit, missing endpoint, missing
GU diagnostics, unexpected diagnostics on a non-GU arm, checkpoint/model files,
NVML monitor failure, and nonfinite JSON. Each must produce the exact
`invalid_scientific` or `failed_infrastructure` class used by the GU matrix
runner.

- [ ] **Step 2: Run executor tests and verify RED**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_tofu_gu_ablation.py -k 'evidence or corruption'
```

Expected: FAIL because `run_job` and the test stub do not exist.

- [ ] **Step 3: Implement `run_job` by reusing validation primitives**

Implement a direct executor with:

```python
def run_job(job, output_dir, launched=None, smoke=False):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("ablation output already exists")
    output.mkdir(parents=True)
    command = build_smoke_command(job, output) if smoke else build_command(job, output)
    command_record = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "job_id": job["job_id"],
        "method": job["method"],
        "arm": job["arm"],
        "seed": job["seed"],
        "argv": command,
        "environment": gu.environment_overrides(_base_job(job["method"])),
        "provenance": deepcopy(job["provenance"]),
    }
    gu._atomic_write_json(output / "command.json", command_record)
```

Complete the body with this fixed lifecycle:

1. open `run.log` and start `subprocess.Popen(command, cwd=gu.CODE_ROOT,
   env=build_environment(job), stdout=run_log, stderr=STDOUT)`;
2. invoke `launched(process.pid)` immediately when supplied;
3. once per second run `gu.NVML_QUERY_COMMAND`, count only the exact child PID,
   and retain maximum observed MiB;
4. wait for the child and classify nonzero exit as `invalid_scientific`, except
   launch/monitor/host-I/O failures which are `failed_infrastructure`;
5. for a non-smoke run require exactly one `TOFU_SUMMARY.json` and one
   `TOFU_EVAL.json`, parse both with `allow_nan` disabled, and require every
   aggregate to be finite;
6. parse the last `global_step` from the log as `optimizer_update_count`;
7. for `gu_rank0/gu_rank8`, require `gu_diagnostics.jsonl`, exact diagnostic
   fields, steps `1..optimizer_update_count`, finite values, and
   `max_violation_after <= projection_tolerance`; set projection/correction/
   zero-step fields from those records;
8. for native/scope arms reject any `gu_diagnostics.jsonl` and set GU-only
   result fields to `None`;
9. recursively reject files named `*.safetensors`, `pytorch_model*.bin`,
   `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`,
   `training_args.bin`, or `*.ckpt`;
10. smoke runs require exit zero, one optimizer update, no endpoints, and the
    same forbidden-artifact rules;
11. atomically write `ABLATION_JOB_RESULT.json` with the fixed result fields and
    return the identical mapping.

Add:

```python
def build_smoke_command(job, output_dir):
    command = [
        value
        for value in build_command(job, output_dir)
        if not value.startswith("trainer.args.max_steps=")
        and not value.startswith("+trainer.args.max_steps=")
        and value != "trainer.args.do_eval=true"
    ]
    command.extend(("+trainer.args.max_steps=1", "trainer.args.do_eval=false"))
    return command
```

Use the exact artifact names and JSON finiteness helpers from
`scripts/run_gu_full_matrix.py`; copy no benchmark-specific metric logic.
Result fields are fixed to:

```python
{
    "schema_version", "protocol", "job_id", "method", "arm", "seed",
    "status", "failure_kind", "returncode", "command_identity",
    "command_path", "run_log_path", "endpoint_summary_path",
    "endpoint_raw_path", "diagnostics_path", "optimizer_update_count",
    "projection_count", "correction_ratio", "max_violation_after",
    "zero_step_count", "wall_clock_seconds", "peak_nvml_mib",
    "gpu_memory_sample_count", "forbidden_artifacts", "issues",
    "provenance",
}
```

- [ ] **Step 4: Implement manifest state and sequential queue**

Add atomic `create_state`, `validate_state`, `run_queue`, and `status` functions.
The state contains all 20 jobs with `pending/running/terminal`, one attempt for
scientific failure, and at most two identical attempts for infrastructure
failure. Guard execution with `/tmp/tofu_gu_ablation_gpu0.lock` and refuse
launch when GPU0 uses 500 MiB or more.

Expose exact CLI commands:

```text
manifest --output-root PATH
preflight --manifest PATH
smoke --manifest PATH
queue --manifest PATH --state PATH
status --state PATH
```

`preflight` validates all four pinned base jobs with `gu.validate_sources` and
composes all 20 Hydra commands without creating job directories. `smoke` runs
the CEU instance of all five arms into `root/smoke/<job_id>` under the same GPU
lock and returns nonzero unless all five complete.

- [ ] **Step 5: Run executor/queue tests and verify GREEN**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_tofu_gu_ablation.py
```

Expected: all registry, evidence, failure, resume, retry, and queue tests pass.

- [ ] **Step 6: Commit the executor**

```bash
git add scripts/run_tofu_gu_ablation.py tests/test_tofu_gu_ablation.py
git commit -m "feat: execute checkpoint-free TOFU ablations"
```

### Task 4: Add Stage-A causal analyzer

**Files:**
- Create: `scripts/analyze_tofu_gu_ablation.py`
- Modify: `tests/test_tofu_gu_ablation.py`

- [ ] **Step 1: Write failing synthetic analyzer test**

Create synthetic completed endpoints with distinct values for each adjacent arm
and assert exact ordered effects:

```python
def test_analyzer_reports_adjacent_causal_effects(tmp_path):
    runner = load_runner()
    root = write_synthetic_completed_ablation(tmp_path, runner)
    analyzer = load_ablation_analyzer()

    report = analyzer.analyze(root)

    ceu = report["methods"]["CEU"]
    assert ceu["effects"]["scope"]["extraction_strength"] == 1.0
    assert ceu["effects"]["precision"]["extraction_strength"] == 1.0
    assert ceu["effects"]["current_projection"]["extraction_strength"] == 1.0
    assert ceu["effects"]["history"]["extraction_strength"] == 1.0
    assert set(report["metric_directions"]) == {
        "extraction_strength",
        "retain_extraction_strength",
        "privleak",
        "model_utility",
    }
```

- [ ] **Step 2: Run analyzer test and verify RED**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_tofu_gu_ablation.py -k analyzer
```

Expected: import failure because the analyzer does not exist.

- [ ] **Step 3: Implement validation, effects, and report rendering**

The analyzer must:

- validate the manifest/state and every completed result sidecar;
- require exact TOFU summary/raw agreement;
- extract all TOFU summary metrics plus resource/diagnostic fields;
- compute raw adjacent deltas `later - earlier` for every metric;
- also compute direction-normalized improvement where positive is favorable;
- render one five-arm table per method and one four-effect table per metric;
- list invalid/incomplete jobs without averaging or replacing them;
- atomically write `ABLATION_RAW.json`, `ABLATION_TABLES.md`, and
  `ABLATION_REPORT.md`.

- [ ] **Step 4: Run analyzer and full focused tests**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  pytest -q tests/test_scope_control.py tests/test_tofu_gu_ablation.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the analyzer**

```bash
git add scripts/analyze_tofu_gu_ablation.py tests/test_tofu_gu_ablation.py
git commit -m "feat: analyze TOFU GU causal effects"
```

### Task 5: Verify, launch, monitor, and hand off Stage A

**Files:**
- Runtime output: `/root/tofu_gu_ablation_20260809_gpu0/`

- [ ] **Step 1: Run complete CPU and static verification**

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/conda run --no-capture-output \
  -n unlearning pytest -q
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  ruff check src/trainer/unlearn/base.py scripts/run_tofu_gu_ablation.py \
  scripts/analyze_tofu_gu_ablation.py tests/test_scope_control.py \
  tests/test_tofu_gu_ablation.py
git diff --check
```

Expected: tests and Ruff pass.

- [ ] **Step 2: Build fresh manifest and run source/config preflight**

```bash
ROOT=/root/tofu_gu_ablation_20260809_gpu0
test ! -e "$ROOT"
GU_MATRIX_GPU=0 /root/miniconda3/bin/conda run --no-capture-output \
  -n unlearning python scripts/run_tofu_gu_ablation.py manifest \
  --output-root "$ROOT"
GU_MATRIX_GPU=0 /root/miniconda3/bin/conda run --no-capture-output \
  -n unlearning python scripts/run_tofu_gu_ablation.py preflight \
  --manifest "$ROOT/manifest.json" > "$ROOT/preflight.json"
```

Expected: 20 exact Hydra commands and all pinned source fingerprints pass.

- [ ] **Step 3: Run one-step smoke for all five arms**

```bash
GU_MATRIX_GPU=0 /root/miniconda3/bin/conda run --no-capture-output \
  -n unlearning python scripts/run_tofu_gu_ablation.py smoke \
  --manifest "$ROOT/manifest.json"
```

Expected: all five arms complete, bind GPU0, and produce no forbidden payload.

- [ ] **Step 4: Launch durable Stage-A queue in tmux**

```bash
tmux new-session -d -s tofu_gu_ablation_stage_a_gpu0_20260809 \
  "bash -lc 'cd /workspace/AR-projects/geo-unlearning; \
  export GU_MATRIX_GPU=0; \
  exec /root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  python scripts/run_tofu_gu_ablation.py queue \
  --manifest /root/tofu_gu_ablation_20260809_gpu0/manifest.json \
  --state /root/tofu_gu_ablation_20260809_gpu0/queue_state.json \
  > /root/tofu_gu_ablation_20260809_gpu0/queue.log 2>&1'"
```

- [ ] **Step 5: Verify physical GPU0 and no-checkpoint behavior**

Verify the live child PID maps to GPU0's UUID, its command environment records
`CUDA_VISIBLE_DEVICES=0`, and a recursive forbidden-artifact scan is empty.

- [ ] **Step 6: Monitor all 20 jobs to terminal status**

Use the runner's read-only `status` command and current job log. Do not weaken
configuration or replace scientific failures. Resume only from durable state.

- [ ] **Step 7: Generate and validate Stage-A report**

```bash
/root/miniconda3/bin/conda run --no-capture-output -n unlearning \
  python scripts/analyze_tofu_gu_ablation.py \
  --root /root/tofu_gu_ablation_20260809_gpu0 --require-complete
```

Expected: all four adjacent effects are reported for every valid method, with
no checkpoint artifacts. Use those effects to write the separate Stage-B/C
implementation plan required by the approved design.
