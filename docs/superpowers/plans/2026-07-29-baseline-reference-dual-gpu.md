# Dual-GPU Baseline Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a checkpoint-free, method-native, seed-0 baseline for the 60 compatible method/benchmark cells on two independent GPUs and render direct baseline-versus-GU improvement tables.

**Architecture:** `scripts/eval_all.sh` becomes the user-facing two-worker launcher. A focused companion, `scripts/run_baseline_reference.py`, imports the pinned GU registry for exact job provenance and commands, removes only GU arguments, owns per-worker atomic state/evidence, and compares validated baseline endpoints with GU seed0 endpoints. GPU0 and GPU1 each run one single-process job at a time; no DDP or checkpoint reload is used.

**Tech Stack:** Bash, Python 3.11, Hydra/OmegaConf, Transformers 4.45.1, PyTorch, subprocess, `fcntl`, `nvidia-smi`, JSON, pytest.

---

## File Map

- Create `scripts/run_baseline_reference.py`: baseline manifest, command builder, run lifecycle, worker state, smoke, comparison, and CLI.
- Rewrite `scripts/eval_all.sh`: thin dual-worker launcher and failure propagation.
- Create `tests/test_baseline_reference.py`: command, evidence, worker, shell, and comparison contracts.
- Create `/root/baseline_reference_20260729/` only after all tests and smokes pass.

### Task 1: Baseline Registry and Exact No-GU Commands

**Files:**
- Create: `scripts/run_baseline_reference.py`
- Create: `tests/test_baseline_reference.py`

- [ ] **Step 0: Add deterministic module-loading helpers to the test file**

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_baseline():
    return load_script(ROOT / "scripts/run_baseline_reference.py", "baseline_reference")


def load_gu_registry():
    return load_script(ROOT / "scripts/run_gu_full_matrix.py", "gu_registry")
```

- [ ] **Step 1: Write failing registry and command tests**

```python
def test_seed_zero_baseline_manifest_matches_gu_registry():
    baseline = load_baseline()
    gu = load_gu_registry()
    manifest = baseline.build_manifest()
    assert len(manifest["jobs"]) == 60
    assert manifest["not_applicable"] == gu.build_manifest(0)["not_applicable"]
    assert [(job["method"], job["benchmark"], job["seed"])
            for job in manifest["jobs"]] == [
        (job["method"], job["benchmark"], 0)
        for job in gu.build_manifest(0)["jobs"]
    ]


def test_baseline_commands_are_live_no_save_and_have_no_gu(tmp_path):
    baseline = load_baseline()
    for job in baseline.build_manifest()["jobs"]:
        argv = baseline.build_command(job, tmp_path / job["job_id"])
        rendered = " ".join(argv)
        assert "trainer.method_args.gu" not in rendered
        assert "save_model_after_train=false" in argv
        assert "trainer.args.save_strategy=no" in argv
        assert "trainer.args.save_only_model=false" in argv
        assert "trainer.args.do_eval=true" in argv
        assert "trainer.args.eval_strategy=no" in argv
```

- [ ] **Step 2: Verify RED**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'manifest or command'
```

Expected: import failure for missing `scripts/run_baseline_reference.py`.

- [ ] **Step 3: Implement the direct registry and command transformation**

```python
#!/usr/bin/env python3
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
    argv = gu.build_command(job, Path(output_dir).resolve())
    return [arg for arg in argv if not arg.startswith(GU_ARGUMENT_PREFIX)]
```

Expose `manifest --output-root PATH` and `preflight --manifest PATH`. Preflight
calls the existing GU registry's source fingerprint validation for all 60 jobs
and writes no job output.

Compose every command in the test with Hydra and assert:

```python
assert "gu" not in config.trainer.method_args
assert config.trainer.handler == job["method"]
assert config.trainer.args.seed == config.trainer.args.data_seed == 0
```

- [ ] **Step 4: Verify GREEN**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'manifest or command'
```

Expected: 60 unique commands compose and pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_baseline_reference.py tests/test_baseline_reference.py
git commit -m "feat: define matched baseline reference jobs"
```

### Task 2: Checkpoint-Free Baseline Job Evidence

**Files:**
- Modify: `scripts/run_baseline_reference.py`
- Modify: `tests/test_baseline_reference.py`

- [ ] **Step 1: Write failing evidence tests**

```python
def test_completed_baseline_requires_endpoint_update_memory_and_no_artifacts(
    tmp_path, monkeypatch
):
    baseline = load_baseline()
    job = baseline.build_manifest()["jobs"][0]
    install_successful_child(monkeypatch, baseline, job, tmp_path / job["job_id"])
    result = baseline.run_job(job, tmp_path / job["job_id"], physical_gpu=0)
    assert result["status"] == "completed"
    assert result["optimizer_update_count"] > 0
    assert result["peak_nvml_mib"] > 0
    assert result["endpoint_summary_path"].endswith("_SUMMARY.json")
    assert result["endpoint_raw_path"].endswith("_EVAL.json")
    assert result["forbidden_artifacts"] == []
    assert not (tmp_path / job["job_id"] / "gu_diagnostics.jsonl").exists()
```

```python
@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("missing_endpoint", "endpoint"),
        ("duplicate_endpoint", "endpoint"),
        ("nonfinite_endpoint", "nonfinite"),
        ("zero_updates", "optimizer update"),
        ("no_gpu_sample", "GPU"),
        ("checkpoint_config", "forbidden"),
        ("model_weight", "forbidden"),
        ("optimizer_state", "forbidden"),
        ("nonzero_exit", "subprocess"),
    ],
)
def test_baseline_rejects_invalid_evidence(
    tmp_path, monkeypatch, corruption, expected
):
    baseline = load_baseline()
    job = baseline.build_manifest()["jobs"][0]
    install_corrupt_child(
        monkeypatch, baseline, job, tmp_path / job["job_id"], corruption
    )
    result = baseline.run_job(job, tmp_path / job["job_id"], physical_gpu=0)
    assert result["status"] != "completed"
    assert expected.lower() in json.dumps(result).lower()
```

- [ ] **Step 2: Verify RED**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'run_job or evidence or artifact'
```

Expected: `run_job` is missing.

- [ ] **Step 3: Implement one direct child lifecycle**

Use the existing registry seams rather than duplicating provenance:

```python
def child_environment(job, physical_gpu):
    environment = gu.build_environment(job)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    return environment
```

`run_job` must:

1. validate the exact registered job before creating output;
2. atomically write `command.json`;
3. launch `src/train.py` in the current worktree;
4. poll direct-child NVML memory with a five-second timeout;
5. capture `run.log` and wall-clock seconds;
6. parse Trainer loss records as optimizer updates;
7. require exactly one registered endpoint pair;
8. allow only `.hydra`, the method log, command/result/run logs, and endpoint pair;
9. atomically write `BASELINE_JOB_RESULT.json`.

Use these terminal values:

```python
TERMINAL = {"completed", "invalid_scientific", "failed_infrastructure"}
```

OOM, NaN/Inf, nonzero objective exit, missing endpoint, and forbidden persistence
are scientific invalid. Launch, cache, host-I/O, and zero valid NVML observations
are infrastructure failures. A recovered transient NVML warning does not negate
a completed endpoint.

- [ ] **Step 4: Verify GREEN**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'run_job or evidence or artifact'
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_baseline_reference.py tests/test_baseline_reference.py
git commit -m "feat: validate checkpoint-free baseline jobs"
```

### Task 3: Two Independent GPU Workers and Resume

**Files:**
- Modify: `scripts/run_baseline_reference.py`
- Modify: `tests/test_baseline_reference.py`

- [ ] **Step 1: Write failing worker tests**

```python
def test_two_worker_partitions_are_disjoint_and_complete():
    jobs = load_baseline().build_manifest()["jobs"]
    left = load_baseline().partition(jobs, worker=0, workers=2)
    right = load_baseline().partition(jobs, worker=1, workers=2)
    assert len(left) == len(right) == 30
    assert {job["job_id"] for job in left}.isdisjoint(
        job["job_id"] for job in right
    )
    assert {job["job_id"] for job in left + right} == {
        job["job_id"] for job in jobs
    }


def test_worker_retries_only_one_identical_infrastructure_failure(tmp_path):
    state = run_synthetic_worker(tmp_path, first="failed_infrastructure", second="completed")
    job = state["jobs"][0]
    assert job["attempt_count"] == 2
    assert job["attempt_history"][0]["command_identity"] == (
        job["attempt_history"][1]["command_identity"]
    )
```

```python
@pytest.mark.parametrize(
    "scenario",
    ["terminal_skip", "stale_running", "scientific_no_retry", "gpu_lock"],
)
def test_worker_state_machine_scenarios(tmp_path, monkeypatch, scenario):
    baseline = load_baseline()
    state, launches = run_worker_scenario(tmp_path, monkeypatch, scenario)
    if scenario == "terminal_skip":
        assert launches == 0
    elif scenario == "stale_running":
        assert state["jobs"][0]["status"] in baseline.TERMINAL
    elif scenario == "scientific_no_retry":
        assert state["jobs"][0]["attempt_count"] == 1
    else:
        assert state["blocked_reason"] == "gpu_lock"


def test_worker_binds_the_requested_physical_gpu(tmp_path):
    baseline = load_baseline()
    job = baseline.build_manifest()["jobs"][0]
    assert baseline.child_environment(job, 1)["CUDA_VISIBLE_DEVICES"] == "1"
```

- [ ] **Step 2: Verify RED**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'partition or worker or resume or retry'
```

- [ ] **Step 3: Implement worker-owned atomic state**

```python
def partition(jobs, worker, workers=2):
    if worker not in range(workers):
        raise ValueError("worker must be 0 or 1")
    return [job for index, job in enumerate(jobs) if index % workers == worker]
```

Each worker writes only `worker{worker}_state.json`, holds
`/tmp/baseline_reference_gpu{gpu}.lock`, and admits launch only when that
physical GPU uses less than 500 MiB. Preserve terminal evidence on restart.
Retry one infrastructure attempt with the identical command; never retry a
scientific failure.

Expose CLI commands:

```text
manifest --output-root PATH
worker --manifest PATH --worker {0,1} --gpu {0,1}
status --root PATH
```

- [ ] **Step 4: Verify GREEN**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'partition or worker or resume or retry'
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_baseline_reference.py tests/test_baseline_reference.py
git commit -m "feat: queue baseline jobs across two GPUs"
```

### Task 4: `eval_all.sh` Dual-Worker Entrypoint

**Files:**
- Modify: `scripts/eval_all.sh`
- Modify: `tests/test_baseline_reference.py`

- [ ] **Step 1: Write failing shell contract test**

```python
def test_eval_all_is_checkpoint_free_dual_worker_launcher():
    text = (ROOT / "scripts/eval_all.sh").read_text()
    assert "run_baseline_reference.py manifest" in text
    assert "--worker 0 --gpu 0" in text
    assert "--worker 1 --gpu 1" in text
    assert "src/eval.py" not in text
    assert "accelerate launch" not in text
    assert "saves/unlearn" not in text
```

- [ ] **Step 2: Verify RED**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k eval_all
```

- [ ] **Step 3: Replace the script with a thin launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${BASELINE_ROOT:-/root/baseline_reference_20260729}"
CODE_DIR="/workspace/re/GU/geo-unlearning/.worktrees/gu"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
cd "$CODE_DIR"

python scripts/run_baseline_reference.py manifest --output-root "$ROOT_DIR"

python scripts/run_baseline_reference.py worker \
  --manifest "$ROOT_DIR/manifest.json" --worker 0 --gpu 0 \
  >"$ROOT_DIR/worker0.log" 2>&1 &
worker0=$!
python scripts/run_baseline_reference.py worker \
  --manifest "$ROOT_DIR/manifest.json" --worker 1 --gpu 1 \
  >"$ROOT_DIR/worker1.log" 2>&1 &
worker1=$!

trap 'kill "$worker0" "$worker1" 2>/dev/null || true' INT TERM
status=0
wait "$worker0" || status=1
wait "$worker1" || status=1
exit "$status"
```

- [ ] **Step 4: Verify GREEN**

```bash
bash -n scripts/eval_all.sh
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k eval_all
```

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_all.sh tests/test_baseline_reference.py
git commit -m "feat: launch baseline reference on two GPUs"
```

### Task 5: Baseline-versus-GU Uplift Tables

**Files:**
- Modify: `scripts/run_baseline_reference.py`
- Modify: `tests/test_baseline_reference.py`

- [ ] **Step 1: Write failing comparison tests**

```python
def test_improvement_sign_is_positive_when_gu_is_better():
    baseline = load_baseline()
    assert baseline.improvement("model_utility", baseline=0.60, gu=0.65) == 0.05
    assert baseline.improvement("extraction_strength", baseline=0.70, gu=0.40) == 0.30


def test_compare_renders_all_six_seed_zero_tables(tmp_path):
    baseline_root, gu_root = write_synthetic_paired_roots(tmp_path)
    report = load_baseline().compare(baseline_root, gu_root)
    markdown = load_baseline().render_comparison(report)
    assert markdown.count("| Method | Baseline | GU | Δ improvement |") > 0
    assert set(report["benchmarks"]) == {
        "tofu_forget01", "tofu_forget05", "tofu_forget10",
        "muse_news", "muse_books", "wmdp_cyber",
    }
```

```python
@pytest.mark.parametrize(
    "corruption",
    [
        "mismatched_provenance",
        "missing_pair",
        "nonfinite_metric",
        "duplicate_identity",
        "wmdp_bio_task",
    ],
)
def test_compare_rejects_unpaired_or_tampered_results(tmp_path, corruption):
    baseline = load_baseline()
    baseline_root, gu_root = write_synthetic_paired_roots(tmp_path)
    corrupt_comparison_input(baseline_root, gu_root, corruption)
    with pytest.raises(ValueError):
        baseline.compare(baseline_root, gu_root)
```

- [ ] **Step 2: Verify RED**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'compare or improvement or table'
```

- [ ] **Step 3: Implement strict paired comparison**

Add CLI:

```text
compare --baseline-root PATH --gu-root PATH
```

Use fixed directions:

```python
HIGHER_IS_BETTER = {
    "retain_extraction_strength", "privleak", "model_utility",
    "retain_knowmem_ROUGE", "mmlu/acc",
}


def improvement(field, baseline, gu):
    return gu - baseline if field in HIGHER_IS_BETTER else baseline - gu
```

Require every baseline seed0 pair and its GU seed0 counterpart to pass their
native result/endpoint validators before rendering. Write atomically:

```text
BASELINE_GU_RAW.json
BASELINE_GU_TABLES.md
BASELINE_GU_REPORT.md
```

Include wall-clock overhead and peak-memory overhead alongside endpoint uplift.
Label all output `seed0 reference; no variance estimate`.

- [ ] **Step 4: Verify GREEN**

```bash
pytest -p no:cacheprovider -q tests/test_baseline_reference.py -k 'compare or improvement or table'
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_baseline_reference.py tests/test_baseline_reference.py
git commit -m "feat: compare baseline and GU seed-zero results"
```

### Task 6: Verification, Two-GPU Smoke, and Launch

**Files:**
- Output: `/root/baseline_reference_20260729/`

- [ ] **Step 1: Run static and CPU verification**

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
CUDA_VISIBLE_DEVICES='' pytest -p no:cacheprovider -q tests/test_baseline_reference.py
ruff check scripts/run_baseline_reference.py tests/test_baseline_reference.py
bash -n scripts/eval_all.sh
git diff --check
```

Generate and preflight the final manifest:

```bash
python scripts/run_baseline_reference.py manifest \
  --output-root /root/baseline_reference_20260729
python scripts/run_baseline_reference.py preflight \
  --manifest /root/baseline_reference_20260729/manifest.json
```

- [ ] **Step 2: Run one-step family smokes on both GPUs**

Expose:

```text
smoke --manifest PATH --gpu GPU --benchmarks tofu,muse,wmdp
```

The smoke command adds only:

```text
trainer.args.max_steps=1
trainer.args.do_eval=false
```

It verifies one optimizer update, no GU mapping, no persistence artifact, and
positive GPU memory evidence. Run disjoint smokes concurrently on GPU0/GPU1 and
require every smoke to pass.

- [ ] **Step 3: Verify both GPUs are free**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Expected: GPU0 and GPU1 each use less than 500 MiB.

- [ ] **Step 4: Expose the local run through the workspace**

```bash
test ! -e saves/exp/BASELINE_REFERENCE_20260729
ln -s /root/baseline_reference_20260729 \
  saves/exp/BASELINE_REFERENCE_20260729
```

- [ ] **Step 5: Launch the seed-zero baseline reference**

```bash
tmux new-session -d -s baseline_reference_20260729 \
  bash -lc 'cd /workspace/re/GU/geo-unlearning/.worktrees/gu && \
  exec env BASELINE_ROOT=/root/baseline_reference_20260729 \
  scripts/eval_all.sh'
```

- [ ] **Step 6: Verify launch**

```bash
tmux list-sessions -F '#{session_name} #{pane_pid}' | grep baseline_reference_20260729
python scripts/run_baseline_reference.py status \
  --root /root/baseline_reference_20260729
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Expected: two workers and at most one child on each physical GPU, with disjoint
running job identities.

- [ ] **Step 7: Monitor, compare, and hand off**

When all 60 baseline jobs are terminal and valid:

```bash
python scripts/run_baseline_reference.py compare \
  --baseline-root /root/baseline_reference_20260729 \
  --gu-root /root/gu_full_matrix_20260726
```

Audit every displayed scalar against both endpoint files before reporting
uplift. Do not claim variance or statistical significance from seed0.
