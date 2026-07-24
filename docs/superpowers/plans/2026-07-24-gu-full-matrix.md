# GU Full GPU Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, execute, resume, and analyze a checkpoint-free 11-method GU matrix on TOFU-8B, MUSE, and WMDP-Cyber using dev0.

**Architecture:** One direct runner owns the exact compatibility registry, Hydra commands, per-job live train/eval lifecycle, artifact validation, GPU sampling, and persistent sequential queue. One analyzer reconciles raw endpoint/resource evidence and generates benchmark tables. No generic scheduler framework or cloud/SSH layer is introduced.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, Transformers 4.45.1, PyTorch, subprocess, JSON, nvidia-smi, pytest, Markdown.

---

## File Map

- Create `scripts/run_gu_full_matrix.py`: registry, manifest, command builder, preflight, run-job, and queue.
- Create `scripts/analyze_gu_full_matrix.py`: evidence reconciliation, three-seed aggregation, tables, and report.
- Create `tests/test_gu_full_matrix.py`: all matrix, command, queue, validity, artifact, and analyzer tests.
- Create `configs/experiment/unlearn/tofu/gu_matrix_8b.yaml`: pinned TOFU-8B live-eval defaults.
- Create `configs/experiment/unlearn/muse/gu_matrix.yaml`: pinned MUSE live-eval defaults.
- Create `configs/experiment/unlearn/wmdp/gu_matrix_cyber.yaml`: pinned WMDP-Cyber live-eval defaults.
- Create `saves/exp/GU_FULL_MATRIX_20260724/manifest.json` only after preflight passes.

### Task 1: Exact Compatibility Registry and 60-Job Manifest

**Files:**
- Create: `scripts/run_gu_full_matrix.py`
- Create: `tests/test_gu_full_matrix.py`

- [ ] **Step 1: Write the failing registry tests**

Define tests for exact methods, settings, DPO N/A records, seed identity, and
60 compatible jobs:

```python
def test_seed_zero_manifest_has_exact_compatible_matrix():
    manifest = matrix.build_manifest(seed=0)
    compatible = [job for job in manifest["jobs"] if job["status"] == "pending"]
    na_rows = manifest["not_applicable"]

    assert len(compatible) == 60
    assert len({job["job_id"] for job in compatible}) == 60
    assert {job["method"] for job in compatible} == set(matrix.METHODS) - {"DPO"}
    assert {(row["method"], row["benchmark"]) for row in na_rows} == {
        ("DPO", "tofu_forget01"),
        ("DPO", "tofu_forget05"),
        ("DPO", "tofu_forget10"),
        ("DPO", "muse_news"),
        ("DPO", "muse_books"),
        ("DPO", "wmdp_cyber"),
    }
    assert all(row["reason"] == "missing_shipped_idk_artifact" for row in na_rows)
```

Require job IDs to encode method, benchmark, split, and seed without path
separators. Require exact stage, schema version, output path, trainer config,
experiment config, model identity, retain/holdout split, and GU regex.

- [ ] **Step 2: Verify RED**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
pytest tests/test_gu_full_matrix.py -k 'manifest or registry' -q
```

Expected: import failure for `scripts.run_gu_full_matrix`.

- [ ] **Step 3: Implement the explicit registry**

Use tuple/dict constants with no dynamic plugin registry:

```python
METHODS = (
    "GradAscent", "GradDiff", "NPO", "DPO", "SimNPO", "RMU",
    "UNDIAL", "CEU", "WGA", "SatImp", "PDU",
)
BENCHMARKS = (
    "tofu_forget01", "tofu_forget05", "tofu_forget10",
    "muse_news", "muse_books", "wmdp_cyber",
)
GENERAL_METHODS = (
    "GradAscent", "GradDiff", "NPO", "SimNPO", "UNDIAL",
    "CEU", "WGA", "SatImp", "PDU",
)
```

Store benchmark-specific model, experiment, split, evaluator, selected regex,
and provenance values in one `BENCHMARK_CONFIG` mapping. Add RMU to every
benchmark and record DPO as N/A on all six because the shipped
`./data/idk.jsonl` is absent. `build_manifest(seed=0)` returns sorted JSON-safe
records and N/A rows and rejects every nonzero or non-integer seed. Task 4 owns
all seed-1/2 expansion from valid seed-0 parents.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/test_gu_full_matrix.py -k 'manifest or registry' -q
```

Expected: exact count and registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_gu_full_matrix.py tests/test_gu_full_matrix.py
git commit -m "feat: define GU full matrix"
```

### Task 2: Pinned Live-Evaluation Commands

**Files:**
- Modify: `scripts/run_gu_full_matrix.py`
- Create: `configs/experiment/unlearn/tofu/gu_matrix_8b.yaml`
- Create: `configs/experiment/unlearn/muse/gu_matrix.yaml`
- Create: `configs/experiment/unlearn/wmdp/gu_matrix_cyber.yaml`
- Test: `tests/test_gu_full_matrix.py`

- [ ] **Step 1: Write failing command-contract tests**

For every compatible job, call `build_command(job, output_dir)` and assert:

```python
def test_commands_train_and_evaluate_live_without_persistence(tmp_path):
    for job in matrix.build_manifest(0)["jobs"]:
        command = matrix.build_command(job, tmp_path / job["job_id"])
        rendered = " ".join(command)
        assert command[1].endswith("src/train.py")
        assert "save_model_after_train=false" in command
        assert "trainer.args.save_strategy=no" in command
        assert "trainer.args.save_only_model=false" in command
        assert "trainer.args.do_eval=true" in command
        assert "trainer.args.eval_on_start=false" in command
        assert "trainer.args.eval_strategy=no" in command
        assert "trainer.method_args.gu.enabled=true" in rendered
        assert "trainer.method_args.gu.retain_history_rank=8" in rendered
        assert "trainer.method_args.gu.projection_eps=1e-6" in rendered
        assert "trainer.method_args.gu.retain_filter=first_order" in rendered
```

Assert the exact benchmark regex, model, split, seed, and objective trainer.
Assert WMDP commands and resolved configs contain no Bio token.

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_gu_full_matrix.py -k 'command or hydra or no_save' -q
```

Expected: command builder/config groups are absent.

- [ ] **Step 3: Implement benchmark config groups and commands**

Each config overrides the shipped benchmark/model/evaluator while leaving the
selected objective trainer's native hyperparameters intact. Commands add only:

- seed/data seed;
- split/model provenance;
- common GU mapping;
- live evaluation flags;
- no-save flags;
- output directory and task name;
- `report_to=none` and deterministic sampler settings already required by the
  shipped benchmark.

Use `+` only for keys absent after Hydra composition. Compose all 60 commands in
tests with Hydra and require no unresolved values.

RMU uses `trainer=RMU` and the completed branch's benchmark-specific module,
steering, and trainable-scope values, translated to the shipped RMU config keys.
The common GU regex remains the benchmark regex from the design.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/test_gu_full_matrix.py -k 'command or hydra or no_save' -q
python scripts/run_gu_full_matrix.py manifest --seed 0 --dry-run
```

Expected: all 60 commands resolve; dry-run creates no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_gu_full_matrix.py configs/experiment/unlearn/tofu/gu_matrix_8b.yaml configs/experiment/unlearn/muse/gu_matrix.yaml configs/experiment/unlearn/wmdp/gu_matrix_cyber.yaml tests/test_gu_full_matrix.py
git commit -m "feat: build checkpoint-free GU jobs"
```

### Task 3: Job Evidence, GPU Sampling, and Validity

**Files:**
- Modify: `scripts/run_gu_full_matrix.py`
- Test: `tests/test_gu_full_matrix.py`

- [ ] **Step 1: Write failing job-evidence tests**

Use a subprocess stub that writes a valid endpoint and diagnostics tree. Assert
`run_job` records command identity, wall time, peak NVML MiB, endpoint paths,
projection count, update count, zero steps, correction statistics, and artifact
audit. Add stubs for nonfinite diagnostics, missing endpoints, wrong projection
count, unchanged selected parameters, forbidden state files, OOM, and
infrastructure exit.

```python
def test_completed_job_requires_exact_gu_and_endpoint_evidence(tmp_path, monkeypatch):
    job = matrix.build_manifest(0)["jobs"][0]
    install_successful_subprocess_stub(monkeypatch, tmp_path, job)
    result = matrix.run_job(job, tmp_path / job["job_id"])
    assert result["status"] == "completed"
    assert result["projection_count"] == result["optimizer_update_count"]
    assert result["selected_parameter_changed"] is True
    assert result["forbidden_artifacts"] == []
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_gu_full_matrix.py -k 'run_job or evidence or validity or artifact' -q
```

Expected: job lifecycle is absent.

- [ ] **Step 3: Implement one direct subprocess lifecycle**

`run_job` creates a fresh job directory, writes `command.json`, starts
`src/train.py`, and polls:

```text
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Track maximum memory for the child process and total end-to-end time. Capture
stdout/stderr to `run.log`. After exit, locate exactly one benchmark endpoint
summary/evaluation pair, parse GU JSONL, parse Trainer log update count, and
scan forbidden artifacts.

Classify terminal state without changing the job command. Write `JOB_RESULT.json`
atomically. An infrastructure retry uses the same record and command identity.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/test_gu_full_matrix.py -k 'run_job or evidence or validity or artifact' -q
```

Expected: successful and failure classification tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_gu_full_matrix.py tests/test_gu_full_matrix.py
git commit -m "feat: validate GU matrix jobs"
```

### Task 4: Persistent Sequential Queue and Seed Expansion

**Files:**
- Modify: `scripts/run_gu_full_matrix.py`
- Test: `tests/test_gu_full_matrix.py`

- [ ] **Step 1: Write failing queue-state tests**

Test atomic state creation, one-running-job invariant, dev0 free-memory gate,
resume reconciliation, identical one-time infrastructure retry, scientific
no-retry, failed-job continuation, and Stage-2 expansion for every valid seed-0
job.

```python
def test_stage_two_expands_every_valid_seed_zero_job():
    state = synthetic_terminal_seed_zero_state(valid_count=17)
    expanded = matrix.expand_replication_jobs(state)
    assert len(expanded) == 34
    assert {job["seed"] for job in expanded} == {1, 2}
    assert {job["parent_seed_zero"] for job in expanded} == {
        job["job_id"] for job in state["jobs"] if job["status"] == "completed"
    }
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_gu_full_matrix.py -k 'queue or resume or replication' -q
```

Expected: queue functions are absent.

- [ ] **Step 3: Implement queue subcommands**

Provide:

```text
manifest --seed 0 --output-root PATH
preflight --manifest PATH
smoke --manifest PATH
queue --manifest PATH --state PATH
expand-seeds --state PATH
status --state PATH
run-job --manifest PATH --job-id ID
```

Use atomic temporary-file replacement for state writes. Before each launch,
require no running matrix child and dev0 used memory below 500 MiB. On restart,
preserve terminal states and reconcile a stale `running` record from its PID and
`JOB_RESULT.json`.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/test_gu_full_matrix.py -k 'queue or resume or replication' -q
```

Expected: queue persistence and expansion tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_gu_full_matrix.py tests/test_gu_full_matrix.py
git commit -m "feat: queue GU matrix on dev0"
```

### Task 5: Raw Reconciliation and Big Tables

**Files:**
- Create: `scripts/analyze_gu_full_matrix.py`
- Modify: `tests/test_gu_full_matrix.py`

- [ ] **Step 1: Write failing analyzer tests**

Construct exact synthetic TOFU, MUSE, and WMDP result trees. Test raw-summary
reconciliation, duplicate/missing seed rejection, N/A retention, sample standard
deviation, invalid-run display, WMDP Bio rejection, resource fields, and exact
Markdown headers.

```python
def test_analyzer_renders_all_benchmark_tables(tmp_path):
    state = write_complete_synthetic_matrix(tmp_path)
    report = analysis.analyze_matrix(state)
    markdown = analysis.render_tables(report)
    assert markdown.count("| Method | ES Re. ↑ | ES Un. ↓ | Priv. ↑ | MU ↑") == 3
    assert markdown.count("| Method | VerbMem ↓ | KnowMem ↓") == 2
    assert "| Method | WMDP-Cyber ↓ | MMLU ↑" in markdown
    assert "missing_shipped_idk_artifact" in markdown
    assert "Bio" not in report["wmdp"]["provenance_scope"]
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_gu_full_matrix.py -k 'analy or table or reconcile' -q
```

Expected: analyzer import failure.

- [ ] **Step 3: Implement strict analysis**

Read only jobs registered in the manifest/state. Validate each endpoint against
its raw evaluation file and sidecars. Aggregate a method only with exact seeds
0/1/2. List incomplete/invalid methods separately. Emit deterministic
`GU_FULL_RAW.json`, `GU_FULL_TABLES.md`, and `GU_FULL_REPORT.md`.

Use sample standard deviation (`ddof=1`). Label peak memory as NVML MiB because
the runner samples process GPU memory externally.

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/test_gu_full_matrix.py -k 'analy or table or reconcile' -q
```

Expected: all table and tamper tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_gu_full_matrix.py tests/test_gu_full_matrix.py
git commit -m "feat: analyze GU full matrix"
```

### Task 6: Preflight, Smokes, Queue Launch, and Monitoring

**Files:**
- Modify: `tests/test_gu_full_matrix.py`
- Output: `saves/exp/GU_FULL_MATRIX_20260724/`

- [ ] **Step 1: Run complete static verification**

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
CUDA_VISIBLE_DEVICES='' pytest -q
ruff check scripts/run_gu_full_matrix.py scripts/analyze_gu_full_matrix.py tests/test_gu_full_matrix.py
git diff --check
```

- [ ] **Step 2: Build manifest and run preflight**

```bash
python scripts/run_gu_full_matrix.py manifest --seed 0 --output-root saves/exp/GU_FULL_MATRIX_20260724
python scripts/run_gu_full_matrix.py preflight --manifest saves/exp/GU_FULL_MATRIX_20260724/manifest.json
```

Require 60 jobs, 6 DPO N/A rows, all Hydra commands resolved, cache/provenance
checks green, and no output job directory yet.

- [ ] **Step 3: Run representative smokes on dev0**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_gu_full_matrix.py smoke --manifest saves/exp/GU_FULL_MATRIX_20260724/manifest.json
```

Run one smoke per compatible method plus RMU benchmark-specific smokes. Preserve
every failure and block only its compatibility class.

- [ ] **Step 4: Launch persistent queue**

```bash
screen -dmS gu_full_matrix_20260724 bash -lc '
  source /root/miniconda3/etc/profile.d/conda.sh &&
  conda activate unlearning &&
  cd /workspace/re/GU/geo-unlearning/.worktrees/gu &&
  CUDA_VISIBLE_DEVICES=0 python scripts/run_gu_full_matrix.py queue \
    --manifest saves/exp/GU_FULL_MATRIX_20260724/manifest.json \
    --state saves/exp/GU_FULL_MATRIX_20260724/queue_state.json \
    2>&1 | tee saves/exp/GU_FULL_MATRIX_20260724/queue.log'
```

- [ ] **Step 5: Verify launch**

```bash
screen -ls | grep gu_full_matrix_20260724
cat saves/exp/GU_FULL_MATRIX_20260724/queue_state.json | jq '.jobs | group_by(.status) | map({(.[0].status): length}) | add'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Expected: exactly one matrix process on dev0 and one job in `running` state.

- [ ] **Step 6: Monitor to Stage-1 completion and expand seeds**

Do not fabricate a completed table before all 60 seed-0 jobs are terminal. After
Stage 1:

```bash
python scripts/run_gu_full_matrix.py expand-seeds --state saves/exp/GU_FULL_MATRIX_20260724/queue_state.json
python scripts/run_gu_full_matrix.py queue --manifest saves/exp/GU_FULL_MATRIX_20260724/manifest.json --state saves/exp/GU_FULL_MATRIX_20260724/queue_state.json
```

- [ ] **Step 7: Generate and audit final tables**

```bash
python scripts/analyze_gu_full_matrix.py --root saves/exp/GU_FULL_MATRIX_20260724 --require-complete
```

Compare every rendered scalar to its raw endpoint and resource record before
reporting results.
