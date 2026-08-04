# GPU 1 GU Full-Matrix Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing GU full-matrix runner select physical GPU 1 reproducibly, provision its pinned runtime sources, and complete the seed-0 plus seed-1/2 experiment matrix with validated reports.

**Architecture:** Keep the registered experiment commands and scientific validation unchanged. Add one environment-derived physical GPU selector shared by child environment binding, admission, evidence, and controller locking; then use the existing offline preflight and durable queue as the authority for source readiness and execution progress.

**Tech Stack:** Python 3.11, pytest, PyTorch 2.4.1, Transformers 4.45.1, Accelerate 0.34.2, Hugging Face Hub/Datasets, Hydra, NVML via `nvidia-smi`, tmux, JSON queue state.

---

### Task 1: Add a fail-closed physical GPU selector

**Files:**
- Modify: `tests/test_gu_full_matrix.py`
- Modify: `scripts/run_gu_full_matrix.py:523-533,585-590,731-751,2099-2154`

- [ ] **Step 1: Write failing selector and environment tests**

Add these tests beside the existing environment tests:

```python
def test_matrix_gpu_defaults_to_zero_and_reaches_child_environment(monkeypatch):
    monkeypatch.delenv("GU_MATRIX_GPU", raising=False)
    registry = load_registry()
    job = registry.build_manifest(seed=0)["jobs"][0]

    assert registry.matrix_gpu_index() == 0
    assert registry.environment_overrides(job)["CUDA_VISIBLE_DEVICES"] == "0"
    assert registry.controller_lock_path() == Path(
        "/tmp/geo_unlearning_gu_matrix_dev0.lock"
    )


def test_matrix_gpu_one_reaches_environment_and_uses_distinct_lock(
    monkeypatch,
):
    monkeypatch.setenv("GU_MATRIX_GPU", "1")
    registry = load_registry()
    job = registry.build_manifest(seed=0)["jobs"][0]

    assert registry.matrix_gpu_index() == 1
    assert registry.environment_overrides(job)["CUDA_VISIBLE_DEVICES"] == "1"
    assert registry.controller_lock_path() == Path(
        "/tmp/geo_unlearning_gu_matrix_dev1.lock"
    )


@pytest.mark.parametrize("value", ["", "-1", "+1", "1.0", "gpu1", " 1"])
def test_matrix_gpu_rejects_invalid_values_before_manifest(
    monkeypatch,
    value,
):
    monkeypatch.setenv("GU_MATRIX_GPU", value)
    registry = load_registry()

    with pytest.raises(ValueError, match="GU_MATRIX_GPU"):
        registry.build_manifest(seed=0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/workspace/envs/verl-vllm0181-fsdp/bin/python -m pytest -q \
  tests/test_gu_full_matrix.py \
  -k 'matrix_gpu_defaults or matrix_gpu_one or matrix_gpu_rejects'
```

Expected: failures because `matrix_gpu_index` and `controller_lock_path` do not exist and child CUDA remains `0`.

- [ ] **Step 3: Implement the minimal selector**

In `scripts/run_gu_full_matrix.py`, replace the fixed controller path with:

```python
MATRIX_GPU_ENV = "GU_MATRIX_GPU"


def matrix_gpu_index():
    """Return the configured physical GPU index, defaulting to device 0."""
    raw = os.environ.get(MATRIX_GPU_ENV, "0")
    if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
        raise ValueError(f"{MATRIX_GPU_ENV} must be a nonnegative integer")
    return int(raw)


def controller_lock_path():
    """Return the controller lock dedicated to the selected physical GPU."""
    return Path(f"/tmp/geo_unlearning_gu_matrix_dev{matrix_gpu_index()}.lock")
```

At the start of `build_manifest`, call `matrix_gpu_index()` before validating the seed. Change `environment_overrides` to:

```python
"CUDA_VISIBLE_DEVICES": str(matrix_gpu_index()),
```

Change `controller_lock` to open `controller_lock_path()` instead of the fixed constant.

- [ ] **Step 4: Run focused selector tests and verify GREEN**

Run the Step 2 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing GPU1 admission tests**

Add beside the existing admission tests:

```python
def test_admission_samples_selected_physical_gpu_one(tmp_path, monkeypatch):
    monkeypatch.setenv("GU_MATRIX_GPU", "1")
    registry = load_registry()
    monkeypatch.setattr(registry, "live_matrix_children", lambda *args: [])

    def gpu_query(argv, **kwargs):
        assert argv == registry.GPU_ADMISSION_COMMAND
        assert kwargs["timeout"] == 5.0
        return SimpleNamespace(
            returncode=0,
            stdout="0, 1200\n1, 499\n",
            stderr="",
        )

    monkeypatch.setattr(registry.subprocess, "run", gpu_query)

    assert registry.admit_launch({"jobs": []}, tmp_path) == 499


def test_admission_reports_selected_physical_gpu_one_busy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GU_MATRIX_GPU", "1")
    registry = load_registry()
    monkeypatch.setattr(registry, "live_matrix_children", lambda *args: [])
    monkeypatch.setattr(
        registry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="0, 0\n1, 500\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match=r"GPU 1 has 500 MiB"):
        registry.admit_launch({"jobs": []}, tmp_path)
```

- [ ] **Step 6: Run the admission tests and verify RED**

Run:

```bash
/workspace/envs/verl-vllm0181-fsdp/bin/python -m pytest -q \
  tests/test_gu_full_matrix.py -k 'admission and physical_gpu_one'
```

Expected: failure because admission still filters and reports GPU 0.

- [ ] **Step 7: Make admission use the selector**

Replace the device-specific body of `admit_launch` with the selected index:

```python
gpu_index = matrix_gpu_index()
selected_samples = []
for line in sample.stdout.splitlines():
    fields = [field.strip() for field in line.split(",", 1)]
    if len(fields) != 2 or fields[0] != str(gpu_index):
        continue
    try:
        selected_samples.append(int(re.sub(r"\s*MiB\s*$", "", fields[1])))
    except ValueError:
        continue
if len(selected_samples) != 1:
    raise RuntimeError(
        f"GPU admission did not return exactly one device {gpu_index} sample"
    )
if selected_samples[0] >= 500:
    raise RuntimeError(
        f"GPU {gpu_index} has {selected_samples[0]} MiB used; "
        "launch requires below 500 MiB"
    )
return selected_samples[0]
```

Update the function and queue docstrings to say “selected physical GPU.”

- [ ] **Step 8: Run focused and full runner tests**

Run:

```bash
/workspace/envs/verl-vllm0181-fsdp/bin/python -m pytest -q \
  tests/test_gu_full_matrix.py \
  -k 'matrix_gpu or admission or global_dev0_lock'
git diff --check
```

Expected: all focused tests pass and default GPU-0 contracts remain unchanged.

- [ ] **Step 9: Commit the GPU selector**

```bash
git add scripts/run_gu_full_matrix.py tests/test_gu_full_matrix.py
git commit -m "feat: select GU matrix physical GPU"
```

### Task 2: Provision and verify the pinned Python environment

**Files:**
- Read: `requirements.txt`
- Read: `setup.py`
- Runtime output: `/root/miniconda3/envs/unlearning/`

- [ ] **Step 1: Create Python 3.11 environment**

```bash
/root/miniconda3/bin/conda create -y -n unlearning python=3.11 pip
```

Expected: `/root/miniconda3/envs/unlearning/bin/python` exists.

- [ ] **Step 2: Install repository and development dependencies**

```bash
/root/miniconda3/bin/conda run -n unlearning \
  python -m pip install --upgrade pip setuptools wheel
/root/miniconda3/bin/conda run -n unlearning \
  python -m pip install -e '.[dev,lm-eval]'
```

Expected: pinned requirements and pytest install successfully.

- [ ] **Step 3: Install the pinned attention kernel**

```bash
/root/miniconda3/bin/conda run -n unlearning \
  python -m pip install --no-build-isolation flash-attn==2.6.3
```

Expected: `flash_attn` imports against Torch 2.4.1.

- [ ] **Step 4: Verify exact runtime versions and CUDA visibility**

```bash
GU_MATRIX_GPU=1 /root/miniconda3/bin/conda run -n unlearning python - <<'PY'
import bitsandbytes
import datasets
import flash_attn
import pytest
import torch
import transformers
import accelerate

assert torch.__version__.startswith("2.4.1")
assert transformers.__version__ == "4.45.1"
assert accelerate.__version__ == "0.34.2"
assert datasets.__version__ == "3.0.1"
assert bitsandbytes.__version__ == "0.44.1"
assert torch.cuda.is_available()
print(torch.__version__, transformers.__version__, accelerate.__version__)
PY
```

Expected: exact versions print and exit zero.

- [ ] **Step 5: Run implementation tests that do not require matrix sources**

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/conda run -n unlearning pytest -q \
  tests/test_minimal_gu.py tests/test_optimizer_geometry.py \
  tests/test_gradient_surgery.py tests/test_component_buffers.py
```

Expected: all selected non-GPU tests pass; GPU-marked tests skip.

### Task 3: Materialize every pinned offline source

**Files:**
- Read: `setup_data.py`
- Read: `scripts/run_gu_full_matrix.py:262-353,754-1135`
- Runtime output: `/dev/shm/gu-matrix-tofu/`
- Runtime output: `/dev/shm/gu-matrix-muse-news/`
- Runtime output: `/dev/shm/gu-matrix-muse-books/`
- Runtime output: `/dev/shm/gu-matrix-wmdp/`
- Runtime output: `saves/eval/`
- Runtime output: `data/wmdp/wmdp-corpora/`

- [ ] **Step 1: Create the registered shared-root alias**

```bash
mkdir -p /workspace/re/GU
ln -s /workspace/AR-projects/geo-unlearning /workspace/re/GU/geo-unlearning
test "$(readlink -f /workspace/re/GU/geo-unlearning)" = \
  /workspace/AR-projects/geo-unlearning
```

Expected: the absolute paths embedded in the registered WMDP and retain-log provenance resolve to the current checkout.

- [ ] **Step 2: Download exact evaluation logs and WMDP corpora**

```bash
/root/miniconda3/bin/conda run -n unlearning python setup_data.py --eval_logs --wmdp
```

Expected: TOFU/MUSE retain JSON files and both WMDP cyber corpora exist.

- [ ] **Step 3: Download all pinned model snapshots into their registered roots**

Run:

```bash
/root/miniconda3/bin/conda run -n unlearning python - <<'PY'
from huggingface_hub import snapshot_download

downloads = (
    ("/dev/shm/gu-matrix-tofu/hub", "open-unlearning/tofu_Llama-3.1-8B-Instruct_full", "1a5c5b1a557f8c99bdadecd5168ebd03f640b00e"),
    ("/dev/shm/gu-matrix-tofu/hub", "madhurjindal/autonlp-Gibberish-Detector-492513457", "76672dd7d3575f68ab980705bcec975cc62de71c"),
    ("/dev/shm/gu-matrix-muse-news/hub", "muse-bench/MUSE-News_target", "a2f39769e9a0b98ec1cdd12f65e9962502208935"),
    ("/dev/shm/gu-matrix-muse-news/hub", "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"),
    ("/dev/shm/gu-matrix-muse-news/hub", "muse-bench/MUSE-News_retrain", "324ef49ee0a038078aba7d8de831edf57235c9b3"),
    ("/dev/shm/gu-matrix-muse-news/hub", "madhurjindal/autonlp-Gibberish-Detector-492513457", "76672dd7d3575f68ab980705bcec975cc62de71c"),
    ("/dev/shm/gu-matrix-muse-books/hub", "muse-bench/MUSE-Books_target", "c8dd3fb23a726762ec66d277933c7cff6767f3c2"),
    ("/dev/shm/gu-matrix-muse-books/hub", "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"),
    ("/dev/shm/gu-matrix-muse-books/hub", "muse-bench/MUSE-Books_retrain", "1d67430e4e8bdf2a65823740e909792519175ac2"),
    ("/dev/shm/gu-matrix-muse-books/hub", "madhurjindal/autonlp-Gibberish-Detector-492513457", "76672dd7d3575f68ab980705bcec975cc62de71c"),
    ("/dev/shm/gu-matrix-wmdp/hub", "HuggingFaceH4/zephyr-7b-beta", "892b3d7a7b1cf10c7a701c60881cd93df615734c"),
)
for cache_dir, repo_id, revision in downloads:
    print(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir))
PY
```

Expected: every returned path ends in the registered 40-hex revision.

- [ ] **Step 4: Build the TOFU and MUSE dataset caches**

```bash
/root/miniconda3/bin/conda run -n unlearning python - <<'PY'
from datasets import load_dataset

tofu_configs = (
    "forget01", "forget01_perturbed", "forget05", "forget05_perturbed",
    "forget10", "forget10_perturbed", "retain90", "retain95", "retain99",
    "retain_perturbed", "real_authors_perturbed", "world_facts_perturbed",
    "holdout01", "holdout05", "holdout10",
)
for config in tofu_configs:
    load_dataset(
        "locuslab/TOFU",
        config,
        revision="324592d84ae4f482ac7249b9285c2ecdb53e3a68",
        cache_dir="/dev/shm/gu-matrix-tofu/datasets",
    )

for split, revision, cache_dir in (
    ("News", "506bd5b150b92814d45e4404a82f120ab2d748bf", "/dev/shm/gu-matrix-muse-news/datasets"),
    ("Books", "051ba90319e920d410d87cfdbd61f25843c1b892", "/dev/shm/gu-matrix-muse-books/datasets"),
):
    for config in ("raw", "knowmem", "verbmem", "privleak"):
        load_dataset(
            f"muse-bench/MUSE-{split}",
            config,
            revision=revision,
            cache_dir=cache_dir,
        )
PY
```

Expected: all four registered cache configs exist for each MUSE benchmark and all required TOFU configs exist.

- [ ] **Step 5: Build the WMDP, MMLU, and WikiText caches**

```bash
/root/miniconda3/bin/conda run -n unlearning python - <<'PY'
from datasets import get_dataset_config_names, load_dataset

root = "/dev/shm/gu-matrix-wmdp/datasets"
load_dataset(
    "cais/wmdp",
    "wmdp-cyber",
    revision="7125571f22f032c56415e7980f48d877dd830ff8",
    cache_dir=root,
)
mmlu_revision = "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60"
for config in get_dataset_config_names(
    "hails/mmlu_no_train",
    revision=mmlu_revision,
):
    load_dataset(
        "hails/mmlu_no_train",
        config,
        revision=mmlu_revision,
        cache_dir=root,
    )
load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    cache_dir=root,
)
PY
```

Expected: one WMDP builder, 57 MMLU builders, and one WikiText builder match the registered directory IDs.

- [ ] **Step 6: Run the complete CPU verification suite now that sources exist**

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/conda run -n unlearning pytest -q
```

Expected: all non-GPU tests pass; GPU-marked tests skip.

- [ ] **Step 7: Create the fresh manifest and run the authoritative preflight**

```bash
export GU_MATRIX_GPU=1
ROOT=saves/exp/GU_FULL_MATRIX_20260804_GPU1
/root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py manifest --seed 0 --output-root "$ROOT"
/root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py preflight --manifest "$ROOT/manifest.json" \
  > "$ROOT/preflight.json"
```

Expected: exit zero, 60 job fingerprints, no source mismatch, and no job output directory.

### Task 4: Run smokes and launch the seed-0 queue on GPU 1

**Files:**
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/smoke/`
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue_state.json`
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue.log`

- [ ] **Step 1: Recheck physical GPU 1 admission**

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Expected: physical GPU 1 is below 500 MiB before launch.

- [ ] **Step 2: Run the representative smoke matrix**

```bash
export GU_MATRIX_GPU=1
ROOT=saves/exp/GU_FULL_MATRIX_20260804_GPU1
/root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py smoke --manifest "$ROOT/manifest.json" \
  > "$ROOT/smoke_results.json"
```

Expected: every selected method/RMU benchmark smoke has status `completed` and physical GPU 1 is the only device with the child PID.

- [ ] **Step 3: Launch the durable seed-0 queue in tmux**

```bash
tmux new-session -d -s gu_full_matrix_gpu1_20260804 \
  "cd /workspace/AR-projects/geo-unlearning && \
   export GU_MATRIX_GPU=1 && \
   /root/miniconda3/bin/conda run -n unlearning \
   python scripts/run_gu_full_matrix.py queue \
   --manifest saves/exp/GU_FULL_MATRIX_20260804_GPU1/manifest.json \
   --state saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue_state.json \
   2>&1 | tee saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue.log"
```

Expected: a detached tmux session owns one sequential queue.

- [ ] **Step 4: Verify the live child is physically on GPU 1**

```bash
ROOT=saves/exp/GU_FULL_MATRIX_20260804_GPU1
GU_MATRIX_GPU=1 /root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py status --state "$ROOT/queue_state.json"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
```

Expected: one running matrix child; its UUID maps to physical GPU 1; its command evidence records `CUDA_VISIBLE_DEVICES=1`.

- [ ] **Step 5: Monitor Stage 1 to terminal completion**

Poll queue status and the latest job log without modifying state:

```bash
GU_MATRIX_GPU=1 /root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py status \
  --state saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue_state.json
tail -n 80 saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue.log
```

Expected: eventually all 60 seed-0 jobs are terminal. Scientific failures remain recorded; infrastructure failures follow the existing one-retry policy.

### Task 5: Expand replications and render the final report

**Files:**
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue_state.json`
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/GU_FULL_RAW.json`
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/GU_FULL_TABLES.md`
- Runtime output: `saves/exp/GU_FULL_MATRIX_20260804_GPU1/GU_FULL_REPORT.md`

- [ ] **Step 1: Expand seeds 1 and 2 atomically**

```bash
export GU_MATRIX_GPU=1
ROOT=saves/exp/GU_FULL_MATRIX_20260804_GPU1
/root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py expand-seeds --state "$ROOT/queue_state.json"
```

Expected: every completed valid seed-0 parent receives exactly seeds 1 and 2; invalid parents receive neither.

- [ ] **Step 2: Drain the Stage-2 queue in a fresh tmux session**

```bash
tmux new-session -d -s gu_full_matrix_gpu1_seeds_20260804 \
  "cd /workspace/AR-projects/geo-unlearning && \
   export GU_MATRIX_GPU=1 && \
   /root/miniconda3/bin/conda run -n unlearning \
   python scripts/run_gu_full_matrix.py queue \
   --manifest saves/exp/GU_FULL_MATRIX_20260804_GPU1/manifest.json \
   --state saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue_state.json \
   2>&1 | tee -a saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue.log"
```

Expected: all derived jobs reach terminal states on physical GPU 1.

- [ ] **Step 3: Run the complete analyzer gate**

```bash
/root/miniconda3/bin/conda run -n unlearning \
  python scripts/analyze_gu_full_matrix.py \
  --root saves/exp/GU_FULL_MATRIX_20260804_GPU1 \
  --require-complete
```

Expected: analyzer exits zero and regenerates raw JSON, tables, and report only from validated evidence.

- [ ] **Step 4: Perform final audit**

```bash
GU_MATRIX_GPU=1 /root/miniconda3/bin/conda run -n unlearning \
  python scripts/run_gu_full_matrix.py status \
  --state saves/exp/GU_FULL_MATRIX_20260804_GPU1/queue_state.json
git status --short
git diff --check
```

Expected: no pending/running jobs, no unvalidated checkpoint payloads, reports exist, and only intentional source changes are tracked.
