# WMDP-Cyber Strong UAM Pair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one checkpoint-free UAM/UAM+GU pair at `lr=1.25e-4`, 160 updates, and a constant schedule, then audit and compare it with the completed primary W2 matrix.

**Architecture:** Parameterize the existing single-arm W2 runner with fail-closed environment overrides while preserving its current defaults. Add a two-arm orchestration script and a focused analyzer that reuses the existing artifact parsers and geometry validator, verifies the exact strong configuration, and classifies the predeclared `[0.27, 0.31]` UAM calibration target.

**Tech Stack:** Bash, Python 3.11, PyYAML, Hydra, Transformers 4.45.1, Accelerate, PagedAdamW32, BF16, FlashAttention 2, pytest, ruff.

---

## File map

- Modify `scripts/wmdp_uam_w2_arm.sh`: add validated learning-rate, update-count, scheduler, and actual-delta-step overrides with unchanged W2 defaults.
- Modify `scripts/analyze_wmdp_uam_w2.py`: allow the geometry validator to receive an expected update count while retaining 80 as its default.
- Create `scripts/wmdp_uam_strong_pair.sh`: sequentially run only UAM and UAM+GU with the approved strong configuration.
- Create `scripts/analyze_wmdp_uam_strong_pair.py`: audit two-arm artifacts, exact Hydra configuration, geometry, target band, endpoint deltas, and resources.
- Modify `tests/test_wmdp_w2_scripts.py`: protect existing defaults and test strong-run override contracts.
- Create `tests/test_analyze_wmdp_uam_strong_pair.py`: fixture-based analyzer tests for the target band and fail-closed behavior.

### Task 1: Parameterize the existing arm runner without changing W2 defaults

**Files:**
- Modify: `scripts/wmdp_uam_w2_arm.sh`
- Modify: `tests/test_wmdp_w2_scripts.py`

- [ ] **Step 1: Write failing script-contract tests**

Add assertions that the arm runner defines these defaults and uses them in the Hydra command:

```python
for token in (
    "learning_rate=${WMDP_W2_LEARNING_RATE:-5e-5}",
    "max_steps=${WMDP_W2_MAX_STEPS:-80}",
    "lr_scheduler_type=${WMDP_W2_LR_SCHEDULER_TYPE:-linear}",
    '"trainer.args.learning_rate=${learning_rate}"',
    '"trainer.args.max_steps=${max_steps}"',
    '"trainer.args.lr_scheduler_type=${lr_scheduler_type}"',
    "'trainer.method_args.geometric_config.actual_delta_steps=[1,'",
):
    assert token in text
```

Also assert the script rejects a non-positive/non-finite learning rate, a
non-positive/non-integer update count, and a scheduler outside
`linear|constant` before `accelerate` is invoked.

- [ ] **Step 2: Run the focused tests and observe failure**

Run:

```bash
conda activate unlearning
pytest -q tests/test_wmdp_w2_scripts.py
```

Expected: the new override-contract assertions fail because the variables do
not exist.

- [ ] **Step 3: Implement fail-closed overrides**

Before conda activation, resolve:

```bash
learning_rate=${WMDP_W2_LEARNING_RATE:-5e-5}
max_steps=${WMDP_W2_MAX_STEPS:-80}
lr_scheduler_type=${WMDP_W2_LR_SCHEDULER_TYPE:-linear}
```

Require `max_steps` to match `^[1-9][0-9]*$`, require the scheduler to be
`linear` or `constant`, and use a small Python numeric check that rejects bool,
NaN, infinity, zero, and negative learning rates. Replace fixed Hydra values
with the variables and set:

```bash
"trainer.method_args.geometric_config.actual_delta_steps=[1,${max_steps}]"
```

- [ ] **Step 4: Run focused tests and Bash syntax checks**

Run:

```bash
pytest -q tests/test_wmdp_w2_scripts.py
bash -n scripts/wmdp_uam_w2_arm.sh scripts/wmdp_uam_w2_matrix.sh
```

Expected: all tests pass and both scripts parse successfully.

- [ ] **Step 5: Commit the runner parameterization**

```bash
git add scripts/wmdp_uam_w2_arm.sh tests/test_wmdp_w2_scripts.py
git commit -m "feat: parameterize WMDP UAM arm strength"
```

### Task 2: Generalize geometry validation for 160 updates

**Files:**
- Modify: `scripts/analyze_wmdp_uam_w2.py`
- Modify: `tests/test_analyze_wmdp_uam_w2.py`

- [ ] **Step 1: Write a failing 160-update validator test**

Create 160 UAM geometry records and call:

```python
geometry = analyzer.validate_geometry(
    "uam",
    arm_dir,
    expected_updates=160,
)
assert len(geometry) == 160
```

Retain the existing test proving that `analyze_matrix()` still rejects 79 of
the default 80 updates.

- [ ] **Step 2: Run the targeted test and observe the signature failure**

```bash
pytest -q tests/test_analyze_wmdp_uam_w2.py
```

Expected: `validate_geometry()` rejects the unexpected keyword argument.

- [ ] **Step 3: Add an explicit expected-update parameter**

Change the signature to:

```python
def validate_geometry(arm, arm_dir, expected_updates=EXPECTED_UPDATES):
```

Use `expected_updates` for record count, contiguous steps, and final counters.
Keep `analyze_matrix()` unchanged so the primary W2 audit remains fixed at 80.

- [ ] **Step 4: Run analyzer tests**

```bash
pytest -q tests/test_analyze_wmdp_uam_w2.py
```

Expected: all primary and 160-update validator tests pass.

- [ ] **Step 5: Commit the reusable validator**

```bash
git add scripts/analyze_wmdp_uam_w2.py tests/test_analyze_wmdp_uam_w2.py
git commit -m "feat: validate configurable WMDP update counts"
```

### Task 3: Add the strong-pair analyzer

**Files:**
- Create: `scripts/analyze_wmdp_uam_strong_pair.py`
- Create: `tests/test_analyze_wmdp_uam_strong_pair.py`

- [ ] **Step 1: Write complete fixture-based failing tests**

Create `uam` and `uam_gu` fixture arms with 160 geometry records, matching
provenance, resources, lm-eval summaries, and Hydra configs containing:

```yaml
trainer:
  args:
    learning_rate: 0.000125
    max_steps: 160
    lr_scheduler_type: constant
  method_args:
    uam_config:
      rho: 0.00005
      mode: uam
```

Test inside/below/above target classification, complementarity/trade-off
classification, exact deltas against the primary W2 summary, and rejection of
159 updates, config drift, provenance drift, unsafe orthogonality, and
checkpoint payloads.

- [ ] **Step 2: Run the new tests and observe import failure**

```bash
pytest -q tests/test_analyze_wmdp_uam_strong_pair.py
```

Expected: the analyzer module does not exist.

- [ ] **Step 3: Implement strict two-arm analysis**

Import artifact helpers from `analyze_wmdp_uam_w2`. Define exact constants:

```python
ARMS = ("uam", "uam_gu")
EXPECTED_UPDATES = 160
EXPECTED_LR = 1.25e-4
EXPECTED_SCHEDULER = "constant"
TARGET_LOW = 0.27
TARGET_HIGH = 0.31
```

For each arm, load metrics/resources/provenance, parse `.hydra/config.yaml`,
assert the constants and arm mode, and call `validate_geometry(..., 160)`.
Load the immutable primary `W2_SUMMARY.json`, compute paired deltas and
resource ratios, and emit `STRONG_SUMMARY.json` plus `STRONG_TABLE.md`.

- [ ] **Step 4: Run analyzer tests and lint**

```bash
pytest -q tests/test_analyze_wmdp_uam_strong_pair.py
ruff check scripts/analyze_wmdp_uam_strong_pair.py \
  tests/test_analyze_wmdp_uam_strong_pair.py
```

Expected: all tests and lint checks pass.

- [ ] **Step 5: Commit the analyzer**

```bash
git add scripts/analyze_wmdp_uam_strong_pair.py \
  tests/test_analyze_wmdp_uam_strong_pair.py
git commit -m "feat: analyze strong WMDP UAM pair"
```

### Task 4: Add the sequential two-arm runner

**Files:**
- Create: `scripts/wmdp_uam_strong_pair.sh`
- Modify: `tests/test_wmdp_w2_scripts.py`

- [ ] **Step 1: Write failing orchestration tests**

Assert the script:

- exports `CUDA_VISIBLE_DEVICES=0`;
- exports `WMDP_W2_LEARNING_RATE=1.25e-4`,
  `WMDP_W2_MAX_STEPS=160`, and
  `WMDP_W2_LR_SCHEDULER_TYPE=constant`;
- sets the persistent root to `saves/exp/WMDP_UAM_W2_STRONG`;
- runs only `uam` then `uam_gu`;
- stops after any non-zero arm;
- invokes the strong analyzer with the immutable primary summary;
- refuses unsafe or existing run IDs.

- [ ] **Step 2: Run script tests and observe the missing-file failure**

```bash
pytest -q tests/test_wmdp_w2_scripts.py
```

Expected: `wmdp_uam_strong_pair.sh` is absent.

- [ ] **Step 3: Implement the strong-pair runner**

Follow the existing matrix manifest and stop-on-failure pattern. Use:

```bash
export WMDP_W2_PERSIST_ROOT=${WMDP_UAM_STRONG_ROOT:-/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2_STRONG}
export WMDP_W2_LEARNING_RATE=1.25e-4
export WMDP_W2_MAX_STEPS=160
export WMDP_W2_LR_SCHEDULER_TYPE=constant
```

After both arms complete, run the analyzer with the primary summary at
`/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2/w2-20260713-2/W2_SUMMARY.json`.

- [ ] **Step 4: Run tests and syntax checks**

```bash
pytest -q tests/test_wmdp_w2_scripts.py
bash -n scripts/wmdp_uam_w2_arm.sh scripts/wmdp_uam_w2_matrix.sh \
  scripts/wmdp_uam_strong_pair.sh
```

Expected: all tests and shell syntax checks pass.

- [ ] **Step 5: Commit the orchestration script**

```bash
git add scripts/wmdp_uam_strong_pair.sh tests/test_wmdp_w2_scripts.py
git commit -m "feat: run strong WMDP UAM pair"
```

### Task 5: Preflight and launch on GPU 0

**Files:**
- Runtime only: `/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2_STRONG/<run-id>`

- [ ] **Step 1: Run complete verification**

```bash
conda activate unlearning
CUDA_VISIBLE_DEVICES=0 pytest -q
ruff check src tests scripts
git diff --check
```

Expected: zero test failures, zero lint errors, and zero whitespace errors.

- [ ] **Step 2: Verify GPU 0 is free**

```bash
nvidia-smi --id=0 --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

Expected: GPU 0 memory use is below 500 MiB before launch.

- [ ] **Step 3: Launch the immutable pair**

Generate a UTC run ID and run:

```bash
bash scripts/wmdp_uam_strong_pair.sh <run-id>
```

Keep the launcher in a managed background process, record its PID/session,
and verify that GPU 0 becomes allocated and the UAM log starts advancing.

- [ ] **Step 4: Audit completion and collect results**

After both arms exit, run:

```bash
python scripts/analyze_wmdp_uam_strong_pair.py \
  --matrix-root /workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2_STRONG/<run-id> \
  --primary-summary /workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2/w2-20260713-2/W2_SUMMARY.json \
  --audit-only
```

Read `STRONG_TABLE.md` and `STRONG_SUMMARY.json`, report the target-band
classification and direct UAM+GU minus UAM deltas, and do not rerun based on
whether the target was hit.
