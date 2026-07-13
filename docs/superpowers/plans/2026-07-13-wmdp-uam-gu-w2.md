# WMDP-Cyber UAM and Residual-GU W2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the approved, checkpoint-free Zephyr-7B WMDP-Cyber five-arm comparison for Base, RMU, RMU+GU, UAM, and residual UAM+GU on GPU 0.

**Architecture:** Add one shared representation-objective module, two narrowly scoped WMDP trainers, and an explicit Euclidean-UAM/frozen-Adam-GU coordinate split. Reuse the existing component buffers, replay transaction, optimizer adapter, diagnostics, and Trainer step path; put matrix safety, evaluation compatibility, and artifact auditing in dedicated scripts.

**Tech Stack:** Python 3.11, PyTorch, Transformers 4.45.1, Accelerate, Hydra/OmegaConf, bitsandbytes PagedAdamW32, FlashAttention 2, lm-eval, pytest, Bash.

---

## File map

- `tests/conftest.py`: isolate Accelerate singleton state between CPU and GPU tests.
- `src/trainer/unlearn/wmdp_representation.py`: exact module capture, masks, representation losses, deterministic noise.
- `src/trainer/unlearn/wmdp_selection.py`: exact layer-5–7 parameter selection and optimizer validation.
- `src/trainer/unlearn/uam_geometry.py`: geometry-neutral UAM reflection primitives.
- `src/trainer/unlearn/uam.py`: selectable UAM reflection geometry while preserving token-UAM defaults.
- `src/trainer/unlearn/wmdp_uam.py`: WMDP representation UAM and residual GU trainer.
- `src/trainer/unlearn/wmdp_rmu.py`: matched WMDP RMU and RMU+GU trainer.
- `src/trainer/__init__.py`: trainer registration.
- `configs/trainer/WMDPUAMUnlearn.yaml`, `configs/trainer/WMDPRMUUnlearn.yaml`: fail-closed paper-derived configurations.
- `tests/test_wmdp_representation.py`, `tests/test_wmdp_selection.py`, `tests/test_wmdp_uam.py`, `tests/test_wmdp_rmu.py`: CPU correctness tests.
- `tests/test_wmdp_gpu_integration.py`: real two-update BF16/Flash/PagedAdamW32 tests.
- `scripts/wmdp_uam_w2_arm.sh`, `scripts/wmdp_uam_w2_matrix.sh`: single-arm and sequential five-arm runners.
- `scripts/analyze_wmdp_uam_w2.py`: compatibility gate, mechanism audit, and table generation.
- `tests/test_wmdp_w2_scripts.py`, `tests/test_analyze_wmdp_uam_w2.py`: fail-closed script tests.
- `WMDP_UAM_GU_W2_REPORT.md`: measured final report generated from raw artifacts.

### Task 0: Restore a trustworthy test baseline

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_uam_gpu_integration.py`
- Test: `tests/test_uam_trainer.py`

- [ ] **Step 1: Preserve the failing order-dependent reproduction**

Run:

```bash
conda activate unlearning
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/test_uam_gpu_integration.py::test_torch_adamw_bf16_runs_two_public_uam_updates \
  tests/test_uam_trainer.py::test_uam_gas_equivalence_matches_effective_batch_update
```

Expected: the GPU tests pass and both CPU GAS cases fail after inheriting Accelerate BF16 singleton state.

- [ ] **Step 2: Add an autouse state-isolation fixture**

```python
import pytest
from accelerate.state import AcceleratorState, GradientState


@pytest.fixture(autouse=True)
def isolate_accelerate_singletons():
    yield
    GradientState._reset_state()
    AcceleratorState._reset_state(reset_partial_state=True)
```

The reset occurs after each test, after trainer fixtures release their models and optimizers.

- [ ] **Step 3: Verify the order-dependent reproduction and full baseline**

Run the Step 1 command, then:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q
```

Expected: `545 passed`; warnings may remain, but there are no failures.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: isolate accelerate state across trainers"
```

### Task 1: Add transaction-safe representation objectives

**Files:**
- Create: `src/trainer/unlearn/wmdp_representation.py`
- Create: `tests/test_wmdp_representation.py`

- [ ] **Step 1: Write failing tests for exact lookup and hook cleanup**

Tests create a tiny model with `model.layers.7`, call the wished-for API, and assert one exact match. A forward exception must leave `module._forward_hooks` empty.

```python
module = find_exact_module(model, "model.layers.7")
activation, outputs = forward_with_activation(model, batch, module)
assert activation.shape[:2] == batch["input_ids"].shape
assert module._forward_hooks == {}
```

Run:

```bash
pytest -q tests/test_wmdp_representation.py
```

Expected: collection fails because `wmdp_representation` does not exist.

- [ ] **Step 2: Implement exact lookup and `try/finally` capture**

```python
def find_exact_module(model, name):
    matches = [module for module_name, module in model.named_modules() if module_name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one module named {name!r}, found {len(matches)}.")
    return matches[0]


def forward_with_activation(model, inputs, module, *, no_grad=False):
    captured = []
    handle = module.register_forward_hook(
        lambda _module, _args, output: captured.append(output[0] if isinstance(output, tuple) else output)
    )
    try:
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one captured activation, found {len(captured)}.")
    return captured[0], outputs
```

- [ ] **Step 3: Write failing tests for masked MSE and deterministic noise**

Assert padding values do not change the loss, an empty mask raises, equal seeds give equal noise, and noise has finite non-zero norm.

- [ ] **Step 4: Implement the loss helpers**

```python
def supervised_token_mask(inputs):
    mask = inputs["attention_mask"].bool() & inputs["labels"].ne(-100)
    if not mask.any().item():
        raise RuntimeError("WMDP representation objective has an empty token mask.")
    return mask


def masked_representation_mse(actual, target, mask):
    if actual.shape != target.shape or actual.shape[:2] != mask.shape:
        raise ValueError("WMDP activation, target, and mask shapes are incompatible.")
    per_token = (actual.float() - target.float()).square().mean(dim=-1)
    loss = per_token.masked_select(mask).mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError("WMDP representation loss is non-finite.")
    return loss
```

Use a run-local `torch.Generator` and `torch.randn(shape, generator=generator, device="cpu")`, then transfer the noise to the activation device. Retain/reference calls never consume this generator.

- [ ] **Step 5: Verify and commit**

```bash
pytest -q tests/test_wmdp_representation.py
git add src/trainer/unlearn/wmdp_representation.py tests/test_wmdp_representation.py
git commit -m "feat: add WMDP representation objectives"
```

### Task 2: Enforce the exact WMDP trainable-tensor contract

**Files:**
- Create: `src/trainer/unlearn/wmdp_selection.py`
- Create: `tests/test_wmdp_selection.py`

- [ ] **Step 1: Write failing selection tests**

Use a fake 9-layer model and require exactly:

```python
EXPECTED_WMDP_PARAMETER_NAMES = (
    "model.layers.5.mlp.down_proj.weight",
    "model.layers.6.mlp.down_proj.weight",
    "model.layers.7.mlp.down_proj.weight",
)
```

Test missing, duplicate, additional trainable, and optimizer-group mismatch cases.

- [ ] **Step 2: Implement selection and optimizer validation**

`select_wmdp_parameters(model)` returns the three ordered `(name, parameter)` pairs and rejects any missing name. `validate_wmdp_optimizer(optimizer, selected)` compares parameter identities from all optimizer groups to the selected identity set and reports missing/extra names.

- [ ] **Step 3: Verify and commit**

```bash
pytest -q tests/test_wmdp_selection.py
git add src/trainer/unlearn/wmdp_selection.py tests/test_wmdp_selection.py
git commit -m "feat: enforce WMDP parameter selection"
```

### Task 3: Split Euclidean UAM reflection from optimizer-metric residual GU

**Files:**
- Modify: `src/trainer/unlearn/uam_geometry.py`
- Modify: `src/trainer/unlearn/uam.py`
- Modify: `configs/trainer/UAMUnlearn.yaml`
- Modify: `tests/test_uam_geometry.py`
- Modify: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write failing hand-computed coordinate tests**

For `gf=(1, 2)`, `gtilde=(3, 4)`, and `gamma=2`, assert:

```python
coefficient = 2 * torch.dot(gf, gtilde) / torch.dot(gf, gf)
expected = gtilde - coefficient * gf
```

Changing a fake Adam square-root denominator must not change this Euclidean candidate. The existing token-UAM default must remain optimizer-coordinate behavior.

- [ ] **Step 2: Add explicit `reflection_geometry` configuration**

Accepted values are `optimizer` and `euclidean`. `UAMUnlearn` defaults to `optimizer`; WMDP will pass `euclidean`. Reject unknown values during runtime validation.

- [ ] **Step 3: Refactor finalization into raw candidate then residual projection**

The Euclidean path computes `g_uam_raw` from mean raw `forget` and `perturbed_retain` buffers. For `uam_gu`, compute `residual_raw = g_uam_raw - g_retain_raw`, transform residual and retain through `_to_adam_coordinates`, globally project there, apply the sign/orthogonality gate, and map only the kept correction back with `_from_adam_coordinates`.

The final raw gradient is exactly:

```python
final_raw = retain_raw + residual_lambda * projected_residual_raw
```

or `retain_raw` when the gate fails.

- [ ] **Step 4: Verify token-UAM regressions and commit**

```bash
pytest -q tests/test_uam_geometry.py tests/test_uam_trainer.py tests/test_uam_replay.py
git add src/trainer/unlearn/uam_geometry.py src/trainer/unlearn/uam.py \
  configs/trainer/UAMUnlearn.yaml tests/test_uam_geometry.py tests/test_uam_trainer.py
git commit -m "feat: separate UAM and GU geometries"
```

### Task 4: Implement `WMDPUAMUnlearn`

**Files:**
- Create: `src/trainer/unlearn/wmdp_uam.py`
- Create: `tests/test_wmdp_uam.py`

- [ ] **Step 1: Write failing objective tests**

Build a tiny layer-7 model and frozen copy. Verify retain loss is zero at initialization, seeded forget noise produces a finite non-zero forget gradient on the selected tensors, and a second trainer with the same seed reproduces it.

- [ ] **Step 2: Implement the WMDP objective subclass**

`WMDPUAMUnlearn(UAMUnlearn)` prepares a frozen reference model, captures layer-7 activations, computes masked retain MSE, and computes forget MSE from `h_7(w) + N(0, 0.01^2)` to `h_7(w0)`. It overrides `_selected_named_parameters` with `select_wmdp_parameters` and validates the optimizer groups before the first step.

- [ ] **Step 3: Write failing paper-normalization and restoration tests**

Assert `delta = rho * gf / (global_raw_sq + eps)`, requested norm diagnostics are correct, and parameters are bitwise restored after both successful replay and an injected replay exception.

- [ ] **Step 4: Implement the WMDP runtime contract**

Require `reflection_geometry="euclidean"`, `rho=5e-5`, `reflection_gamma=2`, BF16/FP32, one process/GPU, non-reentrant checkpointing, beta1 zero, weight decay zero, and a PagedAdamW32 adapter. Error messages must name the unsupported setting.

- [ ] **Step 5: Write and pass residual-GU mechanism tests**

Test one global coefficient across all three tensors, optimizer-coordinate orthogonality at most `1e-6`, failed gate fallback to ordinary retain gradient, finite diagnostics, and one finalizer per effective batch.

- [ ] **Step 6: Verify and commit**

```bash
pytest -q tests/test_wmdp_uam.py
git add src/trainer/unlearn/wmdp_uam.py tests/test_wmdp_uam.py
git commit -m "feat: add WMDP representation UAM trainer"
```

### Task 5: Implement matched `WMDPRMUUnlearn`

**Files:**
- Create: `src/trainer/unlearn/wmdp_rmu.py`
- Create: `tests/test_wmdp_rmu.py`

- [ ] **Step 1: Write failing objective and seeded-control tests**

Assert the layer-7 target is a single seeded unit direction multiplied by 20 and reused across batches. Retain loss uses the same frozen-reference masked MSE as UAM.

- [ ] **Step 2: Implement explicit component losses**

`WMDPRMUUnlearn(GeometricUnlearn)` returns forget representation MSE, retain representation MSE, and forget outputs from `compute_component_losses`. Set `gamma=1`, `alpha=100`; select and validate only the three WMDP tensors.

- [ ] **Step 3: Test both gradient paths**

With `gu_enabled=false`, compare the actual pre-optimizer gradient and one-step parameter delta to `gf + 100 * gr`. With `gu_enabled=true`, compare to a hand-computed global frozen-Adam projection plus `100 * gr` and require relative orthogonality at most `1e-6`.

- [ ] **Step 4: Verify and commit**

```bash
pytest -q tests/test_wmdp_rmu.py
git add src/trainer/unlearn/wmdp_rmu.py tests/test_wmdp_rmu.py
git commit -m "feat: add matched WMDP RMU trainer"
```

### Task 6: Register trainers and add fail-closed configurations

**Files:**
- Modify: `src/trainer/__init__.py`
- Create: `configs/trainer/WMDPUAMUnlearn.yaml`
- Create: `configs/trainer/WMDPRMUUnlearn.yaml`
- Modify: `tests/test_wmdp_uam.py`
- Modify: `tests/test_wmdp_rmu.py`

- [ ] **Step 1: Write failing Hydra and registry tests**

Assert both class names are present in `TRAINER_REGISTRY`. Resolve both YAML files and assert physical batch 1, GAS 4, max steps 80, LR `5e-5`, BF16, FlashAttention supplied by the Zephyr model config, `paged_adamw_32bit`, beta1 zero, weight decay zero, save strategy `no`, and the exact selected parameter names.

- [ ] **Step 2: Add imports, registration, and YAML**

The UAM YAML sets `mode` through an arm override, `reflection_geometry: euclidean`, `rho: 5.0e-5`, `reflection_gamma: 2.0`, `noise_std: 0.01`, `module_name: model.layers.7`, and `seed: 42`. The RMU YAML sets `steering_coeff: 20`, `alpha: 100`, and the same systems settings.

- [ ] **Step 3: Verify and commit**

```bash
pytest -q tests/test_wmdp_uam.py tests/test_wmdp_rmu.py
git add src/trainer/__init__.py configs/trainer/WMDPUAMUnlearn.yaml \
  configs/trainer/WMDPRMUUnlearn.yaml tests/test_wmdp_uam.py tests/test_wmdp_rmu.py
git commit -m "feat: configure WMDP W2 trainers"
```

### Task 7: Prove accumulation equivalence and failure atomicity

**Files:**
- Modify: `tests/test_wmdp_uam.py`
- Modify: `tests/test_wmdp_rmu.py`

- [ ] **Step 1: Add effective-batch equivalence tests**

Use the same four examples with batch 4/GAS 1 and batch 1/GAS 4. Compare component means, geometry coefficients, final raw gradients, one-step parameter deltas, and counters for RMU, RMU+GU, UAM, and UAM+GU.

- [ ] **Step 2: Add short-window and exception tests**

Use two examples with configured GAS 4 and assert only two microsteps are normalized. Inject exceptions during activation capture, perturbation, replay, projection, diagnostic write, and optimizer step; assert no partial counters, component buffers, replay batches, hooks, perturbations, or checkpoint payloads remain.

- [ ] **Step 3: Verify and commit**

```bash
pytest -q tests/test_wmdp_uam.py tests/test_wmdp_rmu.py
git add tests/test_wmdp_uam.py tests/test_wmdp_rmu.py
git commit -m "test: verify WMDP accumulation and transactions"
```

### Task 8: Add real GPU integration tests

**Files:**
- Create: `tests/test_wmdp_gpu_integration.py`

- [ ] **Step 1: Write two-update BF16 tests**

Use a small Llama model with at least eight blocks, FlashAttention 2, BF16, PagedAdamW32, batch 1/GAS 2, and the exact layer 5–7 tensors. Parametrize all four trainable arms.

- [ ] **Step 2: Assert systems and mechanism invariants**

Check two optimizer updates, two finalizer calls, BF16 gradients, finite losses, non-zero selected deltas, unmodified non-selected parameters, exact perturbation restoration, PagedAdamW32 state2, and no `checkpoint-*` directory or model/optimizer/scheduler/RNG payload.

- [ ] **Step 3: Run on GPU 0 and commit**

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q -m gpu tests/test_wmdp_gpu_integration.py
git add tests/test_wmdp_gpu_integration.py
git commit -m "test: validate WMDP W2 GPU path"
```

### Task 9: Add fail-closed arm and matrix runners

**Files:**
- Create: `scripts/wmdp_uam_w2_arm.sh`
- Create: `scripts/wmdp_uam_w2_matrix.sh`
- Create: `tests/test_wmdp_w2_scripts.py`

- [ ] **Step 1: Write script contract tests**

Test arm allowlisting (`base`, `rmu`, `rmu_gu`, `uam`, `uam_gu`), GPU must equal `0`, safe run IDs, refusal to overwrite output, Base-first scheduling, stop-on-failure, and rejection of symlinks and checkpoint payload names.

- [ ] **Step 2: Implement the arm runner**

Activate `unlearning` from `conda info --base`, export `CUDA_VISIBLE_DEVICES=0`, stage each arm under `/tmp/wmdp_uam_w2/<run>/<arm>`, pass all approved Hydra overrides, wrap training/evaluation with `/usr/bin/time -v`, record CUDA/NVML/RSS peaks, audit the staging tree, and atomically promote only JSON, logs, and Hydra config. Never persist model weights.

- [ ] **Step 3: Implement the sequential matrix runner**

Run Base, call the analyzer with `--check-base-gate`, then sequentially run RMU, RMU+GU, UAM, and UAM+GU. A non-zero arm or failed gate prevents subsequent scheduling. Write an append-only TSV manifest with PID, arm, UTC times, exit code, and exact command.

- [ ] **Step 4: Verify and commit**

```bash
pytest -q tests/test_wmdp_w2_scripts.py
bash -n scripts/wmdp_uam_w2_arm.sh scripts/wmdp_uam_w2_matrix.sh
git add scripts/wmdp_uam_w2_arm.sh scripts/wmdp_uam_w2_matrix.sh \
  tests/test_wmdp_w2_scripts.py
git commit -m "feat: add WMDP W2 matrix runner"
```

### Task 10: Add raw-artifact analysis and report generation

**Files:**
- Create: `scripts/analyze_wmdp_uam_w2.py`
- Create: `tests/test_analyze_wmdp_uam_w2.py`

- [ ] **Step 1: Write fixture-based failing tests**

Fixtures cover the Base boundary values, missing task metrics, 79/80 updates, excessive orthogonality, non-finite diagnostics, checkpoint artifacts, inconsistent revisions, and one complete negative-result matrix.

- [ ] **Step 2: Implement strict parsing and gate logic**

Read WMDP-Cyber `acc` and MMLU group `acc` plus stderr from raw lm-eval summaries. The gate is exactly:

```python
abs(mmlu - 0.5810) <= 0.02 and abs(cyber - 0.4400) <= 0.03
```

Require 80 updates/finalizers for trained arms, kept residuals at most `1e-6`, identical provenance fields, and checkpoint/symlink audit success.

- [ ] **Step 3: Generate JSON, Markdown, and table rows**

Output raw values, endpoint deltas, standard errors, training wall-clock, end-to-end wall-clock, allocator/NVML/RSS peaks, and runtime relative to RMU. Narrative selection is deterministic: complementarity, trade-off, dominated, or incomplete; it never claims significance from seed 42.

- [ ] **Step 4: Verify and commit**

```bash
pytest -q tests/test_analyze_wmdp_uam_w2.py
git add scripts/analyze_wmdp_uam_w2.py tests/test_analyze_wmdp_uam_w2.py
git commit -m "feat: analyze WMDP UAM GU matrix"
```

### Task 11: Run full preflight and untouched Base gate

**Files:**
- Runtime artifacts only under `saves/exp/WMDP_UAM_W2/<run-id>/`

- [ ] **Step 1: Run all static and unit checks**

```bash
conda activate unlearning
CUDA_VISIBLE_DEVICES=0 pytest -q
ruff check src tests scripts
git diff --check
```

Expected: all tests and formatting checks pass.

- [ ] **Step 2: Run the Base arm only**

```bash
bash scripts/wmdp_uam_w2_arm.sh base 0 <run-id>
python scripts/analyze_wmdp_uam_w2.py \
  --matrix-root saves/exp/WMDP_UAM_W2/<run-id> \
  --check-base-gate
```

Expected: measured Base is within both fixed tolerances. If not, stop without any trained arm.

- [ ] **Step 3: Commit only code/config changes**

Runtime artifacts remain uncommitted; record the Base gate outcome in the final report after the matrix completes.

### Task 12: Run the sequential five-arm matrix

**Files:**
- Runtime artifacts only under `saves/exp/WMDP_UAM_W2/<run-id>/`

- [ ] **Step 1: Launch on GPU 0**

```bash
bash scripts/wmdp_uam_w2_matrix.sh <run-id>
```

Expected: Base passes first; each trainable arm completes exactly 80 updates before the next begins.

- [ ] **Step 2: Audit raw artifacts**

```bash
python scripts/analyze_wmdp_uam_w2.py \
  --matrix-root saves/exp/WMDP_UAM_W2/<run-id> \
  --output-json saves/exp/WMDP_UAM_W2/<run-id>/W2_SUMMARY.json \
  --output-markdown saves/exp/WMDP_UAM_W2/<run-id>/W2_TABLE.md
```

Expected: every table cell is recomputed from raw JSON; no weights, checkpoint directories, symlinks, or trainer state are present.

### Task 13: Final verification and evidence-bounded report

**Files:**
- Create: `WMDP_UAM_GU_W2_REPORT.md`

- [ ] **Step 1: Write the measured report**

Include the Cyber-only paper-derived-port wording, pinned UAM/RMU commits, Base compatibility delta, exact five-arm table, mechanism diagnostics, runtime/memory, and the analyzer-selected conclusion. Explicitly label GU as current-effective-batch global rank one and approximate frozen-Adam aware.

- [ ] **Step 2: Run final verification**

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q
ruff check src tests scripts
git diff --check
python scripts/analyze_wmdp_uam_w2.py \
  --matrix-root saves/exp/WMDP_UAM_W2/<run-id> \
  --audit-only
```

- [ ] **Step 3: Commit the report and analyzer-produced summary only**

```bash
git add WMDP_UAM_GU_W2_REPORT.md
git commit -m "docs: report WMDP UAM GU W2 comparison"
```

Do not commit model weights, optimizer state, scheduler state, RNG state, Trainer checkpoints, raw generated caches, or symlinks.
