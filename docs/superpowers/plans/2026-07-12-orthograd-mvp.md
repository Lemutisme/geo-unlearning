# Full-Parameter OrthoGrad MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exact full-parameter per-sample Euclidean OrthoGrad for TOFU and produce a matched Control/GU/OrthoGrad compute, GPU-memory, timing, and endpoint comparison.

**Architecture:** Add a separate `OrthogradUnlearn` trainer with a global FP32 per-sample retain basis and its own transactional training step. Reuse current SimNPO/NLL losses, optimizer validation, component probes, and Trainer clipping/step order. Add opt-in resource profiling shared by Control, GU, and OrthoGrad, plus a strict checkpoint-free three-arm runner and deterministic analyzer.

**Tech Stack:** Python 3.11, PyTorch 2.4, Transformers 4.45.1, Accelerate, bitsandbytes 0.44.1, Hydra/OmegaConf, psutil, NVIDIA NVML, pytest, Ruff, Bash.

---

## File Map

- Create `src/trainer/unlearn/orthograd_geometry.py`: global vector dots, two-pass modified Gram–Schmidt, basis projection, rank and FLOP diagnostics.
- Create `src/trainer/unlearn/orthograd.py`: `OrthogradUnlearn`, input slicing, per-sample retain gradients, transactional writeback, runtime validation.
- Create `src/trainer/resource_profiler.py`: matched per-step CUDA/CPU timing, allocator peaks, phase aggregation, atomic JSON output.
- Modify `src/trainer/unlearn/geometric.py`: opt-in GU phase timing and shared resource callback registration.
- Modify `src/trainer/__init__.py`: register `OrthogradUnlearn`.
- Create `configs/trainer/OrthogradUnlearn.yaml`: supported MVP contract and fixed comparison weights.
- Create `tests/test_orthograd_geometry.py`: global geometry, rank deficiency, blockwise counterexample, FLOP and storage accounting.
- Create `tests/test_orthograd_trainer.py`: registration, validation, per-sample reconstruction, lifecycle, exceptions, actual updates.
- Create `tests/test_resource_profiler.py`: synchronized step timing, peaks, atomic profile output, phase aggregation.
- Create `tests/test_orthograd_gpu_integration.py`: BF16 eager/Flash and real PagedAdamW32 public updates.
- Create `scripts/profile_process_tree.py`: child process, CPU RSS, NVML memory, wall time sampler.
- Create `scripts/orthograd_mvp_arm.sh`: one strict no-checkpoint arm.
- Create `scripts/orthograd_mvp_matrix.sh`: sequential Control/GU/OrthoGrad matrix on GPU 0.
- Create `scripts/analyze_orthograd_mvp.py`: validate artifacts/config parity and render raw/overhead tables.
- Create `tests/test_orthograd_scripts.py`: process profiler, launch contract, analyzer fixtures and failure cases.
- Create `ORTHOGRAD_MVP_REPORT.md`: generated real-run report.

### Task 1: Add global OrthoGrad geometry primitives

**Files:**
- Create: `src/trainer/unlearn/orthograd_geometry.py`
- Create: `tests/test_orthograd_geometry.py`

- [ ] **Step 1: Write failing global basis and projection tests**

Create tests using two parameter blocks so a blockwise implementation cannot
pass accidentally:

```python
import pytest
import torch

from trainer.unlearn.orthograd_geometry import GlobalRetainBasis


def test_projection_is_globally_orthogonal_to_each_retain_direction():
    basis = GlobalRetainBasis(rank_tolerance=1e-8)
    retain_a = {"left": torch.tensor([1.0, 0.0]), "right": torch.tensor([1.0])}
    retain_b = {"left": torch.tensor([0.0, 1.0]), "right": torch.tensor([1.0])}
    assert basis.add(retain_a)
    assert basis.add(retain_b)

    candidate = {"left": torch.tensor([3.0, -2.0]), "right": torch.tensor([4.0])}
    projected, removed, stats = basis.project(candidate)

    for retain in (retain_a, retain_b):
        assert GlobalRetainBasis.global_dot(projected, retain).item() == pytest.approx(
            0.0, abs=1e-6
        )
    assert stats.effective_rank == 2
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda activate unlearning
pytest -q tests/test_orthograd_geometry.py -k globally_orthogonal
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimal global basis**

Implement immutable projection stats and a basis that owns FP32 dictionaries:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OrthogradProjectionStats:
    requested_rank: int
    effective_rank: int
    dropped_directions: int
    candidate_norm: float
    projected_norm: float
    preserved_signal_ratio: float
    maximum_relative_residual: float
    basis_bytes: int
    estimated_projection_flops: int


class GlobalRetainBasis:
    def __init__(self, rank_tolerance=1e-8):
        self.rank_tolerance = float(rank_tolerance)
        self.vectors = []
        self.requested_rank = 0
        self.dropped_directions = 0
        self.orthogonalization_flops = 0

    @staticmethod
    def global_dot(left, right):
        result = None
        for name in sorted(left.keys() & right.keys()):
            value = (left[name].float() * right[name].float()).sum()
            result = value if result is None else result + value
        if result is None:
            raise RuntimeError("No common tensors for global dot product.")
        return result

    @classmethod
    def global_norm(cls, vector):
        return cls.global_dot(vector, vector).clamp_min(0.0).sqrt()

    @torch.no_grad()
    def add(self, vector):
        self.requested_rank += 1
        result = {name: tensor.detach().float().clone() for name, tensor in vector.items()}
        original_norm = self.global_norm(result)
        if not torch.isfinite(original_norm) or original_norm.item() == 0.0:
            self.dropped_directions += 1
            return False
        for _ in range(2):
            for direction in self.vectors:
                coefficient = self.global_dot(result, direction)
                for name in result.keys() & direction.keys():
                    result[name].add_(direction[name], alpha=-float(coefficient.item()))
                self.orthogonalization_flops += 4 * sum(
                    result[name].numel() for name in result.keys() & direction.keys()
                )
        norm = self.global_norm(result)
        if norm <= self.rank_tolerance * original_norm:
            self.dropped_directions += 1
            return False
        inverse = norm.reciprocal()
        for tensor in result.values():
            tensor.mul_(inverse)
        self.vectors.append(result)
        return True
```

Add `project`, `clear`, `empty`, `effective_rank`, and `basis_bytes` using one
global coefficient per basis direction. `project` returns
`(projected, removed, stats)`, where `removed = candidate - projected`.

- [ ] **Step 4: Verify the first test GREEN**

Run: `pytest -q tests/test_orthograd_geometry.py -k globally_orthogonal`

Expected: `1 passed`.

- [ ] **Step 5: Add rank, decomposition, and accounting tests**

Add tests asserting:

```python
assert basis.add(retain_a)
assert not basis.add({name: 2.0 * value for name, value in retain_a.items()})
assert basis.requested_rank == 2
assert basis.effective_rank == 1
assert basis.dropped_directions == 1
assert basis.basis_bytes == sum(t.numel() for t in retain_a.values()) * 4

reconstructed = {
    name: projected[name] + removed[name] for name in candidate
}
for name in candidate:
    torch.testing.assert_close(reconstructed[name], candidate[name].float())
```

Add a two-block counterexample that compares the implementation with separate
per-block QR projections and asserts they differ.

- [ ] **Step 6: Run and format the geometry suite**

Run:

```bash
pytest -q tests/test_orthograd_geometry.py
ruff check src/trainer/unlearn/orthograd_geometry.py tests/test_orthograd_geometry.py
ruff format --check src/trainer/unlearn/orthograd_geometry.py tests/test_orthograd_geometry.py
```

Expected: all pass.

- [ ] **Step 7: Commit geometry primitives**

```bash
git add src/trainer/unlearn/orthograd_geometry.py tests/test_orthograd_geometry.py
git commit -m "feat: add global Orthograd geometry"
```

### Task 2: Add the Orthograd trainer contract

**Files:**
- Create: `src/trainer/unlearn/orthograd.py`
- Modify: `src/trainer/__init__.py`
- Create: `configs/trainer/OrthogradUnlearn.yaml`
- Create: `tests/test_orthograd_trainer.py`

- [ ] **Step 1: Write failing registration and safe-default tests**

```python
from omegaconf import OmegaConf

from trainer import TRAINER_REGISTRY
from trainer.unlearn.orthograd import OrthogradUnlearn


def test_orthograd_trainer_is_registered():
    assert TRAINER_REGISTRY["OrthogradUnlearn"] is OrthogradUnlearn


def test_orthograd_yaml_has_mvp_defaults():
    config = OmegaConf.load("configs/trainer/OrthogradUnlearn.yaml")
    assert config.handler == "OrthogradUnlearn"
    assert config.args.save_strategy == "no"
    assert config.args.gradient_accumulation_steps == 1
    assert config.args.adam_beta1 == 0.0
    assert config.args.weight_decay == 0.0
    orthograd = config.method_args.orthograd_config
    assert orthograd.forget_weight == 0.125
    assert orthograd.retain_weight == 1.0
    assert orthograd.maximum_retain_batch_size == 4
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_orthograd_trainer.py -k 'registered or yaml'`

Expected: missing module failure.

- [ ] **Step 3: Add trainer skeleton, config, and registration**

Create `OrthogradUnlearn(GeometricUnlearn)`. Pop `orthograd_config` before
calling `super().__init__`, resolve finite positive weights and tolerance, and
create `GlobalRetainBasis`. Register it after `GeometricUnlearn`.

The YAML inherits `GeometricUnlearn` and sets:

```yaml
handler: OrthogradUnlearn

args:
  gradient_accumulation_steps: 1
  save_strategy: "no"
  adam_beta1: 0.0
  weight_decay: 0.0
  fp16: false

method_args:
  orthograd_config:
    forget_weight: 0.125
    retain_weight: 1.0
    rank_tolerance: 1.0e-8
    maximum_retain_batch_size: 4
    diagnostics_path: null
```

- [ ] **Step 4: Add failing runtime-validation tests**

Use the existing `TinyCausalLM` fixture and parameterize mutations for GAS 2,
FP16, reentrant checkpointing, partial regex selection, batch 5, nonzero
beta1/weight decay, DeepSpeed, FSDP, DDP, and empty selection. Require exact
message fragments such as `gradient_accumulation_steps=1`, `all trainable
parameters`, and `maximum retain batch size`.

- [ ] **Step 5: Implement fail-closed validation**

Reuse `make_optimizer_geometry_adapter(self.optimizer).validate(named_params)`
for optimizer-family validation, then require selected parameter IDs to equal
all trainable parameter IDs. Reject every unsupported mode from the design and
cache validation only after all checks pass.

- [ ] **Step 6: Run trainer contract tests**

Run:

```bash
pytest -q tests/test_orthograd_trainer.py -k 'registered or yaml or validation'
ruff check src/trainer/unlearn/orthograd.py src/trainer/__init__.py tests/test_orthograd_trainer.py
```

- [ ] **Step 7: Commit the contract**

```bash
git add src/trainer/unlearn/orthograd.py src/trainer/__init__.py \
  configs/trainer/OrthogradUnlearn.yaml tests/test_orthograd_trainer.py
git commit -m "feat: register full-parameter Orthograd trainer"
```

### Task 3: Implement nested slicing and per-sample retain reconstruction

**Files:**
- Modify: `src/trainer/unlearn/orthograd.py`
- Modify: `tests/test_orthograd_trainer.py`

- [ ] **Step 1: Write failing nested slicing tests**

Test dictionaries, tuples, lists, tensors, and scalar metadata. A batch of four
must produce four batch-size-one trees without mutating the source.

```python
slices = list(trainer._iter_batch_samples(inputs))
assert len(slices) == 4
assert slices[0]["input_ids"].shape[0] == 1
assert slices[0]["nested"][0].shape[0] == 1
assert slices[0]["metadata"] == "retain"
```

- [ ] **Step 2: Verify RED and implement slicing**

Run the targeted test, then implement `_batch_size` and `_slice_tree` with a
single leading-dimension consistency check. Reject an empty batch and tensors
whose leading dimension differs from labels.

- [ ] **Step 3: Write a failing per-sample reconstruction test**

On `TinyCausalLM`, compute the ordinary batch NLL retain gradient and compare
it with the arithmetic mean of `_compute_per_sample_retain_gradients`:

```python
for batch_gradient, sample_mean in zip(batch_gradients, reconstructed):
    if batch_gradient is None:
        assert sample_mean is None
    else:
        torch.testing.assert_close(batch_gradient, sample_mean, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 4: Implement per-sample gradient collection**

Each sample gets a separate forward and `torch.autograd.grad` with
`retain_graph=False`, `create_graph=False`, and `allow_unused=True`. Add
`gradient/k` to the existing FP32 retain component buffer and immediately feed
the unscaled FP32 sample dictionary into the basis before releasing the
autograd tuple.

- [ ] **Step 5: Verify reconstruction and exception cleanup**

Inject a failure on sample 2 and assert the basis and both component buffers
are empty afterward. Run:

```bash
pytest -q tests/test_orthograd_trainer.py -k 'slice or per_sample or sample_failure'
```

- [ ] **Step 6: Commit per-sample collection**

```bash
git add src/trainer/unlearn/orthograd.py tests/test_orthograd_trainer.py
git commit -m "feat: collect per-sample Orthograd retain gradients"
```

### Task 4: Implement the public training path and transactional writeback

**Files:**
- Modify: `src/trainer/unlearn/orthograd.py`
- Modify: `tests/test_orthograd_trainer.py`

- [ ] **Step 1: Write failing lifecycle and exact-gradient tests**

Train two updates on an eight-example tiny dataset with batch four and GAS one.
Assert:

```python
assert trainer.orthograd_calls == trainer.state.global_step == 2
assert trainer.retain_basis.empty
assert trainer.component_buffers.empty
assert trainer.last_orthograd_diagnostics["requested_rank"] == 4
assert trainer.last_orthograd_diagnostics["effective_rank"] >= 1
assert trainer.last_orthograd_diagnostics["maximum_relative_residual"] < 1e-6
```

For manually supplied gradients, assert every final selected gradient equals
`retain_mean + 0.125 * projected_forget`.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_orthograd_trainer.py -k 'lifecycle or exact_gradient'`

- [ ] **Step 3: Implement `training_step`**

Use this order:

```text
validate runtime
prepare inputs
compute SimNPO forget loss and gradient
store FP32 forget component
collect per-sample retain losses, gradients, mean, and basis
project once globally
write all parameter.grad tensors
prepare actual-delta probe
clear transient state in finally
return detached unprojected logging loss
```

Do not call `accelerator.backward`; every trainable parameter is selected and
the final raw gradients are written explicitly before Trainer clipping and
optimizer step.

Emit one `record_type="orthograd_geometry"` record per update with
`update_step`, `orthograd_calls`, `retain_batch_size`, requested/effective
rank, dropped directions, candidate/projected norms, preserved-signal ratio,
maximum residual, basis bytes, estimated orthogonalization/projection FLOPs,
the two combination weights, and the fixed mode label. Write through
`SurgeryDiagnosticsWriter` and populate `last_orthograd_diagnostics`.

- [ ] **Step 4: Add transactional failure tests**

Inject failures during basis insertion, projection, the middle of writeback,
and diagnostics output. After every failure assert all basis/component state
is empty and every selected `parameter.grad is None`.

- [ ] **Step 5: Add actual-update and baseline regression tests**

Record a full actual parameter delta for steps 1 and 2. Verify it is finite and
positive, the raw projected candidate is orthogonal to each basis vector, and
the trainer performs exactly one optimizer step per finalization. Re-run GU and
PCGrad lifecycle tests to ensure the new registered trainer changes no existing
path.

- [ ] **Step 6: Run the trainer and regressions**

```bash
pytest -q tests/test_orthograd_trainer.py
pytest -q tests/test_geometric_adam.py tests/test_geometric_pcgrad.py
ruff check src/trainer/unlearn/orthograd.py tests/test_orthograd_trainer.py
ruff format --check src/trainer/unlearn/orthograd.py tests/test_orthograd_trainer.py
```

- [ ] **Step 7: Commit public training behavior**

```bash
git add src/trainer/unlearn/orthograd.py tests/test_orthograd_trainer.py
git commit -m "feat: train with per-sample Orthograd projection"
```

### Task 5: Add shared resource and phase profiling

**Files:**
- Create: `src/trainer/resource_profiler.py`
- Modify: `src/trainer/unlearn/geometric.py`
- Modify: `src/trainer/unlearn/orthograd.py`
- Modify: `configs/trainer/GeometricUnlearn.yaml`
- Modify: `configs/trainer/OrthogradUnlearn.yaml`
- Create: `tests/test_resource_profiler.py`
- Modify: `tests/test_geometric_adam.py`
- Modify: `tests/test_orthograd_trainer.py`

- [ ] **Step 1: Write failing profile aggregation tests**

Construct step records `[100, 20, 30]` milliseconds and assert the summary
separates step 1 and aggregates steps 2 onward with deterministic p50/mean/p95.
Also require phase totals and nonnegative allocator peaks.

- [ ] **Step 2: Implement `ResourceProfileCallback`**

The callback accepts a JSON path and enabled flag. On train begin it resets CUDA
peaks and clears records. On step begin/end it synchronizes once, uses
`time.perf_counter_ns`, and records update duration. On train end it atomically
writes sorted JSON containing:

```python
{
    "record_type": "training_resource_profile",
    "update_count": len(step_ms),
    "first_update_ms": step_ms[0],
    "steady_update_mean_ms": statistics.fmean(step_ms[1:]),
    "steady_update_p50_ms": quantile(step_ms[1:], 0.50),
    "steady_update_p95_ms": quantile(step_ms[1:], 0.95),
    "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
    "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
    "phase_totals_ms": aggregate_phases(),
}
```

Use linear interpolation with a locally implemented deterministic quantile;
do not depend on NumPy.

- [ ] **Step 3: Add trainer registration and phase events**

Both trainers read `resource_profile_path`. When set, add the callback. Add a
small `CudaPhaseRecorder` that records start/end events and resolves all events
once per update. GU phases are `component_gradients`, `projection_writeback`;
OrthoGrad phases are `forget_gradient`, `per_sample_retain_gradients`,
`orthonormalization`, and `projection_writeback`.

- [ ] **Step 4: Add profiling-off equivalence tests**

With profiling disabled, compare one GU update before/after instrumentation and
require bitwise-equal tiny-model parameters. With it enabled, require a valid
JSON profile and positive step time.

- [ ] **Step 5: Run profiling and trainer regressions**

```bash
pytest -q tests/test_resource_profiler.py
pytest -q tests/test_geometric_adam.py -k 'profile or equivalence'
pytest -q tests/test_orthograd_trainer.py -k profile
ruff check src/trainer/resource_profiler.py src/trainer/unlearn/geometric.py \
  src/trainer/unlearn/orthograd.py tests/test_resource_profiler.py
```

- [ ] **Step 6: Commit shared profiling**

```bash
git add src/trainer/resource_profiler.py src/trainer/unlearn/geometric.py \
  src/trainer/unlearn/orthograd.py configs/trainer/GeometricUnlearn.yaml \
  configs/trainer/OrthogradUnlearn.yaml tests/test_resource_profiler.py \
  tests/test_geometric_adam.py tests/test_orthograd_trainer.py
git commit -m "feat: profile GU and Orthograd resource overhead"
```

### Task 6: Add process-tree CPU/GPU sampling

**Files:**
- Create: `scripts/profile_process_tree.py`
- Create: `tests/test_orthograd_scripts.py`

- [ ] **Step 1: Write failing child-process profile tests**

Launch a Python child that allocates memory for at least 300 ms. Assert the
wrapper propagates its exit code and atomically writes wall seconds, sample
count, peak process-tree RSS, GPU baseline/peak/delta bytes, command, and poll
interval. Mock only NVML reads; use a real child process and real psutil RSS.

- [ ] **Step 2: Implement the wrapper**

Parse:

```text
profile_process_tree.py --output profile.json --gpu 0 --poll-ms 100 -- COMMAND...
```

Use `subprocess.Popen(start_new_session=True)`, recursively sum RSS with
`psutil.Process(pid).children(recursive=True)`, and query device memory via
`pynvml`. Forward INT/TERM/HUP to the process group, wait for every child, and
write the profile even when the command fails.

- [ ] **Step 3: Add signal and failure tests**

Require nonzero child exit propagation, JSON persistence, no orphan process,
and explicit errors when NVML or the requested GPU is unavailable.

- [ ] **Step 4: Run and format tests**

```bash
pytest -q tests/test_orthograd_scripts.py -k process_profile
ruff check scripts/profile_process_tree.py tests/test_orthograd_scripts.py
ruff format --check scripts/profile_process_tree.py tests/test_orthograd_scripts.py
```

- [ ] **Step 5: Commit process profiling**

```bash
git add scripts/profile_process_tree.py tests/test_orthograd_scripts.py
git commit -m "feat: sample training process resource peaks"
```

### Task 7: Add BF16, FlashAttention 2, and PagedAdamW32 integration

**Files:**
- Create: `tests/test_orthograd_gpu_integration.py`

- [ ] **Step 1: Write a failing BF16 eager-vs-Flash test**

Reuse the tiny Llama construction and fixed synthetic batches from
`tests/test_gu_gpu_integration.py`. Run one public optimizer update with retain
batch two through eager and FlashAttention 2. Assert finite diagnostics,
effective rank, residual below `1e-4`, and BF16-appropriate parameter-update
agreement.

- [ ] **Step 2: Add a real PagedAdamW32 two-update test**

Use `pytest.importorskip("bitsandbytes")`, create the optimizer through the
public Trainer path, and require two positive parameter deltas, FP32 optimizer
state after initialization, two finalizations, and positive CUDA peaks.

- [ ] **Step 3: Add a measured batch-four feasibility test**

On a tiny CUDA model use retain batch four and assert the analytic basis bytes
equal `4*d*rank`, measured reserved memory is at least allocated memory, and
all transient basis tensors are released after the update.

- [ ] **Step 4: Run GPU integrations**

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/test_gu_gpu_integration.py tests/test_orthograd_gpu_integration.py
```

Expected: zero failures and no hidden hardware skips.

- [ ] **Step 5: Commit GPU coverage**

```bash
git add tests/test_orthograd_gpu_integration.py
git commit -m "test: validate Orthograd on BF16 Flash and PagedAdamW32"
```

### Task 8: Add strict no-checkpoint launchers and analyzer

**Files:**
- Create: `scripts/orthograd_mvp_arm.sh`
- Create: `scripts/orthograd_mvp_matrix.sh`
- Create: `scripts/analyze_orthograd_mvp.py`
- Modify: `tests/test_orthograd_scripts.py`

- [ ] **Step 1: Write failing launcher contract tests**

Require exactly three methods and these common tokens:

```python
for token in (
    "trainer.args.per_device_train_batch_size=4",
    "trainer.args.gradient_accumulation_steps=1",
    "trainer.args.max_steps=10",
    "trainer.args.learning_rate=1e-5",
    "trainer.args.optim=paged_adamw_32bit",
    "trainer.args.warmup_epochs=0",
    "trainer.args.bf16=true",
    "model.model_args.attn_implementation=flash_attention_2",
    "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
    "save_model_after_train=false",
    "trainer.args.save_strategy=no",
):
    assert token in arm_text
```

Require `control -> GeometricUnlearn/gu_enabled=false`, `gu ->
GeometricUnlearn/gu_enabled=true`, and `orthograd -> OrthogradUnlearn`.

- [ ] **Step 2: Implement one arm and sequential matrix**

Stage under `/tmp/orthograd_mvp/<timestamp>/<method>`. Wrap the exact
`accelerate launch` command with `profile_process_tree.py`. Run methods
sequentially on physical GPU 0 and write a seven-column atomic manifest with
start/end UTC, exit code, and command.

- [ ] **Step 3: Implement fail-closed artifact persistence**

Persist only run log, resolved Hydra config, TOFU summary, method diagnostics,
training resource profile, and process resource profile. Audit symlinks,
`checkpoint-*`, safetensors, `.bin`, `.pt`, `.pth`, `.ckpt`, optimizer,
scheduler, RNG, and Trainer state before and after publication.

- [ ] **Step 4: Write failing analyzer fixtures**

Create fake three-arm outputs. Assert the analyzer validates method mapping,
all common config fields, ten updates, identical metric keys, finite resource
values, memory ordering, zero payloads, and Orthograd rank/residual criteria.

- [ ] **Step 5: Implement deterministic analysis**

Produce raw metrics, GU-Control and OrthoGrad-Control deltas, and:

```python
time_overhead = method_train_runtime / control_train_runtime - 1.0
allocated_overhead = method_peak_allocated / control_peak_allocated - 1.0
nvml_overhead = method_nvml_delta / control_nvml_delta - 1.0
```

Report first-step, steady mean/p50/p95, phase totals, CPU RSS, basis bytes,
estimated projection FLOPs, rank, signal ratio, and residual. Use `math.fsum`
for every floating aggregate and atomic sorted JSON/Markdown output.

- [ ] **Step 6: Run script tests and syntax checks**

```bash
pytest -q tests/test_orthograd_scripts.py
bash -n scripts/orthograd_mvp_arm.sh scripts/orthograd_mvp_matrix.sh
ruff check scripts/analyze_orthograd_mvp.py scripts/profile_process_tree.py \
  tests/test_orthograd_scripts.py
```

- [ ] **Step 7: Commit smoke infrastructure**

```bash
git add scripts/orthograd_mvp_arm.sh scripts/orthograd_mvp_matrix.sh \
  scripts/analyze_orthograd_mvp.py tests/test_orthograd_scripts.py
git commit -m "feat: add matched Orthograd overhead matrix"
```

### Task 9: Run the real three-arm matrix

**Files:**
- Create: `ORTHOGRAD_MVP_REPORT.md`
- Runtime: `saves/exp/ORTHOGRAD_MVP/<timestamp>/`

- [ ] **Step 1: Verify idle resources and model cache**

```bash
conda activate unlearning
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
free -h
test -d /root/.cache/huggingface
```

Require GPU 0 to have at least 70 GiB free and host available memory to exceed
the analytic basis plus two component vectors with 20% headroom.

- [ ] **Step 2: Launch the matrix**

```bash
timestamp=$(date -u +%m%d%H%M%S)
bash scripts/orthograd_mvp_matrix.sh "$timestamp"
```

If OrthoGrad OOMs, retain only logs/resource telemetry, audit zero checkpoint
payloads, diagnose measured versus analytic memory, and do not fabricate an
endpoint result.

- [ ] **Step 3: Audit successful outputs**

Require three exit-zero manifest rows, three summaries/configs/resource
profiles, ten GU/OrthoGrad geometry records, no orphan process, and no
checkpoint payload in persistent or local roots.

- [ ] **Step 4: Generate report and JSON**

```bash
python scripts/analyze_orthograd_mvp.py \
  "saves/exp/ORTHOGRAD_MVP/${timestamp}" \
  --markdown ORTHOGRAD_MVP_REPORT.md \
  --json "saves/exp/ORTHOGRAD_MVP/${timestamp}/analysis.json"
```

- [ ] **Step 5: Inspect claims and commit report**

Verify every number against raw JSON, state that one seed/ten steps only
supports feasibility and overhead, then commit:

```bash
git add ORTHOGRAD_MVP_REPORT.md
git commit -m "docs: report Orthograd and GU overhead"
```

### Task 10: Final verification and independent review

**Files:**
- All files above

- [ ] **Step 1: Run full CPU regressions**

```bash
pytest -q \
  tests/test_orthograd_geometry.py \
  tests/test_orthograd_trainer.py \
  tests/test_resource_profiler.py \
  tests/test_orthograd_scripts.py \
  tests/test_component_buffers.py \
  tests/test_optimizer_geometry.py \
  tests/test_gradient_surgery.py \
  tests/test_geometric_adam.py \
  tests/test_geometric_pcgrad.py
```

- [ ] **Step 2: Run full GPU integrations**

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/test_gu_gpu_integration.py tests/test_orthograd_gpu_integration.py
```

- [ ] **Step 3: Run static and artifact checks**

Run Ruff check/format on every changed Python file, `bash -n` on both shell
files, `git diff --check abfe95f HEAD`, JSON parsing, repeated analyzer
generation with byte comparison, and checkpoint audits over every Orthograd
runtime root.

- [ ] **Step 4: Request independent review**

Review `abfe95f..HEAD` for per-sample loss reconstruction, global rather than
blockwise geometry, double MGS stability, SimNPO sign and `1:0.125` weights,
transactional clearing, Trainer clipping/step reachability, PagedAdamW32
labeling, profiler boundary fairness, NVML/RSS semantics, and zero-checkpoint
failure paths.

- [ ] **Step 5: Fix every Critical/Important finding test-first**

For each valid finding, add a failing regression, verify RED, implement the
minimal fix, verify GREEN, and request focused re-review.

- [ ] **Step 6: Re-run all final verification**

Repeat Steps 1–3 after the last review fix. Record exact CPU/GPU test counts,
report hashes, matrix root, HEAD, and clean status.

- [ ] **Step 7: Hand off without merging or pushing**

Keep `feat/orthograd` and its worktree. Report the commit range, test counts,
runtime root, measured GU/OrthoGrad overhead, feasibility verdict, limitations,
and checkpoint payload count. Do not modify `dev0`, `feat/uam-gu`, or their
user-owned changes.
