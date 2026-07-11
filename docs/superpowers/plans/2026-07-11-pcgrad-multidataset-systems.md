# PCGrad Multi-Dataset and Systems Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add retain-prioritized PCGrad, PagedAdamW32 frozen-coordinate support, BF16 FlashAttention 2 execution, memory-bounded 7B component buffers, and matched no-checkpoint TOFU/MUSE smoke experiments on `dev0`.

**Architecture:** Separate pure gradient-surgery decisions, optimizer-state access, component-buffer storage, and delta diagnostics from `GeometricUnlearn`. The trainer continues to compute independent effective-batch component gradients and overwrite selected `.grad` values before Transformers clips and steps. Torch Adam/AdamW and PagedAdamW32 share a frozen square-root coordinate interface; large MUSE buffers stream through pinned CPU memory.

**Tech Stack:** Python 3.11, PyTorch 2.4.1, Transformers 4.45.1, Accelerate 0.34.2, bitsandbytes 0.44.1, FlashAttention 2.6.3, Hydra/OmegaConf, pytest 8.3.5, Ruff 0.6.9, two A100-80GB GPUs.

---

## File Map

- Create `src/trainer/unlearn/gradient_surgery.py`: pure GU/PCGrad decisions.
- Create `src/trainer/unlearn/optimizer_geometry.py`: Torch and PagedAdamW32 state adapters.
- Create `src/trainer/unlearn/component_buffers.py`: parameter-device and pinned-CPU FP32 stores.
- Create `src/trainer/unlearn/gu_diagnostics.py`: JSONL summaries and actual-delta callback.
- Modify `src/trainer/unlearn/geometric.py`: stream finalization through the focused modules.
- Modify `configs/trainer/GeometricUnlearn.yaml`: surgery, buffer, and diagnostics configuration.
- Create `tests/test_gradient_surgery.py`, `tests/test_optimizer_geometry.py`, `tests/test_component_buffers.py`, `tests/test_geometric_pcgrad.py`, and `tests/test_gu_gpu_integration.py`.
- Modify `tests/test_geometric_adam.py` and `pytest.ini`.
- Create `scripts/pcgrad_smoke_arm.sh`, `scripts/pcgrad_smoke_matrix.sh`, and `scripts/analyze_pcgrad_smoke.py`.
- Create `tests/test_pcgrad_scripts.py` and `tests/test_analyze_pcgrad_smoke.py`.
- Create `PCGRAD_MULTIDATASET_REPORT.md` after real runs finish.

## Task 1: Extract Pure GU and PCGrad Surgery

**Files:**
- Create: `src/trainer/unlearn/gradient_surgery.py`
- Create: `tests/test_gradient_surgery.py`
- Modify: `tests/test_geometric_adam.py`

- [ ] **Step 1: Write failing negative-conflict and positive-alignment tests**

```python
def test_negative_dot_pcgrad_exactly_matches_gu():
    dot, retain_sq = torch.tensor(-2.0), torch.tensor(4.0)
    gu = decide_global_surgery("gu", dot, retain_sq, 1e-12)
    pc = decide_global_surgery("pcgrad", dot, retain_sq, 1e-12)
    assert gu.coefficient.item() == pytest.approx(-0.5)
    assert pc.coefficient.item() == pytest.approx(-0.5)
    assert pc.conflict is True


def test_nonnegative_dot_pcgrad_is_identity_while_gu_projects():
    dot, retain_sq = torch.tensor(2.0), torch.tensor(4.0)
    gu = decide_global_surgery("gu", dot, retain_sq, 1e-12)
    pc = decide_global_surgery("pcgrad", dot, retain_sq, 1e-12)
    assert pc.coefficient.item() == 0.0
    assert pc.conflict is False
    assert gu.coefficient.item() == pytest.approx(0.5)
```

- [ ] **Step 2: Run `pytest tests/test_gradient_surgery.py -q` and verify import failure**

Expected: `trainer.unlearn.gradient_surgery` is missing.

- [ ] **Step 3: Implement the minimal strategy module**

```python
@dataclass(frozen=True)
class SurgeryDecision:
    mode: str
    coefficient: torch.Tensor
    raw_coefficient: torch.Tensor
    conflict: bool


def decide_global_surgery(mode, dot, retain_sq, eps):
    if mode not in {"gu", "pcgrad"}:
        raise ValueError(f"Unsupported gradient surgery mode: {mode}")
    raw = dot / retain_sq.clamp_min(eps)
    conflict = bool(dot.item() < 0)
    coefficient = raw if mode == "gu" or conflict else torch.zeros_like(raw)
    return SurgeryDecision(mode, coefficient, raw, conflict)


def apply_surgery_tensor(forget, retain, decision):
    return forget.float() - decision.coefficient * retain.float()
```

- [ ] **Step 4: Add global two-tensor, blockwise-counterexample, and zero-retain tests**

The global test sums dots before choosing one coefficient. The zero-retain test requires a finite zero coefficient and an unchanged forget tensor.

- [ ] **Step 5: Run focused and existing geometry tests**

```bash
conda activate unlearning
pytest tests/test_gradient_surgery.py \
  tests/test_geometric_adam.py::test_global_projection_is_orthogonal_across_tensors \
  tests/test_geometric_adam.py::test_global_projection_differs_from_blockwise_projection -q
```

- [ ] **Step 6: Commit**

```bash
git add src/trainer/unlearn/gradient_surgery.py tests/test_gradient_surgery.py tests/test_geometric_adam.py
git commit -m "feat: add retain-prioritized PCGrad geometry"
```

## Task 2: Extract Frozen Optimizer Geometry Adapters

**Files:**
- Create: `src/trainer/unlearn/optimizer_geometry.py`
- Create: `tests/test_optimizer_geometry.py`
- Modify: `src/trainer/unlearn/geometric.py`
- Modify: `tests/test_geometric_adam.py`

- [ ] **Step 1: Write failing Torch state tests**

Instantiate AdamW with `betas=(0.0, 0.999)` and zero weight decay. Verify empty state returns identity, and seeded `step=3` plus nonuniform `exp_avg_sq` returns:

```python
v_hat = exp_avg_sq.float() / (1.0 - beta2**3)
expected = (v_hat.sqrt() + eps).sqrt()
```

Also require missing-step, missing-moment, negative, non-finite, wrong-shape, and non-positive-step states to raise targeted errors.

- [ ] **Step 2: Run `pytest tests/test_optimizer_geometry.py -q` and verify RED**

- [ ] **Step 3: Implement adapters and factory**

```python
class OptimizerGeometryAdapter:
    name = "base"

    def validate(self, named_params):
        raise NotImplementedError

    def prefetch(self, parameter):
        return None

    def sqrt_denominator(self, parameter, group):
        raise NotImplementedError


class TorchAdamGeometryAdapter(OptimizerGeometryAdapter):
    name = "torch_adam"


class PagedAdamW32GeometryAdapter(OptimizerGeometryAdapter):
    name = "paged_adamw_32bit"
```

Implement `make_optimizer_geometry_adapter(optimizer)` by unwrapping Accelerate wrappers, returning `TorchAdamGeometryAdapter` for `torch.optim.Adam`/`AdamW`, returning `PagedAdamW32GeometryAdapter` only when the class module starts with `bitsandbytes`, `is_paged is True`, and `args.optim_bits == 32`, and otherwise raising `NotImplementedError` with the optimizer class name. Torch reads `exp_avg_sq`; PagedAdamW32 reads `state2`.

- [ ] **Step 4: Implement fail-closed group and state validation**

Every group must have beta1 zero, weight decay zero, AMSGrad disabled, and positive finite eps. Paged `state2.dtype == torch.uint8` raises an explicit PagedAdamW8 error. Call `prefetch_state(parameter)` before reading initialized paged buffers.

- [ ] **Step 5: Replace direct trainer state access with the adapter**

Keep optimizer-wrapper unwrapping in the adapter module. Remove duplicate Torch-specific state interpretation from `geometric.py`.

- [ ] **Step 6: Run `pytest tests/test_optimizer_geometry.py tests/test_geometric_adam.py -q`**

- [ ] **Step 7: Commit**

```bash
git add src/trainer/unlearn/optimizer_geometry.py src/trainer/unlearn/geometric.py tests/test_optimizer_geometry.py tests/test_geometric_adam.py
git commit -m "refactor: isolate GU optimizer geometry"
```

## Task 3: Add Memory-Bounded Component Buffers

**Files:**
- Create: `src/trainer/unlearn/component_buffers.py`
- Create: `tests/test_component_buffers.py`
- Modify: `src/trainer/unlearn/geometric.py`

- [ ] **Step 1: Write failing device-equivalence and lifecycle tests**

Accumulate identical synthetic gradients twice in `parameter` and `cpu` modes, stream them back to the parameter device, assert FP32 equality, then verify `clear_component` and `clear` release entries.

- [ ] **Step 2: Write a failing host-memory preflight test**

Patch available memory one byte below `2 * selected_numel * 4 * 1.2` and require `RuntimeError("Insufficient host memory for GU component buffers")`.

- [ ] **Step 3: Run `pytest tests/test_component_buffers.py -q` and verify RED**

- [ ] **Step 4: Implement the store and preflight**

```python
class ComponentGradientBuffers:
    def __init__(self, device_mode, pin_memory=True):
        if device_mode not in {"cpu", "parameter"}:
            raise ValueError(f"Unsupported component buffer device: {device_mode}")
        self.device_mode = device_mode
        self.pin_memory = pin_memory
        self._data = {"forget": {}, "retain": {}}

    @property
    def empty(self):
        return not self._data["forget"] and not self._data["retain"]
```

Implement `add`, `tensor`, `has_component`, `clear_component`, and `clear` against `_data`. `add` validates component name, zips named parameters with gradients, lazily allocates FP32 storage on CPU or the parameter device, and adds detached gradients. `tensor` returns the named tensor on the requested parameter device. CPU storage is pinned when CUDA is available, and copies complete before source autograd tensors are deleted. `required_host_bytes` returns `ceil(2 * selected_numel * 4 * headroom)`; `validate_host_memory` compares it with available pages from `os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")` unless an explicit test value is supplied.

- [ ] **Step 5: Differentiate and accumulate components sequentially**

Compute forget grads, accumulate, and delete the tuple before computing retain grads. Preserve the graph until standard total backward finishes.

- [ ] **Step 6: Run buffer, GAS, and short-window tests**

```bash
pytest tests/test_component_buffers.py \
  tests/test_geometric_adam.py::test_gradient_accumulation_matches_full_effective_batch \
  tests/test_geometric_adam.py::test_short_final_accumulation_window_is_projected -q
```

- [ ] **Step 7: Commit**

```bash
git add src/trainer/unlearn/component_buffers.py src/trainer/unlearn/geometric.py tests/test_component_buffers.py
git commit -m "feat: stream GU component gradients through CPU buffers"
```

## Task 4: Integrate Streaming PCGrad Finalization

**Files:**
- Modify: `src/trainer/unlearn/geometric.py`
- Create: `tests/test_geometric_pcgrad.py`
- Modify: `configs/trainer/GeometricUnlearn.yaml`

- [ ] **Step 1: Write failing Trainer-mode tests**

Build seeded tiny trainers for `gradient_surgery=gu` and `pcgrad`. A negative-dot effective batch must produce identical updated parameters; an injected positive coordinate case must leave PCGrad forget gradients unchanged and report nonzero PCGrad-GU distance.

- [ ] **Step 2: Write failing lifecycle tests**

```python
assert trainer.surgery_calls == trainer.state.global_step == 2
assert trainer.last_surgery_diagnostics["mode"] == "pcgrad"
assert "relative_pcgrad_gu_distance" in trainer.last_surgery_diagnostics
assert trainer.component_buffers.empty
```

- [ ] **Step 3: Run `pytest tests/test_geometric_pcgrad.py -q` and verify RED**

- [ ] **Step 4: Implement two-pass streaming finalization**

Pass one streams each raw forget/retain pair to the parameter device, applies the frozen divisor, and accumulates global dot and squared norms. Choose one global `SurgeryDecision`. Pass two recomputes per-parameter coordinates, applies surgery, combines `gamma * forget + alpha * retain`, maps back, and copies to `.grad`. Do not create transformed full-model dictionaries.

- [ ] **Step 5: Add backward-compatible configuration**

```yaml
geometric_config:
  gradient_surgery: gu
  component_buffer_device: parameter
  diagnostics_path: null
  actual_delta_mode: off
  actual_delta_steps: [1, 10]
  actual_delta_sample_elements: 1000000
```

`gu_enabled=false` still delegates to the parent. An enabled historical config without `gradient_surgery` resolves to `gu`.

- [ ] **Step 6: Run Trainer and baseline-recovery suites**

```bash
pytest tests/test_geometric_pcgrad.py tests/test_geometric_adam.py -q
```

- [ ] **Step 7: Commit**

```bash
git add src/trainer/unlearn/geometric.py configs/trainer/GeometricUnlearn.yaml tests/test_geometric_pcgrad.py
git commit -m "feat: run PCGrad on effective-batch gradients"
```

## Task 5: Persist Geometry and Actual-Delta Diagnostics

**Files:**
- Create: `src/trainer/unlearn/gu_diagnostics.py`
- Modify: `src/trainer/unlearn/geometric.py`
- Modify: `tests/test_geometric_pcgrad.py`

- [ ] **Step 1: Write failing JSONL aggregation tests**

Two synthetic records must aggregate surgery count, conflict rate, maximum residual, mean surgery magnitude, and mean PCGrad-GU distance without treating missing values as zero.

- [ ] **Step 2: Write failing callback tests**

Use `on_pre_optimizer_step` to snapshot parameter probes and `on_optimizer_step` to compute deltas. Full mode verifies exact raw forget/retain directional derivatives. Sampled mode verifies deterministic SHA-256-derived indices and `coverage="sampled"`.

- [ ] **Step 3: Run the diagnostics node ids and verify RED**

- [ ] **Step 4: Implement writer, aggregator, and callback**

Implement `SurgeryDiagnosticsWriter.write_step(record)` as sorted-key JSONL append followed by flush, and `write_summary()` as a readback aggregation that atomically replaces the summary JSON. Implement `ActualDeltaCallback.on_pre_optimizer_step` to snapshot configured full or sampled parameter values before the step, and `on_optimizer_step` to subtract the snapshot, compute stored component directional derivatives, emit the result, and return the unchanged Trainer control. Full mode is permitted only for the TOFU systems-isolation run. MUSE samples one million fixed parameter elements at updates 1 and 10. Clear snapshots and component probes in `finally` blocks.

- [ ] **Step 5: Write one JSONL geometry record per optimizer update**

Write actual-delta records only after `on_optimizer_step`. Store JSON/JSONL in the experiment directory without calling any Trainer checkpoint API.

- [ ] **Step 6: Run `pytest tests/test_geometric_pcgrad.py tests/test_geometric_adam.py -q`**

- [ ] **Step 7: Commit**

```bash
git add src/trainer/unlearn/gu_diagnostics.py src/trainer/unlearn/geometric.py tests/test_geometric_pcgrad.py
git commit -m "feat: record PCGrad conflict and delta diagnostics"
```

## Task 6: Support PagedAdamW32 and BF16 FlashAttention 2

**Files:**
- Create: `tests/test_gu_gpu_integration.py`
- Modify: `tests/test_optimizer_geometry.py`
- Modify: `pytest.ini`
- Modify: `src/trainer/unlearn/geometric.py`

- [ ] **Step 1: Register the GPU marker**

```ini
markers =
    gpu: requires a CUDA GPU and optional CUDA extensions
```

- [ ] **Step 2: Write a failing real PagedAdamW32 test**

On CUDA instantiate `bitsandbytes.optim.AdamW` with `optim_bits=32` and `is_paged=True`, execute one real step, and require integer step 1, FP32 `state2`, finite square-root factors, and adapter name `paged_adamw_32bit`. A fake uint8 state must raise the PagedAdamW8 message.

- [ ] **Step 3: Write a failing BF16 FlashAttention equivalence test**

Create eager and FlashAttention 2 tiny `LlamaForCausalLM` models from one state dict. Use BF16 CUDA parameters, no padding, and non-reentrant checkpointing. Compare component losses and gradients with `rtol=2e-2, atol=2e-3`; require finite tensors.

- [ ] **Step 4: Run GPU tests and verify RED**

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/test_gu_gpu_integration.py -m gpu -q
```

- [ ] **Step 5: Extend runtime validation**

Accept BF16 FlashAttention 2 and PagedAdamW32. Continue rejecting FP16, non-paged bitsandbytes, PagedAdamW8, invalid groups, unsupported state, multi-process, DeepSpeed, FSDP, and Apex before the first component forward.

- [ ] **Step 6: Run GPU and CPU suites**

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/test_gu_gpu_integration.py -m gpu -q
pytest tests/test_geometric_adam.py tests/test_gradient_surgery.py \
  tests/test_optimizer_geometry.py tests/test_component_buffers.py \
  tests/test_geometric_pcgrad.py -q
```

- [ ] **Step 7: Commit**

```bash
git add pytest.ini src/trainer/unlearn/geometric.py tests/test_gu_gpu_integration.py tests/test_optimizer_geometry.py
git commit -m "feat: support PagedAdamW32 and FlashAttention PCGrad"
```

## Task 7: Add No-Checkpoint Smoke Launchers

**Files:**
- Create: `scripts/pcgrad_smoke_arm.sh`
- Create: `scripts/pcgrad_smoke_matrix.sh`
- Create: `tests/test_pcgrad_scripts.py`

- [ ] **Step 1: Write failing launcher-contract tests**

Require both scripts to activate `unlearning`, disable all model/state saves, select PagedAdamW32, beta1 zero, weight decay zero, BF16, FlashAttention 2, non-reentrant checkpointing, CPU buffers for MUSE, and audit all checkpoint payload names.

- [ ] **Step 2: Run `pytest tests/test_pcgrad_scripts.py -q` and verify RED**

- [ ] **Step 3: Implement one-arm launcher**

```text
pcgrad_smoke_arm.sh <dataset> <method> <gpu> <timestamp> <system_mode>
```

Datasets: `tofu01`, `tofu05`, `tofu10`, `muse_news`, `muse_books`. Methods: `control`, `gu`, `pcgrad`. System modes: `production`, `torch_flash`, `torch_eager_fp32`. Map every value to explicit Hydra overrides, tee logs, run in-memory evaluation, and fail if checkpoint payload search finds a file.

- [ ] **Step 4: Implement a bounded two-GPU matrix scheduler**

Run at most one process on CUDA 0 and one on CUDA 1. Schedule 15 production algorithm arms followed by the two additional TOFU01 PCGrad system arms. Record PID, GPU, command, start/end time, and exit code in `RUN_MANIFEST.tsv`. Stop scheduling after a failure and wait for already-running arms.

- [ ] **Step 5: Verify scripts**

```bash
bash -n scripts/pcgrad_smoke_arm.sh scripts/pcgrad_smoke_matrix.sh
pytest tests/test_pcgrad_scripts.py -q
```

- [ ] **Step 6: Commit**

```bash
git add scripts/pcgrad_smoke_arm.sh scripts/pcgrad_smoke_matrix.sh tests/test_pcgrad_scripts.py
git commit -m "feat: add no-checkpoint PCGrad smoke matrix"
```

## Task 8: Add Result Aggregation and Advancement Decision

**Files:**
- Create: `scripts/analyze_pcgrad_smoke.py`
- Create: `tests/test_analyze_pcgrad_smoke.py`

- [ ] **Step 1: Write failing synthetic aggregation tests**

Create temporary control/GU/PCGrad summaries and JSONL diagnostics. Require raw tables, method deltas, conflict rate, mean relative PCGrad-GU distance, `degenerate_to_gu`, and:

```python
advance = conflict_rate < 0.95 or mean_pcgrad_gu_distance > 1e-4
global_stop = all(rate == 1.0 and distance <= 1e-6 for rate, distance in rows)
```

- [ ] **Step 2: Run `pytest tests/test_analyze_pcgrad_smoke.py -q` and verify RED**

- [ ] **Step 3: Implement the analyzer**

Accept a timestamped matrix root, locate arms from `RUN_MANIFEST.tsv`, validate that matched configs differ only in method/task/output fields, emit Markdown and JSON, and raise on missing metrics instead of filling zeros.

- [ ] **Step 4: Run tests and Ruff**

```bash
pytest tests/test_analyze_pcgrad_smoke.py -q
ruff check scripts/analyze_pcgrad_smoke.py tests/test_analyze_pcgrad_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_pcgrad_smoke.py tests/test_analyze_pcgrad_smoke.py
git commit -m "feat: analyze PCGrad conflict across datasets"
```

## Task 9: Run the Smoke Matrix and Write Evidence

**Files:**
- Create: `PCGRAD_MULTIDATASET_REPORT.md`

- [ ] **Step 1: Run preflight**

```bash
conda activate unlearning
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
pytest tests/test_geometric_adam.py tests/test_gradient_surgery.py \
  tests/test_optimizer_geometry.py tests/test_component_buffers.py \
  tests/test_geometric_pcgrad.py tests/test_pcgrad_scripts.py \
  tests/test_analyze_pcgrad_smoke.py -q
CUDA_VISIBLE_DEVICES=0 pytest tests/test_gu_gpu_integration.py -m gpu -q
```

Expected: each GPU uses less than 500 MiB and every test passes.

- [ ] **Step 2: Launch with persistent logs**

```bash
timestamp=$(date +%m%d%H%M%S)
mkdir -p "saves/exp/PCGRAD_SMOKE/${timestamp}"
bash scripts/pcgrad_smoke_matrix.sh "$timestamp" 2>&1 | \
  tee "saves/exp/PCGRAD_SMOKE/${timestamp}/matrix.log"
```

Monitor both GPUs, logs, and the manifest until every arm exits.

- [ ] **Step 3: Audit artifacts**

Search the matrix root for `*.safetensors`, `pytorch_model*.bin`, `training_args.bin`, `trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, and `*.ckpt`. Expected: zero matches.

- [ ] **Step 4: Generate reports**

```bash
python scripts/analyze_pcgrad_smoke.py \
  "saves/exp/PCGRAD_SMOKE/${timestamp}" \
  --markdown PCGRAD_MULTIDATASET_REPORT.md \
  --json "saves/exp/PCGRAD_SMOKE/${timestamp}/analysis.json"
```

The report states each dataset's advancement decision or the cross-dataset PCGrad-to-GU degeneracy conclusion.

- [ ] **Step 5: Commit checked evidence**

```bash
git add PCGRAD_MULTIDATASET_REPORT.md
git diff --cached --check
git commit -m "docs: report PCGrad multi-dataset smoke results"
```

## Task 10: Final Verification and Handoff

**Files:**
- Verify all Task 1-9 files.

- [ ] **Step 1: Run focused tests fresh**

```bash
conda activate unlearning
pytest tests/test_geometric_adam.py tests/test_gradient_surgery.py \
  tests/test_optimizer_geometry.py tests/test_component_buffers.py \
  tests/test_geometric_pcgrad.py tests/test_pcgrad_scripts.py \
  tests/test_analyze_pcgrad_smoke.py -q
CUDA_VISIBLE_DEVICES=0 pytest tests/test_gu_gpu_integration.py -m gpu -q
```

- [ ] **Step 2: Run static and artifact gates**

```bash
ruff check src/trainer/unlearn src/train.py scripts/analyze_pcgrad_smoke.py tests
ruff format --check src/trainer/unlearn src/train.py scripts/analyze_pcgrad_smoke.py tests
bash -n scripts/pcgrad_smoke_arm.sh scripts/pcgrad_smoke_matrix.sh
git diff --check
```

Repeat the checkpoint-payload audit against the completed matrix root.

- [ ] **Step 3: Verify branch and user changes**

Require `git branch --show-current` to print `dev0`. Confirm only task files were committed and pre-existing user modifications to documentation, model configs, and legacy scripts remain unstaged and unchanged.

- [ ] **Step 4: Request review**

Review from the parent of this plan commit through final HEAD against the approved design. Fix all Critical and Important findings, then rerun Steps 1-3.

- [ ] **Step 5: Keep work on `dev0`**

Do not merge, push, or create the UAM branch. Report commits, test counts, raw result paths, conflict/degeneracy conclusion, and checkpoint audit.
