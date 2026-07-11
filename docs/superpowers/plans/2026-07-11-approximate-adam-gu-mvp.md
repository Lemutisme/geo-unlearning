# Approximate Adam/AdamW GU MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unreachable GU hook with a reachable, globally projected, gradient-accumulation-correct Stage-A implementation for standard Torch Adam/AdamW and validate it with SimNPO on TOFU.

**Architecture:** `GeometricUnlearn.training_step()` computes independent forget and retain gradients, accumulates raw FP32 component buffers, transforms the effective-batch gradients with the frozen pre-step Adam square-root metric, performs one global rank-one projection, and writes the optimizer-equivalent raw gradient to selected `p.grad` values. Transformers retains ownership of clipping and `optimizer.step()`. The standard optimizer changes its denominator after projection, so the implementation and diagnostics are explicitly approximate.

**Tech Stack:** Python 3.11, PyTorch 2.4.1, Transformers 4.45.1, Accelerate 0.34.2, Hydra/OmegaConf, pytest 8.3.5, Torch AdamW, TOFU, Llama-3.2-1B.

---

## File Map

- Create `pytest.ini`: discover `tests/` and add `src` to Python imports.
- Modify `setup.py`: include pytest in the development extra.
- Create `tests/__init__.py`: make shared helpers importable.
- Create `tests/helpers.py`: deterministic nested unlearning batches and a tiny causal LM.
- Create `tests/test_geometric_adam.py`: pure geometry, configuration, runtime, lifecycle, GAS, and baseline-recovery tests.
- Replace `src/trainer/unlearn/geometric.py`: component losses, runtime validation, raw buffers, frozen Adam maps, global projection, diagnostics, and reachable `training_step()`.
- Modify `configs/trainer/GeometricUnlearn.yaml`: approximate-Adam defaults and removal of misleading low-rank/sign options.
- Create `configs/accelerate/gu_single_gpu.yaml`: explicit single-process validation launcher.
- Create `scripts/gu_adam_mvp_eval.sh`: paired FP32 SimNPO control/GU training and TOFU evaluation without touching the existing script.
- Create `GU_ADAM_MVP_REPORT.md`: commands, test evidence, run configuration, new summaries, historical comparison, and non-claims.

## Task 1: Establish the Test Harness

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/helpers.py`
- Create: `tests/test_geometric_adam.py`
- Modify: `setup.py`

- [ ] **Step 1: Create an isolated Python 3.11 environment**

Run:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest==8.3.5 ruff==0.6.9
```

Expected: `.venv/bin/python -c "import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__)"` prints `2.4.1 4.45.1 0.34.2`.

- [ ] **Step 2: Add pytest discovery and a failing helper test**

Create `pytest.ini`:

```ini
[pytest]
pythonpath = src .
testpaths = tests
addopts = -ra
```

Create an empty `tests/__init__.py` and start `tests/test_geometric_adam.py` with:

```python
from tests.helpers import make_unlearn_batch


def test_unlearn_batch_has_forget_and_retain_components():
    batch = make_unlearn_batch(batch_size=2, sequence_length=6)
    assert set(batch) == {"forget", "retain"}
    assert batch["forget"]["labels"].shape == (2, 6)
    assert (batch["retain"]["labels"][:, :2] == -100).all()
```

Run:

```bash
.venv/bin/pytest tests/test_geometric_adam.py::test_unlearn_batch_has_forget_and_retain_components -q
```

Expected: FAIL because `tests.helpers` does not exist.

- [ ] **Step 3: Implement deterministic helpers**

Create `tests/helpers.py`:

```python
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


def make_lm_batch(batch_size=2, sequence_length=6, vocab_size=13, seed=0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def make_unlearn_batch(batch_size=2, sequence_length=6, vocab_size=13, seed=0):
    return {
        "forget": make_lm_batch(batch_size, sequence_length, vocab_size, seed),
        "retain": make_lm_batch(batch_size, sequence_length, vocab_size, seed + 1),
    }


def nested_collator(features):
    return {
        component: {
            key: torch.stack([feature[component][key] for feature in features])
            for key in features[0][component]
        }
        for component in features[0]
    }

class TinyCausalLM(nn.Module):
    main_input_name = "input_ids"

    def __init__(self, vocab_size=13, hidden_size=7):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)
```

Run the focused test again. Expected: PASS.

- [ ] **Step 4: Add pytest to the development extra**

Add `pytest==8.3.5` beside ruff in `setup.py` and run:

```bash
.venv/bin/pytest tests/test_geometric_adam.py -q
```

Expected: the helper test passes.

- [ ] **Step 5: Commit the harness**

```bash
git add pytest.ini setup.py tests/__init__.py tests/helpers.py tests/test_geometric_adam.py
git commit -m "test: establish approximate Adam GU harness"
```

## Task 2: Implement Frozen Adam Square-Root Geometry

**Files:**
- Modify: `tests/test_geometric_adam.py`
- Modify: `src/trainer/unlearn/geometric.py`

- [ ] **Step 1: Write failing pure-geometry tests**

Add tests that call the intended API:

```python
def test_sqrt_adam_coordinate_round_trip():
    gradient = torch.tensor([2.0, -3.0])
    sqrt_h = torch.tensor([0.5, 4.0])
    z = GeometricUnlearn._to_adam_coordinates(gradient, sqrt_h)
    restored = GeometricUnlearn._from_adam_coordinates(z, sqrt_h)
    torch.testing.assert_close(restored, gradient)


def test_global_projection_is_orthogonal_across_tensors():
    gf = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0])}
    gr = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    projected, coefficient = GeometricUnlearn._project_rank_one(gf, gr, 1e-12)
    assert coefficient == pytest.approx(0.5)
    assert abs(GeometricUnlearn._global_dot(projected, gr).item()) < 1e-6


def test_global_projection_differs_from_blockwise_projection():
    gf = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0])}
    gr = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    projected, _ = GeometricUnlearn._project_rank_one(gf, gr, 1e-12)
    assert projected["a"].item() == pytest.approx(1.5)
    assert projected["b"].item() == pytest.approx(-0.5)
```

Run the three tests. Expected: FAIL because the new APIs do not exist.

- [ ] **Step 2: Replace the obsolete projector with minimal global helpers**

Delete `RetainNullProjector`, `optimizer_step()`, `nullProjProjectTan()`, and all Adam full-whitening, low-rank basis, and sign-selective code. Add:

```python
@staticmethod
def _global_dot(left, right):
    result = None
    for name in left.keys() & right.keys():
        value = (left[name].float() * right[name].float()).sum()
        result = value if result is None else result + value
    if result is None:
        raise RuntimeError("No common tensors for global dot product.")
    return result

@staticmethod
def _to_adam_coordinates(gradient, sqrt_h):
    return gradient.float() if sqrt_h is None else gradient.float() / sqrt_h

@staticmethod
def _from_adam_coordinates(vector, sqrt_h):
    return vector.float() if sqrt_h is None else vector.float() * sqrt_h

@classmethod
def _project_rank_one(cls, gf, gr, eps):
    denominator = cls._global_dot(gr, gr).clamp_min(eps)
    coefficient = cls._global_dot(gf, gr) / denominator
    projected = {
        name: tensor.float() - coefficient * gr[name].float()
        if name in gr else tensor.float().clone()
        for name, tensor in gf.items()
    }
    return projected, coefficient
```

Run the focused tests. Expected: PASS.

- [ ] **Step 3: Add frozen-preconditioner safety and zero-retain tests**

Use nonuniform `sqrt_h` tensors, transform raw `gf` and `gr`, project globally, map back, form `delta=-P*g_star`, and assert the relative raw retain directional derivative is below `1e-6`. Also verify a zero retain gradient leaves `gf` unchanged without NaN.

Run the focused tests. Expected: PASS.

- [ ] **Step 4: Commit pure geometry**

```bash
git add src/trainer/unlearn/geometric.py tests/test_geometric_adam.py
git commit -m "fix: add global square-root Adam GU geometry"
```

## Task 3: Separate Component Losses and Resolve Coefficients Once

**Files:**
- Modify: `tests/test_geometric_adam.py`
- Modify: `src/trainer/unlearn/geometric.py`

- [ ] **Step 1: Write failing SimNPO component tests**

Create a helper that instantiates `GeometricUnlearn` with `TinyCausalLM`, CPU `TrainingArguments`, `geometric_config.loss="simnpo"`, and a SimNPO config containing `gamma=0.125`, `alpha=1.0`, `retain_loss_type="NLL"`, `beta=4.5`, and `delta=0.0`.

Add tests that assert:

```python
forget_loss, retain_loss, outputs = trainer.compute_component_losses(model, batch)
total = trainer.compute_loss(model, batch)
torch.testing.assert_close(total, 0.125 * forget_loss + retain_loss)
assert outputs.logits.shape[:2] == batch["forget"]["labels"].shape
assert trainer.gamma == 0.125
assert trainer.alpha == 1.0
assert trainer.retain_loss_type == "NLL"
```

Then mutate `trainer.simnpo_config.gamma` after initialization and assert a second `compute_loss()` does not change `trainer.gamma`.

Run the tests. Expected: FAIL because components are not separated and current `compute_loss()` mutates coefficients.

- [ ] **Step 2: Implement initialization-time resolution and components**

Pop all method configs before `super().__init__()`, resolve the selected config, inject its `gamma`, `alpha`, and `retain_loss_type` into the parent kwargs, and validate `gamma>0` and `alpha>=0`. Preserve the current forget-objective dispatch in a dedicated `compute_forget_loss()` method, with SimNPO as the tested path. Create reference models only for objectives that need them.

Implement:

```python
def compute_component_losses(self, model, inputs):
    forget_loss, forget_outputs = self.compute_forget_loss(model, inputs["forget"])
    retain_loss = self.compute_retain_loss(model, inputs["retain"])
    return forget_loss, retain_loss, forget_outputs

def compute_loss(self, model, inputs, return_outputs=False):
    forget_loss, retain_loss, outputs = self.compute_component_losses(model, inputs)
    loss = self.gamma * forget_loss + self.alpha * retain_loss
    return (loss, outputs) if return_outputs else loss
```

Run the component tests. Expected: PASS.

- [ ] **Step 3: Add invalid coefficient tests**

Assert `gamma=0` and `alpha<0` each raise targeted `ValueError` messages during trainer construction. Run and expect PASS.

- [ ] **Step 4: Commit component separation**

```bash
git add src/trainer/unlearn/geometric.py tests/test_geometric_adam.py
git commit -m "fix: separate GU component gradients"
```

## Task 4: Implement Reachable Accumulation and Gradient Replacement

**Files:**
- Modify: `tests/test_geometric_adam.py`
- Modify: `src/trainer/unlearn/geometric.py`

- [ ] **Step 1: Write failing buffer and lifecycle tests**

Add tests for `_accumulate_component_grads()`, `_clear_gu_buffers()`, and a two-update tiny Trainer run. The run must assert:

```python
assert trainer.gu_projection_calls == trainer.state.global_step == 2
assert trainer._gu_forget_buffer == {}
assert trainer._gu_retain_buffer == {}
assert trainer.last_gu_diagnostics["mode"] == "approximate_adam_stage_a"
```

Run the tests. Expected: FAIL because buffers, `training_step()`, and diagnostics do not exist.

- [ ] **Step 2: Implement raw FP32 buffers**

Initialize the two dictionaries, accumulate detached gradients by selected name in FP32, and clear them after finalization. Do not reconstruct either component from `.grad`.

- [ ] **Step 3: Implement optimizer state access and in-place coordinate transforms**

Add `_unwrap_optimizer()`, parameter-to-group lookup, and `_frozen_sqrt_denominator(param, group)`. For initialized state use:

```python
step = int(state["step"].item())
v_hat = state["exp_avg_sq"].float() / (1.0 - beta2**step)
sqrt_h = (v_hat.sqrt() + eps).sqrt()
```

Return `None` for uninitialized state, meaning identity coordinates. Transform the accumulated buffers only after the full effective batch is available, and recompute `sqrt_h` when mapping the final vector back so no full-model factor copy is retained.

- [ ] **Step 4: Implement `_finalize_gu_gradients()`**

Transform both buffers, compute one global coefficient, combine
`gamma * projected_zf + alpha * zr`, map back, and copy to selected `.grad` with the existing device and dtype. Record mode, coefficient, norms, before/after dots, relative orthogonality residual, and identity fallback count. Increment `gu_projection_calls` exactly once and clear buffers.

- [ ] **Step 5: Implement `training_step()`**

For projection-enabled training, mirror Transformers 4.45.1 setup, compute components, call `autograd.grad()` twice on losses scaled by GAS, accumulate them, call `accelerator.backward(total_loss)`, finalize only when `accelerator.sync_gradients`, and return `total_loss.detach() / GAS`. For `gu_enabled=false`, call `super().training_step(model, inputs)`.

Run the lifecycle tests. Expected: PASS.

- [ ] **Step 6: Add and pass GAS equivalence**

Run identical seeded effective batches through batch 8/GAS 1 and batch 2/GAS 4 trainers with the same frozen optimizer state. Assert final model parameters and recorded projected gradients are close at `rtol=1e-5, atol=1e-6`.

- [ ] **Step 7: Commit reachable training**

```bash
git add src/trainer/unlearn/geometric.py tests/test_geometric_adam.py
git commit -m "fix: run GU on the real Trainer step path"
```

## Task 5: Fail Closed and Recover the SimNPO Baseline

**Files:**
- Modify: `tests/test_geometric_adam.py`
- Modify: `src/trainer/unlearn/geometric.py`

- [ ] **Step 1: Write failing runtime rejection tests**

Parameterize tests for nonzero Adam `beta1`, nonzero weight decay, AMSGrad, SGD, and reentrant checkpointing. Patch `is_deepspeed_enabled`, `is_fsdp_enabled`, `args.world_size`, and `args.fp16` in separate tests. Each call to `_validate_gu_runtime()` must raise a targeted error naming the unsupported mode.

Run the tests. Expected: FAIL because validation is absent.

- [ ] **Step 2: Implement runtime validation**

Unwrap Accelerate only to the standard Torch optimizer. Reject all unsupported modes from the approved design before component forward. Cache successful validation, but do not cache failures. Validate that the selected parameter list is nonempty.

Run the rejection tests. Expected: PASS.

- [ ] **Step 3: Write failing baseline-recovery test**

Initialize identical tiny models and standard AdamW optimizers. Run one step with native `SimNPO` and one with `GeometricUnlearn(gu_enabled=false)` using the same batch. Assert scalar losses and all updated parameters are exactly equal or close at `rtol=0, atol=0` when deterministic CPU kernels permit, otherwise `rtol=1e-7, atol=1e-8`.

Run the test. Expected: FAIL if component semantics differ.

- [ ] **Step 4: Make baseline recovery pass without projection special cases**

Fix only objective or disabled-path discrepancies found by the test. Do not add a second SimNPO formula. Run the full test file. Expected: PASS.

- [ ] **Step 5: Commit fail-closed behavior**

```bash
git add src/trainer/unlearn/geometric.py tests/test_geometric_adam.py
git commit -m "fix: fail closed for approximate Adam GU"
```

## Task 6: Add Safe Configuration and Validation Entry Point

**Files:**
- Modify: `configs/trainer/GeometricUnlearn.yaml`
- Create: `configs/accelerate/gu_single_gpu.yaml`
- Create: `scripts/gu_adam_mvp_eval.sh`
- Modify: `tests/test_geometric_adam.py`

- [ ] **Step 1: Write a failing YAML contract test**

Load `configs/trainer/GeometricUnlearn.yaml` with OmegaConf and assert:

```python
assert cfg.args.optim == "adamw_torch"
assert cfg.args.adam_beta1 == 0.0
assert cfg.args.weight_decay == 0.0
assert cfg.args.fp16 is False
assert cfg.method_args.geometric_config.gu_enabled is True
assert cfg.method_args.geometric_config.sign_selective is False
assert cfg.method_args.simnpo_config.gamma == 0.125
assert cfg.method_args.simnpo_config.retain_loss_type == "NLL"
```

Run the test. Expected: FAIL against the current PagedAdamW/default config.

- [ ] **Step 2: Update the trainer config**

Set Torch AdamW, `adam_beta1: 0.0`, `weight_decay: 0.0`, `fp16: false`, explicit non-reentrant checkpointing, `gu_enabled: true`, `projection_eps: 1e-12`, `sign_selective: false`, and `trainable_params_regex: [".*"]`. Remove obsolete basis, whitening, and tangent options. Keep SimNPO `gamma=0.125`, `alpha=1.0`, and NLL retain loss.

Run the YAML contract test. Expected: PASS.

- [ ] **Step 3: Add the single-process Accelerate config**

Create `configs/accelerate/gu_single_gpu.yaml` with `distributed_type: NO`, `num_processes: 1`, `mixed_precision: no`, and no DeepSpeed or FSDP plugin.

- [ ] **Step 4: Add the paired validation script**

Create `scripts/gu_adam_mvp_eval.sh` with `set -euo pipefail`. It must use only SimNPO/TOFU forget01, GPU 0, FP32 model loading, batch 4, GAS 8, `max_steps=10`, seed 0, Torch AdamW, `adam_beta1=0`, weight decay 0, and distinct task names for disabled control and enabled GU. It trains each arm from the same base model and evaluates each into a new timestamped `saves/exp/GU_ADAM_MVP/` directory. It must not invoke or edit `scripts/gu_eval.sh`.

- [ ] **Step 5: Shell-parse and commit configuration**

Run:

```bash
bash -n scripts/gu_adam_mvp_eval.sh
.venv/bin/pytest tests/test_geometric_adam.py -q
```

Expected: shell parse succeeds and tests pass.

Commit:

```bash
git add configs/trainer/GeometricUnlearn.yaml configs/accelerate/gu_single_gpu.yaml scripts/gu_adam_mvp_eval.sh tests/test_geometric_adam.py
git commit -m "feat: add approximate Adam GU validation config"
```

## Task 7: Verify Code Before the 1B Run

**Files:**
- No planned modifications. If a check fails, return to the task that owns the
  failing behavior and repeat its red-green cycle.

- [ ] **Step 1: Run the focused tests**

```bash
.venv/bin/pytest tests/test_geometric_adam.py -vv
```

Expected: all tests pass with no warnings caused by GU.

- [ ] **Step 2: Run repository quality checks on touched Python files**

```bash
.venv/bin/ruff check src/trainer/unlearn/geometric.py tests/helpers.py tests/test_geometric_adam.py setup.py
.venv/bin/ruff format --check src/trainer/unlearn/geometric.py tests/helpers.py tests/test_geometric_adam.py setup.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Inspect the diff and status**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the user's pre-existing unrelated edits may remain uncommitted.

## Task 8: Run Paired SimNPO Validation and Write the Report

**Files:**
- Create: `GU_ADAM_MVP_REPORT.md`

- [ ] **Step 1: Run the paired training and full TOFU evaluation**

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/gu_adam_mvp_eval.sh
```

Expected: two ten-step models and two new `TOFU_SUMMARY.json` files are produced without overwriting existing paths.

- [ ] **Step 2: Verify lifecycle evidence**

Read the GU training log and `trainer_state.json`. Confirm global step 10, ten projection diagnostics, mode `approximate_adam_stage_a`, finite coefficients/residuals, and no unsupported-mode fallback beyond the documented first-step identity metric.

- [ ] **Step 3: Compare summaries**

Generate a table containing the new control, new GU, historical baseline, and historical ineffective GU metrics. Explicitly record the optimizer, precision, weight decay, SimNPO gamma, and whether projection executed for each arm.

- [ ] **Step 4: Write the report**

Create `GU_ADAM_MVP_REPORT.md` containing exact commands, commit IDs, test output, output paths, lifecycle evidence, metric table, frozen-metric claims, observed realized-Adam limitations, and the deferred custom-optimizer work. Do not claim strict realized-delta safety.

- [ ] **Step 5: Final verification and report commit**

```bash
.venv/bin/pytest tests/test_geometric_adam.py -q
git diff --check
```

Expected: tests pass and no whitespace errors.

Commit only the report and any final test-backed corrections:

```bash
git add GU_ADAM_MVP_REPORT.md
git commit -m "docs: report approximate Adam GU validation"
```
