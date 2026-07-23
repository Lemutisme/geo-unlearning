# Minimal Correct GU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one objective-agnostic GU branch to `UnlearnTrainer` that constrains the realized AdamW parameter delta with current and historical retain covectors.

**Architecture:** Existing objective trainers continue to own their losses. The common trainer accumulates retain-NLL covectors during `training_step`, snapshots selected parameters before ordinary AdamW, and corrects the realized post-step delta with a bounded active-set SafeProx solve. Optional finite-step mode verifies actual retain NLL by deterministic backtracking.

**Tech Stack:** Python 3.11, PyTorch optimizer step hooks, Transformers 4.45.1 Trainer, Accelerate, Hydra/OmegaConf, pytest, BF16/FP32, Torch AdamW and shipped 32-bit paged AdamW.

---

## File Map

- Modify `src/trainer/unlearn/base.py`: the sole authoritative GU runtime branch.
- Modify `src/trainer/unlearn/geometric.py`: reject the retired legacy GU mode while retaining PCGrad.
- Modify `configs/trainer/GeometricUnlearn.yaml`: make the legacy trainer explicitly PCGrad-only.
- Modify `scripts/gu_adam_mvp_eval.sh`: use an existing objective trainer plus `method_args.gu`.
- Modify `scripts/gu_eval.sh`: map each supported loss to its existing objective trainer and enable GU there.
- Modify `scripts/pcgrad_smoke_arm.sh`: route only the GU comparison arm through the new common GU branch.
- Create `tests/test_minimal_gu.py`: all unit, lifecycle, traceability, and BF16 integration coverage.

No runtime, manager, projector, solver, adapter, mixin, callback, or new trainer class is created.

### Task 1: Configuration Seam and Exact Disabled Baseline

**Files:**
- Modify: `src/trainer/unlearn/base.py:1-184`
- Create: `tests/test_minimal_gu.py`

- [ ] **Step 1: Add the tiny model and trainer fixtures**

Add these concrete fixtures to `tests/test_minimal_gu.py`:

```python
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from transformers import TrainingArguments

from trainer.unlearn.base import UnlearnTrainer


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.protected = nn.Linear(2, 4, bias=False)
        self.frozen = nn.Linear(2, 4, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden = self.embedding(input_ids)
        logits = self.protected(hidden) + self.frozen(hidden)
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


class TinyObjectiveTrainer(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs["forget"])
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def gu_config(**overrides):
    values = {
        "enabled": True,
        "parameter_regex": ["protected[.]weight"],
        "retain_history_rank": 8,
        "projection_eps": 1.0e-6,
        "retain_filter": "first_order",
        "retain_budget": 1.0e-4,
        "backtracking_scales": [1.0, 0.5, 0.25, 0.125],
        "diagnostics_path": "gu_diagnostics.jsonl",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def tiny_batch():
    forget = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2, 3]]),
    }
    retain = {
        "input_ids": torch.tensor([[3, 2, 1, 0]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 1, 0]]),
    }
    return {"forget": forget, "retain": retain}


def make_tiny_trainer(
    model,
    output_dir,
    *,
    gu,
    gradient_accumulation_steps=1,
    adam_beta1=0.9,
    weight_decay=0.0,
):
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.05,
        optim="adamw_torch",
        adam_beta1=adam_beta1,
        adam_beta2=0.9,
        weight_decay=weight_decay,
        report_to=[],
        save_strategy="no",
    )
    trainer = TinyObjectiveTrainer(model=model, args=args, gu=gu)
    trainer.create_optimizer()
    return trainer


def run_one_trainer_update(trainer, batch=None):
    batch = tiny_batch() if batch is None else batch
    trainer.model.train()
    trainer.optimizer.zero_grad()
    loss = trainer.training_step(trainer.model, batch)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    return loss


def assert_optimizer_states_equal(left, right):
    left_parameters = [parameter for group in left.param_groups for parameter in group["params"]]
    right_parameters = [parameter for group in right.param_groups for parameter in group["params"]]
    assert len(left_parameters) == len(right_parameters)
    for left_parameter, right_parameter in zip(left_parameters, right_parameters):
        left_state = left.state[left_parameter]
        right_state = right.state[right_parameter]
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            if torch.is_tensor(left_state[key]):
                torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)
            else:
                assert left_state[key] == right_state[key]
```

- [ ] **Step 2: Write the disabled-baseline regression**

Create two identical tiny trainers with Torch AdamW. Omit `gu` from one and pass
`gu_config(enabled=False)` to the other. Run one identical optimizer update and
assert identical losses, parameters, and optimizer state dictionaries:

```python
def test_disabled_gu_matches_unmodified_objective_update(tmp_path):
    torch.manual_seed(0)
    control_model = TinyLM()
    disabled_model = deepcopy(control_model)
    control = make_tiny_trainer(control_model, tmp_path / "control", gu=None)
    disabled = make_tiny_trainer(
        disabled_model,
        tmp_path / "disabled",
        gu=gu_config(enabled=False),
    )

    run_one_trainer_update(control)
    run_one_trainer_update(disabled)

    for left, right in zip(control_model.parameters(), disabled_model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert_optimizer_states_equal(control.optimizer, disabled.optimizer)
```

These test functions exercise real `TrainingArguments`, `training_step`, and
optimizer hooks. They are test infrastructure and are not copied into production.

- [ ] **Step 3: Verify RED**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
pytest tests/test_minimal_gu.py::test_disabled_gu_matches_unmodified_objective_update -q
```

Expected: `TinyObjectiveTrainer` rejects the unexpected `gu` keyword.

- [ ] **Step 4: Add the optional configuration seam**

Add an `__init__` to `UnlearnTrainer` with this exact disabled behavior:

```python
def __init__(self, *args, gu=None, **kwargs):
    self.gu_config = gu
    self.gu_enabled = bool(gu is not None and gu.get("enabled", False))
    self.gu_projection_calls = 0
    self.gu_last_diagnostics = None
    super().__init__(*args, **kwargs)
```

Do not normalize, allocate, freeze, register hooks, or open diagnostics when GU
is disabled.

- [ ] **Step 5: Verify GREEN and baseline stability**

Run:

```bash
pytest tests/test_minimal_gu.py::test_disabled_gu_matches_unmodified_objective_update -q
pytest -q
```

Expected: the focused test passes and the repository remains at 111 passing
baseline tests plus the new test.

- [ ] **Step 6: Commit**

```bash
git add src/trainer/unlearn/base.py tests/test_minimal_gu.py
git commit -m "feat: add objective-agnostic GU seam"
```

### Task 2: Parameter Scope and Runtime Validation

**Files:**
- Modify: `src/trainer/unlearn/base.py`
- Test: `tests/test_minimal_gu.py`

- [ ] **Step 1: Write exact validation tests**

Parameterize invalid configurations and assert the exact failure category:

```python
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"parameter_regex": []}, "parameter_regex"),
        ({"retain_history_rank": -1}, "retain_history_rank"),
        ({"retain_history_rank": 9}, "retain_history_rank"),
        ({"projection_eps": 0.0}, "projection_eps"),
        ({"retain_filter": "quadratic"}, "retain_filter"),
        ({"retain_budget": -1.0}, "retain_budget"),
        ({"backtracking_scales": [0.5, 1.0]}, "backtracking_scales"),
    ],
)
def test_gu_rejects_invalid_configuration(tmp_path, override, message):
    trainer = make_tiny_trainer(TinyLM(), tmp_path, gu=gu_config(**override))
    with pytest.raises((TypeError, ValueError), match=message):
        trainer.create_optimizer()
```

Add separate tests for no regex match, a trainable parameter outside the
selection after setup, FP16, world size greater than one, reentrant gradient
checkpointing, DeepSpeed, FSDP, and an optimizer without finite `exp_avg_sq`.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'invalid_configuration or rejects_' -q
```

Expected: the invalid settings are currently accepted.

- [ ] **Step 3: Implement validation before optimizer creation**

When GU is enabled, normalize only the approved keys and reject extra keys:

```python
required = {
    "enabled",
    "parameter_regex",
    "retain_history_rank",
    "projection_eps",
    "retain_filter",
    "retain_budget",
    "backtracking_scales",
    "diagnostics_path",
}
if set(self.gu_config) != required:
    raise ValueError("GU configuration keys do not match the public contract")
```

Compile each regex once, freeze every unmatched model parameter before calling
the parent optimizer creation, and store the selected `(name, parameter)` pairs
in model order. After creation, require every optimizer parameter to be selected
and every selected parameter to occur exactly once in its groups.

Accept only Torch AdamW or the shipped 32-bit paged AdamW. Require finite
floating-point `exp_avg_sq` for Torch or `state2` for the paged optimizer.
Reject eight-bit state. Validate the single-process BF16/FP32 runtime before
the first training step.

Resolve `diagnostics_path` under `args.output_dir`, reject path traversal and
symlinks, create its parent, and prove it is writable before training begins.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'invalid_configuration or rejects_ or parameter_scope' -q
```

Expected: all validation and parameter-freezing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/unlearn/base.py tests/test_minimal_gu.py
git commit -m "feat: validate minimal GU runtime"
```

### Task 3: Effective-Batch Retain Covectors and FIFO History

**Files:**
- Modify: `src/trainer/unlearn/base.py`
- Test: `tests/test_minimal_gu.py`

- [ ] **Step 1: Write retain-covector and GAS tests**

Use answer-masked retain inputs and compare one effective batch represented as
GAS 1 and GAS 4:

```python
def collect_effective_retain_covector(trainer, microbatches):
    trainer.optimizer.zero_grad()
    for _ in range(microbatches):
        with trainer.accelerator.accumulate(trainer.model):
            trainer.training_step(trainer.model, tiny_batch())
    return tuple(tensor.detach().clone() for tensor in trainer._gu_current_covector)


def test_gu_retain_covector_matches_equivalent_effective_batch(tmp_path):
    full = make_tiny_trainer(TinyLM(), tmp_path / "full", gu=gu_config())
    accumulated = make_tiny_trainer(
        deepcopy(full.model),
        tmp_path / "accumulated",
        gu=gu_config(),
        gradient_accumulation_steps=4,
    )
    full_covector = collect_effective_retain_covector(full, microbatches=1)
    gas_covector = collect_effective_retain_covector(accumulated, microbatches=4)
    for left, right in zip(full_covector, gas_covector):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
```

Add tests that the current covector is always first, FIFO history keeps exactly
the configured previous directions, rank 0 keeps only the current direction,
global normalization has unit norm, and no parameter block is normalized
independently.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'retain_covector or history' -q
```

Expected: `UnlearnTrainer` has no GU effective-batch state.

- [ ] **Step 3: Implement the explicit training-step branch**

Override `training_step`. When GU is disabled, return the parent result
immediately. When enabled:

1. prepare the input once;
2. require `inputs["retain"]`;
3. compute answer-masked retain NLL with `compute_batch_nll`;
4. call `torch.autograd.grad` only for selected parameters;
5. divide by `gradient_accumulation_steps` and accumulate detached FP32 tensors;
6. call the ordinary parent `training_step` for the objective update;
7. finalize the current covector only when `accelerator.sync_gradients` is true.

Store constraints as ordered lists aligned with selected parameters. Normalize
with one global FP64 sum of FP32 block norms. Preserve the unnormalized current
covector until diagnostics are complete, then append its normalized copy to the
FIFO after the optimizer step.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'retain_covector or history or effective_batch' -q
```

Expected: all effective-batch and FIFO tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/unlearn/base.py tests/test_minimal_gu.py
git commit -m "feat: collect GU retain constraints"
```

### Task 4: Realized AdamW Proposal Capture

**Files:**
- Modify: `src/trainer/unlearn/base.py`
- Test: `tests/test_minimal_gu.py`

- [ ] **Step 1: Write actual-delta tests**

Run two AdamW updates so the second proposal contains momentum and nonzero
second moment. Use nonzero weight decay and assert the captured proposal equals
the actual parameter difference, not `-lr * grad`:

```python
def test_gu_captures_full_realized_adamw_delta(tmp_path):
    trainer = make_tiny_trainer(
        TinyLM(),
        tmp_path,
        gu=gu_config(),
        adam_beta1=0.9,
        weight_decay=0.1,
    )
    run_one_trainer_update(trainer)
    before = [parameter.detach().clone() for _, parameter in trainer._gu_selected]
    raw_gradient = [parameter.grad.detach().clone() for _, parameter in trainer._gu_selected]
    run_one_trainer_update(trainer)
    observed = [parameter.detach() - old for (_, parameter), old in zip(trainer._gu_selected, before)]
    for captured, actual in zip(trainer._gu_proposal_delta, observed):
        torch.testing.assert_close(captured, actual)
    assert any(not torch.allclose(delta, -trainer.args.learning_rate * grad) for delta, grad in zip(observed, raw_gradient))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_minimal_gu.py::test_gu_captures_full_realized_adamw_delta -q
```

Expected: no optimizer hooks or proposal delta exist.

- [ ] **Step 3: Register direct optimizer hooks**

After ordinary optimizer creation, register bound optimizer pre/post hook entry
points. The pre-hook clones selected parameters in FP32 and requires one ready
effective-batch constraint set. The post-hook computes
`parameter_after.float() - parameter_before` before performing any correction.

Read `exp_avg_sq` for Torch or `state2` for the shipped paged optimizer after
AdamW updates it. Store its square root plus group epsilon as the frozen
diagonal metric for this proposal. Check all tensors for finiteness and exact
shape.

Do not register a Trainer callback, replace the optimizer, or patch its `step`
attribute.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'realized_adamw or optimizer_state' -q
```

Expected: actual delta and optimizer-state checks pass.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/unlearn/base.py tests/test_minimal_gu.py
git commit -m "feat: capture realized AdamW proposals"
```

### Task 5: Bounded Active-Set SafeProx

**Files:**
- Modify: `src/trainer/unlearn/base.py`
- Test: `tests/test_minimal_gu.py`

- [ ] **Step 1: Write actual-update safety tests**

Construct one current and two historical covectors with multiple active
constraints. After a real optimizer update, assert all applied directional
derivatives are nonpositive and KKT diagnostics are bounded:

```python
def install_known_constraints(trainer):
    template = [torch.zeros_like(parameter, dtype=torch.float32) for _, parameter in trainer._gu_selected]
    current = [tensor.clone() for tensor in template]
    first_history = [tensor.clone() for tensor in template]
    second_history = [tensor.clone() for tensor in template]
    current[0][0, 0] = 1.0
    first_history[0][0, 1] = 1.0
    second_history[0][1, 0] = 1.0
    trainer._gu_current_covector = tuple(current)
    trainer._gu_history = [tuple(first_history), tuple(second_history)]


def test_safeprox_constrains_actual_parameter_delta(tmp_path):
    trainer = make_tiny_trainer(TinyLM(), tmp_path, gu=gu_config(retain_history_rank=2))
    install_known_constraints(trainer)
    before = [parameter.detach().clone() for _, parameter in trainer._gu_selected]
    run_one_trainer_update(trainer)
    applied = [parameter.detach().float() - old.float() for (_, parameter), old in zip(trainer._gu_selected, before)]
    for constraint in trainer._gu_constraints_used:
        derivative = sum((block * delta).sum() for block, delta in zip(constraint, applied))
        assert derivative.item() <= 1.0e-7
    assert trainer.gu_last_diagnostics["kkt_residual"] <= 1.0e-8
```

Add cases for an already-safe proposal, dependent constraints, all-zero retain
direction, singular active subsets, and no feasible finite solution due to
nonfinite state. The safe case must be bitwise unchanged.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'safeprox or kkt or dependent_constraints' -q
```

Expected: harmful realized deltas remain uncorrected.

- [ ] **Step 3: Implement the bounded solve in the post-hook**

For at most nine constraints:

1. compute proposal violations `G.T @ delta0` globally across blocks;
2. return the proposal unchanged when every violation is within tolerance;
3. build the FP64 Gram matrix `K = G.T @ M^-1 @ G`;
4. enumerate masks from `1` to `2**constraint_count - 1`;
5. solve each active system with `torch.linalg.lstsq`;
6. reject negative multipliers or excessive active residuals;
7. reconstruct the candidate delta block by block;
8. check every primal constraint and compute the metric correction cost;
9. retain the feasible candidate with minimum cost;
10. verify primal, dual, complementarity, and KKT residuals before writing
    `parameter_before + corrected_delta`.

Use `projection_eps` only as the explicit numerical tolerance and least-squares
rank threshold. Do not add retries, adaptive margins, or another solver path.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'safeprox or kkt or dependent_constraints or safe_proposal' -q
```

Expected: all actual-delta and KKT tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/unlearn/base.py tests/test_minimal_gu.py
git commit -m "feat: constrain realized AdamW deltas"
```

### Task 6: Finite-Step Filter and One-Record Diagnostics

**Files:**
- Modify: `src/trainer/unlearn/base.py`
- Test: `tests/test_minimal_gu.py`

- [ ] **Step 1: Write finite-step and diagnostics tests**

Use a convex retain loss where scale 1 violates the budget and scale 0.5 is the
largest feasible configured value:

```python
def test_finite_step_uses_largest_actual_safe_scale(tmp_path):
    trainer = make_tiny_trainer(
        TinyLM(),
        tmp_path,
        gu=gu_config(
            retain_filter="finite_step",
            retain_budget=0.01,
            backtracking_scales=[1.0, 0.5, 0.25],
        ),
    )
    run_one_trainer_update(trainer)
    assert trainer.gu_last_diagnostics["applied_scale"] == 0.5
    assert trainer.gu_last_diagnostics["retain_loss_after"] - trainer.gu_last_diagnostics["retain_loss_before"] <= 0.01
```

Add a no-feasible-scale case that restores the selected parameters exactly and
records `zero_step=true`. Add a first-order case proving that no retain-loss
forward is performed. For the largest-scale test, set the initial tiny-model
weights and `retain_budget` so direct no-grad evaluation gives drift above the
budget at scale 1, within budget at scale 0.5, and within budget at scale 0.25;
assert these three measured preconditions before calling the post-hook. Parse
the diagnostics JSONL and require exactly one record per optimizer update with
the exact public fields from the design. Do not add a production testing seam.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'finite_step or diagnostics_record or zero_step' -q
```

Expected: no finite-step filtering or JSONL contract exists.

- [ ] **Step 3: Implement deterministic actual-loss filtering**

In the post-hook, after first-order correction:

- preserve the model's training/eval state;
- evaluate answer-masked retain NLL at the pre-step selected parameters in eval
  mode;
- evaluate configured scales in their validated descending order;
- commit the first scale within `retain_budget`;
- restore the pre-step selected parameters if none is feasible;
- restore the model mode;
- leave AdamW proposal state unchanged and record
  `optimizer_state_semantics="proposal_state_committed"`.

Write one compact JSON line with `allow_nan=False`, flush it, increment
`gu_projection_calls`, clear the effective-batch covector buffer, and append the
current normalized covector to FIFO history.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'finite_step or diagnostics_record or zero_step or first_order' -q
```

Expected: filter, zero-step, and one-record diagnostics tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/unlearn/base.py tests/test_minimal_gu.py
git commit -m "feat: verify finite-step retain safety"
```

### Task 7: Retire the Ambiguous Legacy GU Entry

**Files:**
- Modify: `src/trainer/unlearn/geometric.py`
- Modify: `configs/trainer/GeometricUnlearn.yaml`
- Modify: `scripts/gu_adam_mvp_eval.sh`
- Modify: `scripts/gu_eval.sh`
- Modify: `scripts/pcgrad_smoke_arm.sh`
- Test: `tests/test_minimal_gu.py`

- [ ] **Step 1: Write CLI and migration tests**

Test that legacy GU mode raises a direct migration error, legacy PCGrad remains
reachable, and every shipped GU command resolves to an existing objective
trainer with `trainer.method_args.gu.enabled=true`:

```python
from tests.test_geometric_adam import make_geometric_trainer


def test_legacy_gu_mode_has_one_migration_target(tmp_path):
    with pytest.raises(ValueError, match="trainer.method_args.gu.enabled=true"):
        make_geometric_trainer(
            tmp_path / "legacy-gu",
            geometric_overrides={"gradient_surgery": "gu"},
        )
    trainer, _, _ = make_geometric_trainer(
        tmp_path / "legacy-pcgrad",
        geometric_overrides={"gradient_surgery": "pcgrad"},
    )
    assert trainer.gradient_surgery == "pcgrad"
```

Inspect shell command arrays in tests rather than launching training. Require
SimNPO, NPO, DPO, UNDIAL, CEU, WGA, and SatImp commands to select their existing
trainer config and use the common GU keys.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'legacy_gu or shipped_gu_command' -q
```

Expected: shipped commands still select `GeometricUnlearn` GU mode.

- [ ] **Step 3: Implement the migration**

In `GeometricUnlearn.__init__`, reject only enabled `gradient_surgery == "gu"`
with this message:

```text
Legacy GU is retired; select the objective trainer and set trainer.method_args.gu.enabled=true.
```

Keep PCGrad and disabled control behavior unchanged. Set the legacy YAML default
to PCGrad explicitly.

In GU scripts, map loss names to existing configs with a Bash `case` and pass
the exact public GU mapping. In `pcgrad_smoke_arm.sh`, keep control and PCGrad on
the legacy trainer, but route the GU comparison arm to `trainer=SimNPO` plus the
new `gu` mapping. Preserve all existing attention, optimizer, buffer, output,
and diagnostics knobs that remain applicable.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/test_minimal_gu.py -k 'legacy_gu or shipped_gu_command or cli_trace' -q
bash -n scripts/gu_adam_mvp_eval.sh scripts/gu_eval.sh scripts/pcgrad_smoke_arm.sh
```

Expected: migration and shell syntax tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/unlearn/geometric.py configs/trainer/GeometricUnlearn.yaml scripts/gu_adam_mvp_eval.sh scripts/gu_eval.sh scripts/pcgrad_smoke_arm.sh tests/test_minimal_gu.py
git commit -m "refactor: make common GU path authoritative"
```

### Task 8: Integration, Artifact, and Simplicity Gates

**Files:**
- Modify: `tests/test_minimal_gu.py`
- Modify: `README.md`

- [ ] **Step 1: Add the complete BF16 integration test**

On one CUDA device, instantiate a tiny BF16 causal model and a real shipped
objective trainer with Torch AdamW. Run GAS 2 for two optimizer updates and
assert:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_trainer_path_applies_one_safe_delta_per_update(tmp_path):
    from tests.helpers import TinyCausalLM, make_unlearn_batch
    from tests.test_geometric_adam import unbatch
    from trainer.unlearn.simnpo import SimNPO

    model = TinyCausalLM().to(device="cuda", dtype=torch.bfloat16)
    dataset = unbatch(make_unlearn_batch(batch_size=4, sequence_length=6, seed=17))
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=2,
        learning_rate=1.0e-3,
        optim="adamw_torch",
        bf16=True,
        report_to=[],
        save_strategy="no",
        remove_unused_columns=False,
    )
    trainer = SimNPO(
        model=model,
        args=args,
        train_dataset=dataset,
        beta=4.5,
        delta=0.0,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type="NLL",
        gu=gu_config(parameter_regex=["lm_head[.]weight"]),
    )
    trainer.create_optimizer()
    before = [parameter.detach().clone() for _, parameter in trainer._gu_selected]
    trainer.train()
    after = [parameter.detach().clone() for _, parameter in trainer._gu_selected]
    assert trainer.gu_projection_calls == trainer.state.global_step == 2
    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    records = [json.loads(line) for line in (tmp_path / "gu_diagnostics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(record["max_violation_after"] <= 1.0e-6 for record in records)
```

Audit the output directory and fail on model weights, optimizer state,
checkpoints, `trainer_state.json`, or `training_args.bin`.

Add a CUDA test guarded by `pytest.importorskip("bitsandbytes")` that runs one
real `bitsandbytes.optim.AdamW(..., optim_bits=32, is_paged=True)` update through
the same GU branch and asserts finite `state2`, one corrected realized delta,
and one diagnostic record. Do not add an optimizer adapter.

- [ ] **Step 2: Add objective-agnostic coverage**

Inspect `TRAINER_REGISTRY` and require every registered subclass of
`UnlearnTrainer` to inherit the common GU branch. Instantiate representative
GradAscent, NPO, RMU, and PDU trainers with GU disabled to prove no objective
dispatch was copied into the GU implementation.

- [ ] **Step 3: Document the one supported entry**

Replace README examples that use legacy GU mode with one existing objective
trainer plus the exact `method_args.gu` mapping. State the supported optimizer,
single-process precision boundary, first-order default, optional finite-step
filter, and diagnostic file.

- [ ] **Step 4: Run complete verification**

Run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
pytest tests/test_minimal_gu.py -q
pytest -q
ruff check src/trainer/unlearn/base.py src/trainer/unlearn/geometric.py tests/test_minimal_gu.py
git diff --check dev0...HEAD
```

Expected: focused tests pass, the complete suite exceeds the 111-test baseline
without regressions, Ruff passes, and the diff is clean.

- [ ] **Step 5: Enforce the production line budget**

Run:

```bash
added=$(git diff --numstat dev0...HEAD -- src/trainer/unlearn/base.py src/trainer/unlearn/geometric.py | awk '{sum += $1} END {print sum + 0}')
test "$added" -le 600
test "$(git diff --name-only dev0...HEAD -- src/trainer/unlearn | grep -Ec 'runtime|manager|projector|solver|adapter|callback')" -eq 0
```

Expected: no more than 600 added production lines and no forbidden abstraction
file is introduced.

- [ ] **Step 6: Commit**

```bash
git add tests/test_minimal_gu.py README.md
git commit -m "test: verify minimal correct GU end to end"
```

## Final Review Checklist

- [ ] Every code change followed a witnessed RED then GREEN cycle.
- [ ] `gu.enabled=false` is exactly baseline-compatible.
- [ ] Every applied parameter delta, not an intermediate tensor, is checked.
- [ ] Current retain covector is never omitted when history is full.
- [ ] First-order and finite-step claims are labeled separately.
- [ ] Optimizer proposal state semantics are visible in diagnostics.
- [ ] No objective-specific loss was added to common GU code.
- [ ] No shipped feature or applicable observability knob was silently disabled.
- [ ] Comments contain concise external-facing reasons only.
- [ ] Production additions are at most 600 lines.
