# Correct Geometric Unlearning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate a mathematically correct Stage-A GU projector, then extend it with a Stage-B realized-AdamW-delta safety correction, without saving model checkpoints.

**Architecture:** Objective formulas live in a shared component registry; pure Torch geometry lives in a Trainer-independent projector; runtime code owns accumulation buffers, optimizer-state adapters, and raw optimizer hooks. Stage A overwrites the current optimizer-equivalent raw gradient before the optimizer step. Stage B observes and corrects the realized parameter delta after momentum and weight decay. DeepSpeed/FSDP fail fast in this implementation.

**Tech Stack:** Python 3.11, PyTorch 2.4.1, Transformers 4.45.1, Accelerate 0.34.2, bitsandbytes 0.44.1, Hydra/OmegaConf, pytest, native Accelerate DDP, CUDA BF16/FP16.

---

## File Map

- Create `pytest.ini`: test discovery and `src` import path.
- Modify `setup.py`: add pytest to the development extra.
- Create `tests/helpers.py`: deterministic toy causal LM, batches, and optimizer fixtures.
- Create `tests/test_objectives.py`: scalar/gradient parity and schema tests for eight objectives.
- Create `src/trainer/unlearn/objectives.py`: shared `ObjectiveParts` registry.
- Modify `src/trainer/utils.py`: host shared CEU and forward-KL helpers.
- Modify eight files under `src/trainer/unlearn/`: delegate baseline losses to the registry.
- Create `tests/test_geometric_projector.py`: metric, basis, paper, cone, and Stage-B property tests.
- Create `src/trainer/unlearn/geometric_projector.py`: pure mathematical implementation.
- Create `tests/test_geometric_runtime.py`: accumulation, optimizer adapter, and hook tests.
- Create `src/trainer/unlearn/geometric_runtime.py`: runtime buffers and optimizer hooks.
- Replace `src/trainer/unlearn/geometric.py`: validated GU orchestration.
- Modify `configs/trainer/GeometricUnlearn.yaml`: explicit objective and geometry configuration.
- Create `configs/accelerate/ddp.yaml`: supported native DDP configuration.
- Create `tests/test_geometric_trainer.py`: Trainer-level hook and parity tests.
- Modify `.github/workflows/tests.yml` and `Makefile`: enable tests.
- Create `scripts/quick_validate_geometric.py`: checkpoint-free 1B diagnostic runner.
- Create `tests/test_quick_validate_no_checkpoint.py`: enforce the no-checkpoint contract.
- Modify `.gitignore`: ignore generated quick-validation outputs while retaining `.gitkeep`.
- Create `results/geometric_quick/.gitkeep`: tracked result directory only; generated JSON remains untracked.
- Create `GEOMETRIC_IMPLEMENTATION_REPORT.md`: verified implementation and experiment analysis.

## Task 1: Establish the Test Harness

**Files:**
- Create: `pytest.ini`
- Create: `tests/helpers.py`
- Create: `tests/test_test_harness.py`
- Modify: `setup.py:20-28`

- [ ] **Step 1: Create a Python 3.11 test environment**

Run:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  pytest==8.3.5 torch==2.4.1 packaging==24.2 ruff==0.6.9
```

Expected: `.venv/bin/python --version` reports Python 3.11 and importing Torch succeeds. Do not install the full training stack until the integration-test task.

- [ ] **Step 2: Add test configuration and a failing helper test**

Create `pytest.ini`:

```ini
[pytest]
pythonpath = src .
testpaths = tests
addopts = -ra
```

Create `tests/test_test_harness.py`:

```python
from tests.helpers import make_lm_batch


def test_make_lm_batch_masks_prompt_tokens():
    batch = make_lm_batch(batch_size=2, sequence_length=5, vocab_size=11)
    assert batch["labels"].shape == (2, 5)
    assert (batch["labels"][:, :2] == -100).all()
```

Run:

```bash
.venv/bin/pytest tests/test_test_harness.py -q
```

Expected: FAIL because `tests.helpers` does not exist.

- [ ] **Step 3: Implement deterministic test helpers**

Create `tests/__init__.py` as an empty file and `tests/helpers.py` with:

```python
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def make_lm_batch(batch_size=2, sequence_length=5, vocab_size=11, seed=0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, sequence_length), generator=generator
    )
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=11, hidden_size=7):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].transpose(1, 2),
                labels[:, 1:],
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)
```

- [ ] **Step 4: Verify the harness is green**

Run:

```bash
.venv/bin/pytest tests/test_test_harness.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Add pytest to the development extra**

Add `pytest==8.3.5` beside Ruff in `setup.py` without changing runtime dependencies.

- [ ] **Step 6: Commit the harness**

```bash
git add pytest.ini setup.py tests/__init__.py tests/helpers.py tests/test_test_harness.py
git commit -m "test: establish geometric unlearning harness"
```

## Task 2: Create One Source of Truth for Objective Semantics

**Files:**
- Create: `src/trainer/unlearn/objectives.py`
- Create: `tests/test_objectives.py`
- Modify: `src/trainer/utils.py`
- Modify: `src/trainer/unlearn/grad_diff.py`
- Modify: `src/trainer/unlearn/ceu.py`
- Modify: `src/trainer/unlearn/dpo.py`
- Modify: `src/trainer/unlearn/npo.py`
- Modify: `src/trainer/unlearn/simnpo.py`
- Modify: `src/trainer/unlearn/undial.py`
- Modify: `src/trainer/unlearn/satimp.py`
- Modify: `src/trainer/unlearn/wga.py`

- [ ] **Step 1: Write failing registry tests**

Add tests that construct identical toy models and assert the registry returns exact coefficients:

```python
import pytest
import torch

from tests.helpers import TinyCausalLM, make_lm_batch
from trainer.unlearn.objectives import ObjectiveConfig, compute_objective_parts


@pytest.mark.parametrize(
    ("name", "gamma", "alpha"),
    [
        ("graddiff", 1.0, 1.0),
        ("ceu", 1.0, 0.0),
        ("dpo", 1.0, 1.0),
        ("npo", 1.0, 1.0),
        ("satimp", 0.1, 1.0),
        ("simnpo", 0.125, 1.0),
        ("undial", 1.0, 0.0),
        ("wga", 1.0, 1.0),
    ],
)
def test_default_objective_coefficients(name, gamma, alpha):
    config = ObjectiveConfig.for_name(name)
    assert config.gamma == gamma
    assert config.alpha == alpha
```

Also add:

- scalar and gradient parity against the pre-refactor class behavior for all eight methods;
- DPO nested-input validation;
- CEU forget-only objective with no objective-retain term;
- UNDIAL `alpha=0` while still requiring its reference model;
- a forward-KL test that distinguishes `KL(p_theta || p_ref)` from reverse KL;
- masking tests that exclude prompt and padding positions from the anchor.

Run:

```bash
.venv/bin/pytest tests/test_objectives.py -q
```

Expected: collection FAIL because `trainer.unlearn.objectives` does not exist.

- [ ] **Step 2: Define the shared data contract**

Create `src/trainer/unlearn/objectives.py` with the following public API:

```python
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


OBJECTIVE_DEFAULTS = {
    "graddiff": {"gamma": 1.0, "alpha": 1.0},
    "ceu": {"gamma": 1.0, "alpha": 0.0, "ignore_first_n_answer_tokens": 1},
    "dpo": {"gamma": 1.0, "alpha": 1.0, "beta": 0.1},
    "npo": {"gamma": 1.0, "alpha": 1.0, "beta": 0.1},
    "satimp": {"gamma": 0.1, "alpha": 1.0, "beta1": 5.0, "beta2": 1.0},
    "simnpo": {"gamma": 0.125, "alpha": 1.0, "beta": 4.5, "delta": 0.0},
    "undial": {"gamma": 1.0, "alpha": 0.0, "beta": 10.0},
    "wga": {"gamma": 1.0, "alpha": 1.0, "beta": 1.0},
}


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    gamma: float
    alpha: float
    retain_loss_type: str = "NLL"
    params: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def for_name(cls, name: str, **overrides: Any) -> "ObjectiveConfig":
        normalized = name.lower().replace("_", "")
        if normalized not in OBJECTIVE_DEFAULTS:
            raise ValueError(f"Unknown unlearning objective: {name}")
        values = dict(OBJECTIVE_DEFAULTS[normalized])
        values.update(overrides)
        gamma = float(values.pop("gamma"))
        alpha = float(values.pop("alpha"))
        retain_loss_type = str(values.pop("retain_loss_type", "NLL"))
        return cls(normalized, gamma, alpha, retain_loss_type, values)


@dataclass
class ObjectiveParts:
    forget_loss: torch.Tensor
    retain_loss: torch.Tensor | None
    outputs: Any
    gamma: float
    alpha: float
    retain_outputs: Any | None = None

    @property
    def total_loss(self) -> torch.Tensor:
        total = self.gamma * self.forget_loss
        if self.retain_loss is not None:
            total = total + self.alpha * self.retain_loss
        return total


def compute_objective_parts(
    config: ObjectiveConfig,
    model,
    inputs,
    ref_model=None,
) -> ObjectiveParts:
    raise NotImplementedError(f"Objective branch not implemented: {config.name}")
```

Implement exact defaults from the approved design and raise `ValueError` with the objective name and missing key when schemas are invalid. Do not silently reinterpret MUSE/WMDP data as DPO data.

- [ ] **Step 3: Move common CEU and forward-KL functions into `trainer.utils`**

Expose:

```python
def compute_batch_ceu(model, inputs, ignore_first_n_answer_tokens=1):
    outputs = model(**inputs)
    labels = inputs["labels"]
    valid_mask = labels.ne(-100)
    ignored = (valid_mask.cumsum(dim=-1) <= ignore_first_n_answer_tokens) & valid_mask
    labels = labels.masked_fill(ignored, -100)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = outputs.logits[..., :-1, :].contiguous()
    loss = cross_entropy_unlearning_loss(
        shifted_logits, shifted_labels, ignore_index=-100
    )
    return loss, outputs


def compute_forward_kl(model, ref_model, inputs):
    with torch.no_grad():
        ref_logits = ref_model(**inputs).logits[:, :-1]
    current_logits = model(**inputs).logits[:, :-1]
    labels = inputs["labels"][:, 1:]
    mask = labels.ne(-100)
    log_p = current_logits.log_softmax(dim=-1)
    log_q = ref_logits.log_softmax(dim=-1)
    p = log_p.exp()
    per_token = (p * (log_p - log_q)).sum(dim=-1)
    return per_token[mask].mean()
```

Remove the duplicate `ref_probs` assignment in `compute_kl_divergence`. Preserve the existing reverse-KL helper for baseline parity; the new forward-KL function is the explicitly named GU anchor.

- [ ] **Step 4: Implement all eight registry branches**

Reuse the existing loss helpers and the exact baseline coefficients. Do not read GU top-level `alpha/gamma`. Use each `ObjectiveConfig` as the only coefficient source.

Implement the dispatcher with this exact branch structure:

```python
def compute_objective_parts(config, model, inputs, ref_model=None):
    name = config.name
    if "forget" not in inputs:
        raise ValueError(f"{name} requires inputs['forget']")
    forget = inputs["forget"]

    if name == "graddiff":
        outputs = model(**forget)
        forget_loss = -outputs.loss
    elif name == "ceu":
        forget_loss, outputs = compute_batch_ceu(
            model,
            forget,
            int(config.params["ignore_first_n_answer_tokens"]),
        )
    elif name == "dpo":
        if not isinstance(forget, Mapping) or not {"original", "alternate"} <= forget.keys():
            raise ValueError("dpo requires forget.original and forget.alternate")
        forget_loss, outputs = compute_dpo_loss(
            model,
            ref_model,
            win_inputs=forget["alternate"],
            lose_inputs=forget["original"],
            beta=float(config.params["beta"]),
        )
    elif name == "npo":
        forget_loss, outputs = compute_dpo_loss(
            model,
            ref_model,
            win_inputs=None,
            lose_inputs=forget,
            beta=float(config.params["beta"]),
        )
    elif name == "satimp":
        forget_loss, outputs = compute_satimp_loss(
            model,
            forget,
            float(config.params["beta1"]),
            float(config.params["beta2"]),
        )
    elif name == "simnpo":
        labels = forget["labels"]
        token_count = labels.ne(-100).sum(dim=-1)
        sequence_nll, outputs = compute_batch_nll(model, forget)
        normalized = sequence_nll / token_count - float(config.params["delta"])
        beta = float(config.params["beta"])
        forget_loss = -F.logsigmoid(beta * normalized).mean() * 2.0 / beta
    elif name == "undial":
        forget_loss, outputs = compute_undial_loss(
            model, ref_model, forget, float(config.params["beta"])
        )
    elif name == "wga":
        forget_loss, outputs = compute_wga_loss(
            model, forget, float(config.params["beta"])
        )
    else:
        raise ValueError(f"Unknown unlearning objective: {name}")

    retain_loss = None
    retain_outputs = None
    if name != "ceu":
        if "retain" not in inputs:
            raise ValueError(f"{name} requires inputs['retain']")
        retain = inputs["retain"]
        if config.retain_loss_type.upper() == "NLL":
            retain_outputs = model(**retain)
            retain_loss = retain_outputs.loss
        elif config.retain_loss_type.upper() == "KL":
            retain_loss, retain_outputs = compute_kl_divergence(
                model, ref_model, retain
            )
        else:
            raise ValueError(
                f"Unsupported retain loss for {name}: {config.retain_loss_type}"
            )
    return ObjectiveParts(
        forget_loss=forget_loss,
        retain_loss=retain_loss,
        outputs=outputs,
        gamma=config.gamma,
        alpha=config.alpha,
        retain_outputs=retain_outputs,
    )
```

- [ ] **Step 5: Verify registry tests turn green**

Run:

```bash
.venv/bin/pytest tests/test_objectives.py -q
```

Expected: all registry tests pass.

- [ ] **Step 6: Refactor baseline trainers one at a time**

Each trainer declares its objective name and delegates:

```python
parts = compute_objective_parts(self.objective_config, model, inputs, self.ref_model)
return (parts.total_loss, parts.outputs) if return_outputs else parts.total_loss
```

After each file, run the corresponding parametrized parity test. Preserve constructor arguments and public trainer names.

- [ ] **Step 7: Run the full objective suite**

```bash
.venv/bin/pytest tests/test_objectives.py -q
```

Expected: scalar and gradient parity passes for all eight objectives.

- [ ] **Step 8: Commit objective unification**

```bash
git add src/trainer/unlearn src/trainer/utils.py tests/test_objectives.py
git commit -m "refactor: unify unlearning objective semantics"
```

## Task 3: Implement the Correct Metric and Basis

**Files:**
- Create: `src/trainer/unlearn/geometric_projector.py`
- Create: `tests/test_geometric_projector.py`

- [ ] **Step 1: Write failing coordinate-map tests**

Use float64 vectors and assert:

```python
denominator = torch.tensor([1.0, 4.0], dtype=torch.float64)
gradient = torch.tensor([2.0, 8.0], dtype=torch.float64)
z = raw_gradient_to_white(gradient, denominator)
assert torch.allclose(z, gradient / denominator.sqrt())
assert torch.allclose(white_to_equivalent_gradient(z, denominator), gradient)
assert torch.allclose(white_to_update(z, denominator), gradient / denominator)
```

Add tests for positivity validation, dtype promotion, zero-size tensors, and identity denominator.

Run:

```bash
.venv/bin/pytest tests/test_geometric_projector.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 2: Implement coordinate functions**

Create these pure functions:

```python
def raw_gradient_to_white(gradient, denominator):
    return gradient.float() * denominator.float().rsqrt()


def white_to_equivalent_gradient(vector, denominator):
    return vector.float() * denominator.float().sqrt()


def white_to_update(vector, denominator):
    return vector.float() * denominator.float().rsqrt()
```

Validate that every denominator is finite and strictly positive.

- [ ] **Step 3: Verify RED becomes GREEN**

```bash
.venv/bin/pytest tests/test_geometric_projector.py -k coordinate -q
```

Expected: coordinate tests pass.

- [ ] **Step 4: Write failing basis tests**

Test two-pass modified Gram--Schmidt, `Q.T @ Q == I`, dependent-vector rejection, raw FIFO eviction, re-whitening under a changed denominator, mandatory current-gradient insertion, and empty basis.

- [ ] **Step 5: Implement `RawGradientBasis`**

Public contract:

```python
from collections import deque


class RawGradientBasis:
    def __init__(self, max_rank, residual_tolerance, history_size=None):
        self.max_rank = int(max_rank)
        self.residual_tolerance = float(residual_tolerance)
        size = self.max_rank if history_size is None else int(history_size)
        self.history = deque(maxlen=size)

    def append_raw(self, gradient):
        self.history.append(gradient.detach().flatten().to(torch.float16).clone())

    def build(self, denominator, mandatory=()):
        candidates = [item.detach().flatten() for item in mandatory]
        candidates.extend(reversed(tuple(self.history)))
        columns = []
        flat_denominator = denominator.detach().flatten().float()
        for raw in candidates:
            white = raw.to(flat_denominator.device).float() * flat_denominator.rsqrt()
            reference_norm = white.norm().clamp_min(1e-12)
            residual = white.clone()
            for _ in range(2):
                for column in columns:
                    residual -= column * torch.dot(column, residual)
            residual_norm = residual.norm()
            if residual_norm / reference_norm >= self.residual_tolerance:
                columns.append(residual / residual_norm.clamp_min(1e-12))
            if len(columns) == self.max_rank:
                break
        if not columns:
            return torch.empty(flat_denominator.numel(), 0, device=flat_denominator.device)
        return torch.stack(columns, dim=1)

    @staticmethod
    def coverage(white_gradient, q):
        vector = white_gradient.flatten().float()
        if vector.norm() == 0:
            return 0.0
        if q.shape[1] == 0:
            return 1.0
        residual = vector - q @ (q.T @ vector)
        return float(residual.norm() / vector.norm())
```

Store detached FP16/BF16-compatible raw history, but build and project in FP32. Rebuild against the current metric every update.

- [ ] **Step 6: Verify basis tests**

```bash
.venv/bin/pytest tests/test_geometric_projector.py -k basis -q
```

Expected: basis tests pass with orthogonality error below `1e-5` in FP32.

- [ ] **Step 7: Commit metric and basis**

```bash
git add src/trainer/unlearn/geometric_projector.py tests/test_geometric_projector.py
git commit -m "feat: add correct optimizer metric and retain basis"
```

## Task 4: Implement Paper and Safe-Cone Stage-A Projection

**Files:**
- Modify: `src/trainer/unlearn/geometric_projector.py`
- Modify: `tests/test_geometric_projector.py`

- [ ] **Step 1: Write failing paper-rule tests**

For `Q=e1` and `z_r=e1`:

```python
helpful = torch.tensor([1.0, 1.0])
harmful = torch.tensor([-1.0, 1.0])
assert paper_project(helpful, z_r, q, tau=0.0, kappa=1.0).safe[0] > 0
assert paper_project(harmful, z_r, q, tau=0.0, kappa=1.0).safe[0] == 0
```

Also test cap norm, zero-normal behavior, repair, empty basis, and `g_r.T @ (-update) <= tolerance`.

- [ ] **Step 2: Implement structured projection results**

Add:

```python
@dataclass
class ProjectionResult:
    safe: torch.Tensor
    normal: torch.Tensor
    kept_tangent: torch.Tensor
    coverage: float
    predicted_retain_derivative: float
    diagnostics: dict[str, float]
```

Implement `paper_project` with `a*b > tau`, not the old reversed condition.

- [ ] **Step 3: Verify paper tests**

```bash
.venv/bin/pytest tests/test_geometric_projector.py -k paper -q
```

Expected: all paper-rule tests pass.

- [ ] **Step 4: Write failing cone tests**

Test a two-constraint case where componentwise deletion is more destructive than the cone solution. Assert:

- all `R.T @ z >= -tol`;
- the cone result is no farther from the candidate than the pure-normal fallback;
- norm does not increase;
- deterministic convergence;
- forced non-convergence returns the pure-normal fallback and records it.

- [ ] **Step 5: Implement the low-rank half-space solver**

Implement Dykstra projection in tangent coordinates with explicit `max_iterations`, `tolerance`, and `max_violation`. Keep the normal component unchanged. Do not add SciPy.

- [ ] **Step 6: Run all projector tests**

```bash
.venv/bin/pytest tests/test_geometric_projector.py -q
```

Expected: all metric, basis, paper, and cone tests pass.

- [ ] **Step 7: Commit Stage-A math**

```bash
git add src/trainer/unlearn/geometric_projector.py tests/test_geometric_projector.py
git commit -m "feat: implement paper and safe-cone projections"
```

## Task 5: Implement Accumulation and Optimizer Runtime

**Files:**
- Create: `src/trainer/unlearn/geometric_runtime.py`
- Create: `tests/test_geometric_runtime.py`

- [ ] **Step 1: Write failing accumulation tests**

Test windows of 1, 2, and 4 microbatches. The buffer must equal the explicitly scaled sum and reject mixed coefficients or window IDs.

Public API under test:

```python
window = GradientWindow(parameter_names, accumulation_steps=2)
window.add("forget", gradients, window_id=0, gamma=0.125, alpha=1.0)
reduced = window.finalize(reducer=lambda tensor: tensor)
```

Expected first run: import FAIL.

- [ ] **Step 2: Implement `GradientWindow`**

Keep independent FP32 maps for forget, objective retain, and safety/anchor retain. Never derive one component from a differently defined component. Clear stale data on a new window ID.

- [ ] **Step 3: Test DDP reduction seam**

Use an injected reducer and assert exactly one mean reduction per populated parameter/component. The real trainer will inject `accelerator.reduce(tensor, reduction="mean")`.

- [ ] **Step 4: Write failing optimizer metric tests**

Test:

- Torch `exp_avg_sq` with group epsilon and beta2 bias correction;
- bitsandbytes-style floating `state2`;
- invalid 8-bit state rejection;
- first-step identity fallback;
- parameters in different groups.

- [ ] **Step 5: Implement `OptimizerMetricAdapter`**

Expose:

```python
def denominator_for(self, parameter, gradient=None) -> torch.Tensor:
    optimizer = self.optimizer
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    group = next(
        group
        for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    )
    epsilon = float(group.get("eps", 1e-8))
    state = optimizer.state.get(parameter, {})
    second_moment = state.get("exp_avg_sq", state.get("state2"))
    if second_moment is None:
        self.last_source = "identity-first-step"
        return torch.ones_like(parameter, dtype=torch.float32)
    if not second_moment.is_floating_point() or second_moment.shape != parameter.shape:
        raise ValueError("Unsupported optimizer second-moment state")
    step = state.get("step", 0)
    step = int(step.item()) if torch.is_tensor(step) else int(step)
    beta2 = float(group.get("betas", (0.9, 0.999))[1])
    corrected = second_moment.float()
    if step > 0:
        corrected = corrected / (1.0 - beta2**step)
    self.last_source = "optimizer-state"
    return corrected.sqrt().add(epsilon)
```

Unwrap Accelerate optimizers without importing private types. Use shape/dtype checks and log the metric source.

- [ ] **Step 6: Write failing raw-hook lifecycle tests**

With `torch.optim.AdamW`, assert:

- pre-hook runs once per real step;
- hook installation is idempotent;
- projected gradients are clipped only after replacement;
- post-hook receives an exact before/after snapshot;
- cleanup handles a skipped window.

- [ ] **Step 7: Implement hook installation and runtime state**

Expose idempotent `install_hooks(optimizer, runtime)` and removable hook handles. Preserve the original clip threshold and use `torch.nn.utils.clip_grad_norm_` inside the raw pre-hook.

- [ ] **Step 8: Run runtime tests**

```bash
.venv/bin/pytest tests/test_geometric_runtime.py -q
```

Expected: all runtime tests pass.

- [ ] **Step 9: Commit runtime**

```bash
git add src/trainer/unlearn/geometric_runtime.py tests/test_geometric_runtime.py
git commit -m "feat: add geometric optimizer runtime"
```

## Task 6: Replace the GU Trainer and Configuration

**Files:**
- Replace: `src/trainer/unlearn/geometric.py`
- Modify: `configs/trainer/GeometricUnlearn.yaml`
- Create: `configs/accelerate/ddp.yaml`
- Create: `tests/test_geometric_trainer.py`

- [ ] **Step 1: Write failing trainer-construction tests**

Test exact parameter selection, final-norm selection that excludes block norms, explicit regex validation, `freeze_others`, and configuration normalization for all eight objectives.

- [ ] **Step 2: Implement a small configuration dataclass**

Normalize OmegaConf into immutable settings:

```python
@dataclass(frozen=True)
class GeometricConfig:
    projection_mode: str = "paper"
    stage_b: bool = False
    null_rank: int = 8
    refresh_every: int = 2
    coverage_tolerance: float = 0.05
    kappa: float = 0.5
    tau: float = 0.0
    freeze_others: bool = False
```

Reject unknown projection modes, zero rank, negative tolerances, and empty explicit selections.

- [ ] **Step 3: Write failing compute-loss parity tests**

With geometry disabled/empty basis, compare GU and each baseline scalar loss and all gradients. Confirm CEU and UNDIAL keep `alpha=0` and SimNPO keeps `gamma=0.125`.

- [ ] **Step 4: Implement GU orchestration**

The new class must:

- call the shared objective registry;
- capture separately differentiated component gradients for selected tensors;
- use the same Accelerate accumulation scaling;
- capture a correctly masked forward-KL anchor only when scheduled;
- override `create_optimizer()` to install raw hooks;
- delete the dead `optimizer_step()` path;
- fail fast for DeepSpeed/FSDP and reentrant checkpointing;
- register cleanup/logging callback state;
- preserve unselected baseline gradients.

- [ ] **Step 5: Add supported DDP config**

Create `configs/accelerate/ddp.yaml` with `distributed_type: MULTI_GPU`, BF16, no DeepSpeed section, and configurable process count. Keep the old default unchanged for non-GU methods, but GU must reject it with a command suggesting the DDP file.

- [ ] **Step 6: Rewrite `GeometricUnlearn.yaml`**

Use explicit objective blocks and remove dead nesting/fields. Default to SimNPO's exact `gamma=0.125, alpha=1.0`, paper projection, rank 8, refresh 2, `kappa=0.5`, `tau=0`, Stage B off, and first-step identity metric.

- [ ] **Step 7: Run trainer tests**

```bash
.venv/bin/pytest tests/test_geometric_trainer.py -q
```

Expected: construction, selection, parity, and hook reachability tests pass.

- [ ] **Step 8: Commit trainer integration**

```bash
git add src/trainer/unlearn/geometric.py configs/trainer/GeometricUnlearn.yaml configs/accelerate/ddp.yaml tests/test_geometric_trainer.py
git commit -m "feat: integrate reachable Stage-A GU trainer"
```

## Task 7: Verify Stage A Under Accumulation, AMP, and DDP

**Files:**
- Modify: `tests/test_geometric_trainer.py`
- Modify: `.github/workflows/tests.yml`
- Modify: `Makefile`

- [ ] **Step 1: Install the pinned integration stack**

Run:

```bash
uv pip install --python .venv/bin/python \
  transformers==4.45.1 accelerate==0.34.2 huggingface-hub==0.29.1 \
  omegaconf==2.3.0 hydra-core==1.3.2 datasets==3.0.1
```

Install bitsandbytes 0.44.1 only for the CUDA adapter test.

```bash
uv pip install --python .venv/bin/python bitsandbytes==0.44.1
```

- [ ] **Step 2: Write failing accumulation integration tests**

For accumulation 1, 2, and 4, compare one GU update against a hand-computed reference. Include a dataset shorter than the accumulation window. Assert exactly one hook per optimizer update.

- [ ] **Step 3: Make accumulation tests green**

Fix only lifecycle/scaling errors exposed by the tests. Do not weaken expected values.

- [ ] **Step 4: Add CUDA BF16/FP16 tests**

Skip only when CUDA is unavailable. Assert scale/unscale/replacement/clip ordering and inject an Inf to confirm no stale window leaks into the next update.

- [ ] **Step 5: Add a two-rank DDP test**

Launch with:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m pytest tests/test_geometric_trainer.py -k ddp -q
```

Expected: both ranks produce equal parameters and match the equivalent single-process batch.

- [ ] **Step 6: Enable pytest in CI and Makefile**

Make `make test` run the CPU suite. CI installs the dev extra and runs CPU tests; GPU/DDP tests remain explicitly marked integration tests for the local verification command.

- [ ] **Step 7: Run Stage-A verification**

```bash
.venv/bin/pytest tests/test_test_harness.py tests/test_objectives.py \
  tests/test_geometric_projector.py tests/test_geometric_runtime.py \
  tests/test_geometric_trainer.py -m "not gpu and not ddp" -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest tests/test_geometric_trainer.py -m gpu -q
```

Expected: zero failures.

- [ ] **Step 8: Commit integration verification**

```bash
git add tests Makefile .github/workflows/tests.yml
git commit -m "test: verify Stage-A geometric training lifecycle"
```

## Task 8: Run the Checkpoint-Free Stage-A 1B Validation

**Files:**
- Create: `scripts/quick_validate_geometric.py`
- Create: `tests/test_quick_validate_no_checkpoint.py`
- Modify: `.gitignore`
- Create: `results/geometric_quick/.gitkeep`
- Create: `GEOMETRIC_IMPLEMENTATION_REPORT.md`

- [ ] **Step 1: Write a failing no-checkpoint test**

The test runs the driver in `--dry-run` mode against a temporary directory and asserts the resolved plan sets:

- `save_strategy=no`;
- `report_to=none`;
- no call to `save_model()` or `save_state()`;
- allowed outputs end only in `.json` or `.jsonl`.

- [ ] **Step 2: Implement the validation driver**

The driver must expose:

```text
--model open-unlearning/tofu_Llama-3.2-1B-Instruct_full
--forget-split forget05
--retain-split retain95
--max-steps 8
--train-examples 32
--eval-examples 32
--extraction-examples 16
--arms baseline,paper,cone
--output-dir results/geometric_quick/run01
--no-checkpoints
```

It reloads the same base model per arm, uses seed 0 and the same sample order, trains in memory, evaluates compact metrics, writes one JSON summary plus step-level JSONL diagnostics, and asserts no checkpoint/model/state files exist.

Add these ignore rules:

```gitignore
results/geometric_quick/*
!results/geometric_quick/.gitkeep
```

- [ ] **Step 3: Verify dry-run and no-checkpoint behavior**

```bash
.venv/bin/pytest tests/test_quick_validate_no_checkpoint.py -q
.venv/bin/python scripts/quick_validate_geometric.py --dry-run --no-checkpoints
```

Expected: test passes and the dry-run prints the three arms without loading a model.

- [ ] **Step 4: Run baseline, paper, and cone arms on one A100**

```bash
HF_HOME=/workspace/.cache/huggingface CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/quick_validate_geometric.py \
  --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --forget-split forget05 --retain-split retain95 \
  --max-steps 8 --train-examples 32 --eval-examples 32 \
  --extraction-examples 16 --arms baseline,paper,cone \
  --output-dir results/geometric_quick/run01 --no-checkpoints
```

Expected: three arms finish; only JSON/JSONL files exist under the output directory.

- [ ] **Step 5: Apply the fixed Pareto gate**

Reject an arm if retain NLL is more than 2% above baseline, forward KL exceeds `baseline + 0.01`, or an available ES-Re/Privacy/MU metric is more than 0.02 below baseline. Among feasible arms, choose the lowest forgetting score. Do not change tolerances after seeing results.

- [ ] **Step 6: Escalate only if all arms are indistinguishable**

If every effect is within measured numerical noise, rerun all three arms at 24 steps under `run02`. Otherwise do not spend the extra compute.

- [ ] **Step 7: Write Stage-A analysis**

Record commands, environment, hook counts, metric source, coverage, harmful mass, predicted retain derivative, finite retain change, cap activation, wall time, memory, and the Pareto decision in `GEOMETRIC_IMPLEMENTATION_REPORT.md`. Label the run diagnostic, not submission-grade.

- [ ] **Step 8: Commit code and report, not generated result JSON**

```bash
git add scripts/quick_validate_geometric.py tests/test_quick_validate_no_checkpoint.py \
  .gitignore results/geometric_quick/.gitkeep GEOMETRIC_IMPLEMENTATION_REPORT.md
git commit -m "exp: add checkpoint-free Stage-A validation"
```

## Task 9: Implement Stage-B Realized-Delta Projection

**Files:**
- Modify: `src/trainer/unlearn/geometric_projector.py`
- Modify: `src/trainer/unlearn/geometric_runtime.py`
- Modify: `src/trainer/unlearn/geometric.py`
- Modify: `tests/test_geometric_projector.py`
- Modify: `tests/test_geometric_runtime.py`
- Modify: `tests/test_geometric_trainer.py`

- [ ] **Step 1: Write the momentum/weight-decay counterexample test**

Construct an AdamW parameter whose current gradient is retain-normal but whose old momentum and weight decay produce a harmful realized delta. Assert Stage A alone permits the harmful actual derivative and Stage B must correct it to non-positive.

- [ ] **Step 2: Write pure realized-delta tests**

For `delta = theta_after - theta_before`, test:

```python
corrected = project_realized_delta(delta, retain_gradients, denominator)
for retain_gradient in retain_gradients:
    assert torch.dot(retain_gradient.flatten(), corrected.flatten()) <= 1e-6
```

Also test nearestness, identity on already-safe deltas, zero delta, and multiple constraints.

- [ ] **Step 3: Implement pure Stage-B projection**

Map `delta` with `C=sqrt(H)`, solve the `R.T @ y <= 0` cone projection, and map back with `S=P^(1/2)`. Reuse the Stage-A low-rank solver with an explicit sign convention rather than duplicating it.

- [ ] **Step 4: Write failing pre/post-hook integration tests**

Assert the pre-hook snapshots protected parameters and the post-hook corrects the exact realized delta after AdamW state, momentum, and weight decay have been applied. Verify the optimizer state itself is not rewritten.

- [ ] **Step 5: Implement Stage-B runtime**

When `stage_b=true`:

- pre-hook runs Stage A and snapshots selected parameters plus frozen metric/constraints;
- post-hook computes and corrects the realized delta;
- diagnostics include raw derivative, corrected derivative, and correction ratio;
- all snapshots are released immediately after the post-hook.

- [ ] **Step 6: Verify selected/global guarantee boundaries**

Add one test showing unselected harmful updates remain possible when `freeze_others=false`, and one showing the complete toy-model derivative is safe when every trainable parameter is selected.

- [ ] **Step 7: Run Stage-B tests**

```bash
.venv/bin/pytest tests/test_geometric_projector.py tests/test_geometric_runtime.py \
  tests/test_geometric_trainer.py -k "stage_b or realized_delta or momentum" -q
```

Expected: all Stage-B tests pass.

- [ ] **Step 8: Commit Stage B**

```bash
git add src/trainer/unlearn/geometric_projector.py \
  src/trainer/unlearn/geometric_runtime.py src/trainer/unlearn/geometric.py \
  tests/test_geometric_projector.py tests/test_geometric_runtime.py \
  tests/test_geometric_trainer.py
git commit -m "feat: constrain realized AdamW updates"
```

## Task 10: Run Paired Stage-B 1B Validation and Deepen the Theory Analysis

**Files:**
- Modify: `scripts/quick_validate_geometric.py`
- Modify: `GEOMETRIC_IMPLEMENTATION_REPORT.md`

- [ ] **Step 1: Add the Stage-B arm to dry-run tests**

The arm name is `cone-stage-b`; it must use the same initial model, data order, seed, objective, and step budget as the earlier arms.

- [ ] **Step 2: Run the Stage-B arm**

```bash
HF_HOME=/workspace/.cache/huggingface CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/quick_validate_geometric.py \
  --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --forget-split forget05 --retain-split retain95 \
  --max-steps 8 --train-examples 32 --eval-examples 32 \
  --extraction-examples 16 --arms cone-stage-b \
  --output-dir results/geometric_quick/run01 --no-checkpoints
```

Expected: the arm finishes without checkpoints and appends a compatible summary.

- [ ] **Step 3: Compare theory with observations**

Compute and report:

- frequency and magnitude of Stage-A safe prediction versus harmful realized AdamW delta;
- Stage-B correction ratio distribution;
- whether correction improves retain finite-step behavior;
- whether cone mode preserves more forget descent than paper mode;
- whether coverage, unselected tensors, or second-order curvature explains any residual violation;
- runtime and peak-memory increments.

- [ ] **Step 4: Apply the same frozen Pareto gate**

Do not introduce a new tolerance for Stage B. Compare all four arms under the original gate and select the best feasible arm.

- [ ] **Step 5: Update the theory boundaries**

In `GEOMETRIC_IMPLEMENTATION_REPORT.md`, explicitly state which of these conclusions the data supports:

- momentum/weight decay materially reintroduce harmful tangent motion;
- Stage A already makes the actual-step gap negligible;
- cone projection is less conservative than the paper rule;
- finite-step curvature, not first-order geometry, is the remaining limiter;
- basis/layer coverage is the remaining limiter.

Do not infer a conclusion when the measured signal is below numerical noise.

- [ ] **Step 6: Commit the paired analysis**

```bash
git add scripts/quick_validate_geometric.py GEOMETRIC_IMPLEMENTATION_REPORT.md
git commit -m "exp: analyze realized-step geometric safety"
```

## Task 11: Final Verification and Handoff

**Files:**
- Modify as required by verified failures only.

- [ ] **Step 1: Run the complete CPU suite**

```bash
.venv/bin/pytest -m "not gpu and not ddp" -q
```

Expected: zero failures.

- [ ] **Step 2: Run GPU and DDP integration tests**

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/pytest -m gpu -q
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m pytest tests/test_geometric_trainer.py -k ddp -q
```

Expected: zero failures on both commands.

- [ ] **Step 3: Run quality checks**

```bash
.venv/bin/ruff check scripts src tests setup.py setup_data.py
.venv/bin/ruff format --check scripts src tests setup.py setup_data.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 4: Verify no checkpoints exist**

```bash
find results/geometric_quick -type f \
  \( -name '*.bin' -o -name '*.safetensors' -o -name '*.pt' \
     -o -name 'trainer_state.json' -o -name 'optimizer.pt' \) -print
```

Expected: no output.

- [ ] **Step 5: Audit the implementation against the design**

Check every completion criterion in `docs/superpowers/specs/2026-07-10-geometric-unlearning-correctness-design.md`; record any unsupported item rather than silently omitting it.

- [ ] **Step 6: Request independent code and theory review**

Dispatch one reviewer for mathematical invariants and one for Trainer/runtime correctness. Address only technically verified findings, then rerun Steps 1--4.

- [ ] **Step 7: Commit final verified fixes**

```bash
git add -u
git commit -m "fix: address geometric implementation audit"
```

Skip this commit when the independent review requires no changes.
