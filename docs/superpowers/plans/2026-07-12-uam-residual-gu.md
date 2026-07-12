# UAM and Residual-GU-UAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add checkpoint-free UAM and residual-projected UAM+GU trainers with NLL/SimNPO forget signals, paper and optimizer-metric perturbations, exact GAS replay, and Adam/PagedAdamW32 BF16/FP32 support.

**Architecture:** Add a separate `UAMUnlearn` trainer that reuses the existing selected-parameter, optimizer-geometry, component-buffer, and actual-delta infrastructure without adding UAM branches to the GU/PCGrad finalizer. Global UAM decisions are computed as scalar coefficients in `uam_geometry.py`; `uam.py` performs two-pass effective-batch collection, temporary perturbation, retain replay, exact restoration, and streaming gradient writeback.

**Tech Stack:** Python 3.11, PyTorch 2.4, Transformers 4.45.1, Accelerate, bitsandbytes 0.44.1, OmegaConf/Hydra, pytest, Ruff, Bash.

---

## File Map

- Create `src/trainer/unlearn/uam_geometry.py`: optimizer-coordinate UAM and residual-GU scalar decisions plus tensor transforms.
- Create `src/trainer/unlearn/uam.py`: `UAMUnlearn`, replay buffer, perturbation transaction, training step, finalizer, diagnostics.
- Modify `src/trainer/unlearn/component_buffers.py`: add the `perturbed_retain` component and configurable host-memory accounting.
- Modify `src/trainer/__init__.py`: import and register `UAMUnlearn`.
- Create `configs/trainer/UAMUnlearn.yaml`: supported defaults and no-checkpoint contract.
- Create `tests/test_uam_geometry.py`: mathematical unit tests.
- Create `tests/test_uam_replay.py`: replay-tree and reversible-perturbation tests.
- Create `tests/test_uam_trainer.py`: config, signal, lifecycle, GAS, and actual-update tests.
- Modify `tests/test_gu_gpu_integration.py`: BF16/Flash UAM equivalence check.
- Create `tests/test_uam_gpu_integration.py`: Torch AdamW and PagedAdamW32 UAM update checks.
- Create `scripts/uam_smoke_arm.sh`: one no-checkpoint UAM arm.
- Create `scripts/uam_smoke_matrix.sh`: four-arm two-GPU scheduler and manifest.
- Create `scripts/analyze_uam_smoke.py`: validate artifacts and summarize geometry/metrics.
- Create `tests/test_uam_scripts.py`: script contract and analyzer fixtures.
- Create `UAM_SMOKE_REPORT.md`: generated mechanism report after successful runs.

### Task 1: Add global UAM geometry decisions

**Files:**
- Create: `src/trainer/unlearn/uam_geometry.py`
- Create: `tests/test_uam_geometry.py`

- [ ] **Step 1: Write failing perturbation-normalization tests**

```python
import pytest
import torch

from trainer.unlearn.uam_geometry import decide_perturbation


def test_fixed_loss_normalization_has_requested_linearized_increase():
    gradient = torch.tensor([3.0, 4.0])
    decision = decide_perturbation(
        "fixed_loss",
        raw_forget_sq=gradient.square().sum(),
        optimizer_forget_sq=gradient.square().sum(),
        rho=0.2,
        eps=1e-12,
    )
    delta = decision.coefficient * gradient
    assert torch.dot(gradient, delta).item() == pytest.approx(0.2)


def test_metric_trust_normalization_has_requested_metric_radius():
    gradient = torch.tensor([2.0, -1.0])
    denominator = torch.tensor([4.0, 0.25])
    coordinates = gradient / denominator.sqrt()
    decision = decide_perturbation(
        "metric_trust",
        raw_forget_sq=gradient.square().sum(),
        optimizer_forget_sq=coordinates.square().sum(),
        rho=0.3,
        eps=1e-12,
    )
    delta = decision.coefficient * gradient / denominator
    metric_norm = (delta.square() * denominator).sum().sqrt()
    assert metric_norm.item() == pytest.approx(0.3)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda activate unlearning
pytest -q tests/test_uam_geometry.py -k normalization
```

Expected: collection fails with `ModuleNotFoundError: trainer.unlearn.uam_geometry`.

- [ ] **Step 3: Implement perturbation decisions**

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerturbationDecision:
    mode: str
    coefficient: torch.Tensor


def decide_perturbation(
    mode,
    raw_forget_sq,
    optimizer_forget_sq,
    rho,
    eps,
):
    if mode == "fixed_loss":
        denominator = raw_forget_sq.clamp_min(eps)
    elif mode == "metric_trust":
        denominator = optimizer_forget_sq.clamp_min(eps).sqrt()
    else:
        raise ValueError(f"Unsupported UAM perturbation normalization: {mode}")
    if float(raw_forget_sq.item()) == 0.0:
        raise RuntimeError("UAM forget gradient has zero norm.")
    return PerturbationDecision(
        mode=mode,
        coefficient=torch.as_tensor(rho, device=denominator.device) / denominator,
    )
```

- [ ] **Step 4: Verify normalization tests GREEN**

Run: `pytest -q tests/test_uam_geometry.py -k normalization`

Expected: `2 passed`.

- [ ] **Step 5: Write failing UAM and residual decision tests**

```python
from trainer.unlearn.uam_geometry import (
    apply_residual_gu_tensor,
    apply_uam_tensor,
    decide_residual_gu,
    decide_uam,
)


def test_gamma_two_is_householder_reflection():
    forget = torch.tensor([1.0, 0.0])
    perturbed_retain = torch.tensor([3.0, 4.0])
    decision = decide_uam(
        forget_perturbed_retain_dot=torch.dot(forget, perturbed_retain),
        optimizer_forget_sq=forget.square().sum(),
        reflection_gamma=2.0,
        eps=1e-12,
    )
    reflected = apply_uam_tensor(perturbed_retain, forget, decision)
    torch.testing.assert_close(reflected, torch.tensor([-3.0, 4.0]))


def test_residual_gu_keeps_retain_descent_and_normal_correction():
    forget = torch.tensor([1.0, 1.0])
    retain = torch.tensor([1.0, 0.0])
    uam = torch.tensor([0.0, -2.0])
    residual = uam - retain
    decision = decide_residual_gu(
        residual_retain_dot=torch.dot(residual, retain),
        residual_forget_dot=torch.dot(residual, forget),
        forget_retain_dot=torch.dot(forget, retain),
        optimizer_retain_sq=retain.square().sum(),
        residual_lambda=1.0,
        sign_tau=0.0,
        eps=1e-12,
    )
    final, normal = apply_residual_gu_tensor(uam, retain, decision)
    assert torch.dot(normal, retain).item() == pytest.approx(0.0, abs=1e-7)
    assert torch.dot(forget, normal).item() < 0.0
    torch.testing.assert_close(final, torch.tensor([1.0, -2.0]))
```

- [ ] **Step 6: Run the tests and verify RED**

Run: `pytest -q tests/test_uam_geometry.py -k 'householder or residual_gu'`

Expected: import errors for the missing decision and apply functions.

- [ ] **Step 7: Implement UAM and residual-GU decisions**

```python
@dataclass(frozen=True)
class UAMDecision:
    coefficient: torch.Tensor


@dataclass(frozen=True)
class ResidualGUDecision:
    projection_coefficient: torch.Tensor
    residual_lambda: float
    gate_dot: torch.Tensor
    keep: bool


def decide_uam(
    forget_perturbed_retain_dot,
    optimizer_forget_sq,
    reflection_gamma,
    eps,
):
    coefficient = (
        float(reflection_gamma)
        * forget_perturbed_retain_dot
        / optimizer_forget_sq.clamp_min(eps)
    )
    return UAMDecision(coefficient=coefficient)


def apply_uam_tensor(perturbed_retain, forget, decision):
    return perturbed_retain.float() - decision.coefficient * forget.float()


def decide_residual_gu(
    residual_retain_dot,
    residual_forget_dot,
    forget_retain_dot,
    optimizer_retain_sq,
    residual_lambda,
    sign_tau,
    eps,
):
    if float(optimizer_retain_sq.item()) == 0.0:
        raise RuntimeError("Residual-GU-UAM retain gradient has zero norm.")
    coefficient = residual_retain_dot / optimizer_retain_sq.clamp_min(eps)
    gate_dot = residual_forget_dot - coefficient * forget_retain_dot
    return ResidualGUDecision(
        projection_coefficient=coefficient,
        residual_lambda=float(residual_lambda),
        gate_dot=gate_dot,
        keep=bool(gate_dot.item() < -float(sign_tau)),
    )


def apply_residual_gu_tensor(uam, retain, decision):
    residual = uam.float() - retain.float()
    normal = residual - decision.projection_coefficient * retain.float()
    correction = decision.residual_lambda * normal if decision.keep else 0.0 * normal
    return retain.float() + correction, normal
```

- [ ] **Step 8: Add validation and direct-total-projection regression cases**

Add tests that assert:

```python
with pytest.raises(RuntimeError, match="forget gradient has zero norm"):
    decide_perturbation("fixed_loss", torch.tensor(0.0), torch.tensor(0.0), 1.0, 1e-12)

with pytest.raises(RuntimeError, match="retain gradient has zero norm"):
    decide_residual_gu(
        torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.0),
        torch.tensor(0.0), 1.0, 0.0, 1e-12,
    )

direct_total_projection = uam - torch.dot(uam, retain) * retain
assert torch.dot(direct_total_projection, retain).item() == pytest.approx(0.0)
assert torch.dot(final, retain).item() == pytest.approx(retain.square().sum().item())
```

- [ ] **Step 9: Run and format the geometry suite**

Run:

```bash
pytest -q tests/test_uam_geometry.py
ruff check src/trainer/unlearn/uam_geometry.py tests/test_uam_geometry.py
ruff format --check src/trainer/unlearn/uam_geometry.py tests/test_uam_geometry.py
```

Expected: all tests and checks pass.

- [ ] **Step 10: Commit geometry primitives**

```bash
git add src/trainer/unlearn/uam_geometry.py tests/test_uam_geometry.py
git commit -m "feat: add UAM residual geometry"
```

### Task 2: Extend component buffers for perturbed retain gradients

**Files:**
- Modify: `src/trainer/unlearn/component_buffers.py`
- Modify: `tests/test_component_buffers.py`

- [ ] **Step 1: Write failing third-component and memory-accounting tests**

```python
def test_perturbed_retain_component_is_independent():
    parameter = torch.nn.Parameter(torch.ones(2))
    buffers = ComponentGradientBuffers("parameter", pin_memory=False)
    buffers.add("perturbed_retain", [("weight", parameter)], [torch.tensor([2.0, 3.0])])
    torch.testing.assert_close(
        buffers.tensor("perturbed_retain", "weight", parameter.device),
        torch.tensor([2.0, 3.0]),
    )
    assert buffers.has_component("perturbed_retain")


def test_host_memory_estimate_accepts_component_count():
    expected = 3 * 10 * torch.finfo(torch.float32).bits // 8
    assert ComponentGradientBuffers.required_host_bytes(
        10, headroom=1.0, component_count=3
    ) == expected
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_component_buffers.py -k 'perturbed or component_count'`

Expected: unsupported component and unexpected keyword failures.

- [ ] **Step 3: Implement the generic component count**

Change:

```python
_COMPONENTS = {"forget", "retain", "perturbed_retain"}
```

and:

```python
@staticmethod
def required_host_bytes(
    selected_numel: int,
    headroom: float = 1.2,
    component_count: int = 2,
) -> int:
    if component_count <= 0:
        raise ValueError("component_count must be positive.")
    return math.ceil(
        component_count
        * selected_numel
        * torch.finfo(torch.float32).bits
        / 8
        * headroom
    )
```

Thread `component_count` through `validate_host_memory`, retaining the default
of two so existing GU memory checks remain unchanged.

- [ ] **Step 4: Run buffer and GU regression tests**

Run:

```bash
pytest -q tests/test_component_buffers.py tests/test_geometric_adam.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit buffer support**

```bash
git add src/trainer/unlearn/component_buffers.py tests/test_component_buffers.py
git commit -m "feat: buffer perturbed retain gradients"
```

### Task 3: Register and validate the UAM trainer configuration

**Files:**
- Create: `src/trainer/unlearn/uam.py`
- Modify: `src/trainer/__init__.py`
- Create: `configs/trainer/UAMUnlearn.yaml`
- Create: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write failing registration and configuration tests**

```python
from omegaconf import OmegaConf

from trainer import TRAINER_REGISTRY
from trainer.unlearn.uam import UAMUnlearn


def test_uam_trainer_is_registered():
    assert TRAINER_REGISTRY["UAMUnlearn"] is UAMUnlearn


def test_uam_yaml_has_safe_defaults():
    config = OmegaConf.load("configs/trainer/UAMUnlearn.yaml")
    assert config.handler == "UAMUnlearn"
    assert config.args.save_strategy == "no"
    assert config.args.adam_beta1 == 0.0
    assert config.args.weight_decay == 0.0
    assert config.args.fp16 is False
    assert config.args.gradient_checkpointing_kwargs.use_reentrant is False
    assert config.method_args.uam_config.mode == "uam"
    assert config.method_args.uam_config.perturbation_normalization == "auto"
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_uam_trainer.py -k 'registered or yaml'`

Expected: `ModuleNotFoundError: trainer.unlearn.uam`.

- [ ] **Step 3: Add the trainer skeleton and registration**

Create:

```python
import math

from trainer.unlearn.geometric import GeometricUnlearn


class UAMUnlearn(GeometricUnlearn):
    def __init__(self, *args, **kwargs):
        self.uam_config = kwargs.pop("uam_config")
        super().__init__(*args, **kwargs)
        self.uam_mode = str(self.uam_config.mode).lower()
        self.forget_signal = str(self.uam_config.forget_signal).lower()
        normalization = str(self.uam_config.perturbation_normalization).lower()
        if normalization == "auto":
            normalization = "fixed_loss" if self.uam_mode == "uam" else "metric_trust"
        self.perturbation_normalization = normalization
        self.rho = float(self.uam_config.rho)
        self.reflection_gamma = float(self.uam_config.reflection_gamma)
        self.residual_lambda = float(self.uam_config.residual_lambda)
        self.sign_tau = float(self.uam_config.sign_tau)
        self.replay_device = str(self.uam_config.replay_device).lower()
        self.uam_calls = 0
        self.replay_calls = 0
        self.last_uam_diagnostics = {}
        self._uam_runtime_validated = False
        self._uam_microsteps = 0
        self._retain_replay_batches = []
```

Import `UAMUnlearn` in `src/trainer/__init__.py` and call
`_register_trainer(UAMUnlearn)` after `GeometricUnlearn` registration.

- [ ] **Step 4: Add the Hydra configuration**

Create `configs/trainer/UAMUnlearn.yaml` by inheriting
`GeometricUnlearn`, setting `handler: UAMUnlearn`, retaining
`save_strategy: "no"`, and adding:

```yaml
method_args:
  retain_loss_type: NLL
  uam_config:
    mode: uam
    forget_signal: nll
    perturbation_normalization: auto
    rho: 5.0e-5
    reflection_gamma: 2.0
    residual_lambda: 1.0
    sign_tau: 0.0
    replay_device: cpu
```

- [ ] **Step 5: Write failing static and runtime validation tests**

Create this fixture beside the tests:

```python
def make_uam_trainer(tmp_path, **overrides):
    model = overrides.pop("model", TinyCausalLM())
    train_dataset = overrides.pop("train_dataset", None)
    uam_config = SimpleNamespace(
        mode=overrides.pop("mode", "uam"),
        forget_signal=overrides.pop("forget_signal", "nll"),
        perturbation_normalization=overrides.pop(
            "perturbation_normalization", "auto"
        ),
        rho=overrides.pop("rho", 5e-2),
        reflection_gamma=overrides.pop("reflection_gamma", 2.0),
        residual_lambda=overrides.pop("residual_lambda", 1.0),
        sign_tau=overrides.pop("sign_tau", 0.0),
        replay_device=overrides.pop("replay_device", "cpu"),
    )
    geometric_config = SimpleNamespace(
        loss="simnpo",
        gu_enabled=True,
        gradient_surgery="gu",
        projection_eps=1e-12,
        trainable_params_regex=[".*"],
        component_buffer_device="parameter",
        diagnostics_path=None,
        actual_delta_mode="off",
        actual_delta_steps=[1, 10],
        actual_delta_sample_elements=1000,
    )
    simnpo_config = SimpleNamespace(
        delta=0.0,
        beta=4.5,
        alpha=1.0,
        gamma=0.125,
        retain_loss_type=overrides.pop("retain_loss_type", "NLL"),
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=overrides.pop("per_device_train_batch_size", 2),
        gradient_accumulation_steps=overrides.pop(
            "gradient_accumulation_steps", 1
        ),
        max_steps=overrides.pop("max_steps", -1),
        learning_rate=1e-3,
        optim="adamw_torch",
        adam_beta1=0.0,
        weight_decay=0.0,
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="no",
    )
    assert not overrides, overrides
    trainer = UAMUnlearn(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=nested_collator if train_dataset is not None else None,
        gamma=0.125,
        alpha=1.0,
        retain_loss_type=simnpo_config.retain_loss_type,
        geometric_config=geometric_config,
        uam_config=uam_config,
        simnpo_config=simnpo_config,
        npo_config=None,
        dpo_config=None,
        undial_config=None,
        wga_config=None,
        satimp_config=None,
    )
    return trainer, model
```

Parameterize invalid `mode`, `forget_signal`, normalization, replay device,
`rho <= 0`, negative lambda/tau, KL retain loss, FP16, reentrant checkpointing,
SGD, nonzero beta1, and nonzero weight decay. For each case, call
`trainer.create_optimizer()`, apply the one mutation, and assert
`_validate_uam_runtime()` raises with the exact message fragment in this table:

| Mutation | Message fragment |
|---|---|
| mode | `Unsupported UAM mode` |
| signal | `Unsupported UAM forget signal` |
| normalization | `perturbation normalization` |
| replay device | `replay device` |
| rho | `rho must be positive` |
| lambda | `residual_lambda` |
| tau | `sign_tau` |
| KL | `requires NLL` |
| FP16 | `BF16/FP32` |
| reentrant | `use_reentrant=false` |
| SGD | `SGD` |
| beta1 | `beta1=0` |
| weight decay | `weight_decay=0` |

- [ ] **Step 6: Verify validation tests RED**

Run: `pytest -q tests/test_uam_trainer.py -k validation`

Expected: tests fail because `_validate_uam_runtime` is absent.

- [ ] **Step 7: Implement `_validate_uam_runtime`**

```python
def _validate_uam_runtime(self):
    if self._uam_runtime_validated:
        return
    if self.uam_mode not in {"uam", "uam_gu"}:
        raise ValueError(f"Unsupported UAM mode: {self.uam_mode}")
    if self.forget_signal not in {"nll", "simnpo"}:
        raise ValueError(f"Unsupported UAM forget signal: {self.forget_signal}")
    if self.perturbation_normalization not in {"fixed_loss", "metric_trust"}:
        raise ValueError(
            "Unsupported UAM perturbation normalization: "
            f"{self.perturbation_normalization}"
        )
    if self.replay_device not in {"cpu", "parameter"}:
        raise ValueError(f"Unsupported UAM replay device: {self.replay_device}")
    if self.rho <= 0.0 or not math.isfinite(self.rho):
        raise ValueError("UAM rho must be positive and finite.")
    if self.residual_lambda < 0.0 or not math.isfinite(self.residual_lambda):
        raise ValueError("UAM residual_lambda must be non-negative and finite.")
    if self.sign_tau < 0.0 or not math.isfinite(self.sign_tau):
        raise ValueError("UAM sign_tau must be non-negative and finite.")
    if str(self.retain_loss_type).upper() != "NLL":
        raise NotImplementedError("UAM v1 requires NLL retain loss.")

    if self.is_deepspeed_enabled:
        raise NotImplementedError("Approximate Adam UAM does not support DeepSpeed.")
    if self.is_fsdp_enabled:
        raise NotImplementedError("Approximate Adam UAM does not support FSDP.")
    if self.args.fp16:
        raise NotImplementedError("Approximate Adam UAM supports BF16/FP32 only.")
    if self.use_apex:
        raise NotImplementedError("Approximate Adam UAM does not support Apex.")
    if self.args.world_size != 1 or self.args.n_gpu > 1:
        raise NotImplementedError(
            "Approximate Adam UAM supports one process and one GPU only."
        )
    if self.args.gradient_checkpointing:
        checkpointing_kwargs = self.args.gradient_checkpointing_kwargs or {}
        if checkpointing_kwargs.get("use_reentrant", True):
            raise NotImplementedError(
                "Approximate Adam UAM requires use_reentrant=false."
            )
    named_params = self._selected_named_parameters()
    if not named_params:
        raise ValueError("UAM requires at least one selected trainable parameter.")
    adapter = make_optimizer_geometry_adapter(self.optimizer)
    adapter.validate(named_params)
    if self.component_buffer_device == "cpu":
        ComponentGradientBuffers.validate_host_memory(
            sum(parameter.numel() for _, parameter in named_params),
            component_count=3,
        )
    self._optimizer_geometry_adapter = adapter
    self._uam_runtime_validated = True
```

- [ ] **Step 8: Run registration/config/validation tests**

Run: `pytest -q tests/test_uam_trainer.py -k 'registered or yaml or validation'`

Expected: all selected tests pass.

- [ ] **Step 9: Commit trainer contract**

```bash
git add src/trainer/unlearn/uam.py src/trainer/__init__.py \
  configs/trainer/UAMUnlearn.yaml tests/test_uam_trainer.py
git commit -m "feat: register UAM trainer contract"
```

### Task 4: Implement NLL and sign-converted SimNPO forget signals

**Files:**
- Modify: `src/trainer/unlearn/uam.py`
- Modify: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write failing signal-value and gradient-direction tests**

```python
def test_nll_signal_is_mean_answer_token_nll(tmp_path):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="nll")
    batch = make_unlearn_batch(batch_size=2, sequence_length=6, seed=7)
    signal, _ = trainer.compute_uam_forget_signal(model, batch["forget"])
    sequence_nll, _ = compute_batch_nll(model, batch["forget"])
    counts = batch["forget"]["labels"][..., 1:].ne(-100).sum(-1)
    torch.testing.assert_close(signal, (sequence_nll / counts).mean())


def test_simnpo_signal_is_negative_simnpo_objective(tmp_path):
    trainer, model = make_uam_trainer(tmp_path, forget_signal="simnpo")
    batch = make_unlearn_batch(batch_size=2, sequence_length=6, seed=9)
    signal, _ = trainer.compute_uam_forget_signal(model, batch["forget"])
    simnpo_loss, _ = trainer.compute_forget_loss(model, batch["forget"])
    torch.testing.assert_close(signal, -simnpo_loss)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_uam_trainer.py -k 'nll_signal or simnpo_signal'`

Expected: missing `compute_uam_forget_signal` failures.

- [ ] **Step 3: Implement the signal method**

```python
def compute_uam_forget_signal(self, model, forget_inputs):
    if self.forget_signal == "nll":
        sequence_nll, outputs = compute_batch_nll(model, forget_inputs)
        answer_counts = forget_inputs["labels"][..., 1:].ne(-100).sum(-1)
        if (answer_counts == 0).any():
            raise RuntimeError("UAM NLL forget batch contains an empty answer mask.")
        return (sequence_nll / answer_counts).mean(), outputs
    simnpo_loss, outputs = self.compute_forget_loss(model, forget_inputs)
    return -simnpo_loss, outputs
```

- [ ] **Step 4: Add a gradient-sign regression**

Compute gradients of NLL and `-SimNPO`; assert both have positive directional
derivative along their own gradient and that the implementation never uses
the minimized SimNPO gradient without negation.

- [ ] **Step 5: Run signal and existing SimNPO tests**

Run:

```bash
pytest -q tests/test_uam_trainer.py -k signal
pytest -q tests/test_geometric_adam.py -k component_losses
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit signal semantics**

```bash
git add src/trainer/unlearn/uam.py tests/test_uam_trainer.py
git commit -m "feat: add UAM forget signals"
```

### Task 5: Add retain replay and reversible parameter perturbation

**Files:**
- Modify: `src/trainer/unlearn/uam.py`
- Create: `tests/test_uam_replay.py`

- [ ] **Step 1: Write failing replay-tree tests**

```python
def test_replay_buffer_detaches_and_moves_nested_inputs():
    buffer = RetainReplayBuffer("cpu")
    source = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[-100, 2]])}
    buffer.append(source)
    source["input_ids"].fill_(9)
    replayed = list(buffer.batches(torch.device("cpu")))
    assert replayed[0]["input_ids"].tolist() == [[1, 2]]
    assert replayed[0]["input_ids"].requires_grad is False
    buffer.clear()
    assert buffer.empty
```

- [ ] **Step 2: Verify replay test RED**

Run: `pytest -q tests/test_uam_replay.py -k replay_buffer`

Expected: missing `RetainReplayBuffer` import.

- [ ] **Step 3: Implement `RetainReplayBuffer`**

```python
class RetainReplayBuffer:
    def __init__(self, device_mode):
        self.device_mode = device_mode
        self._batches = []

    @staticmethod
    def _map(value, function):
        if isinstance(value, dict):
            return {key: RetainReplayBuffer._map(item, function) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(RetainReplayBuffer._map(item, function) for item in value)
        if isinstance(value, list):
            return [RetainReplayBuffer._map(item, function) for item in value]
        return function(value) if isinstance(value, torch.Tensor) else value

    def append(self, inputs):
        target = torch.device("cpu") if self.device_mode == "cpu" else None
        self._batches.append(
            self._map(
                inputs,
                lambda tensor: tensor.detach().to(target or tensor.device).clone(),
            )
        )

    def batches(self, device):
        for batch in self._batches:
            yield self._map(batch, lambda tensor: tensor.to(device))

    @property
    def empty(self):
        return not self._batches

    def clear(self):
        self._batches.clear()
```

- [ ] **Step 4: Write failing exact-restoration and BF16 tests**

Use a two-parameter module, CPU snapshot storage, and requested deltas. Assert:

```python
before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
with TemporaryParameterPerturbation(named_params, deltas, storage_device="cpu") as stats:
    assert any(not torch.equal(parameter, before[name]) for name, parameter in named_params)
for name, parameter in named_params:
    assert torch.equal(parameter, before[name])
assert stats.requested_norm > 0.0
assert stats.effective_norm > 0.0
```

Add a BF16 case with a representable delta and a second case whose delta
rounds entirely to zero and raises `BF16 perturbation rounded entirely to zero`.

- [ ] **Step 5: Verify perturbation tests RED**

Run: `pytest -q tests/test_uam_replay.py -k perturbation`

Expected: missing `TemporaryParameterPerturbation` import.

- [ ] **Step 6: Implement the perturbation context manager**

```python
@dataclass
class PerturbationStats:
    requested_norm: float = 0.0
    effective_norm: float = 0.0


class TemporaryParameterPerturbation:
    def __init__(self, named_params, deltas, storage_device="cpu"):
        self.named_params = list(named_params)
        self.deltas = deltas
        self.storage_device = torch.device(storage_device)
        self.originals = {}
        self.stats = PerturbationStats()

    @torch.no_grad()
    def __enter__(self):
        requested_sq = 0.0
        effective_sq = 0.0
        for name, parameter in self.named_params:
            delta = self.deltas.get(name)
            if delta is None:
                continue
            self.originals[name] = parameter.detach().to(self.storage_device).clone()
            requested = delta.to(parameter.device, torch.float32)
            perturbed = (parameter.detach().float() + requested).to(parameter.dtype)
            effective = perturbed.float() - parameter.detach().float()
            requested_sq += float(requested.square().sum().item())
            effective_sq += float(effective.square().sum().item())
            parameter.copy_(perturbed)
        self.stats.requested_norm = math.sqrt(requested_sq)
        self.stats.effective_norm = math.sqrt(effective_sq)
        if self.stats.requested_norm > 0.0 and self.stats.effective_norm == 0.0:
            self.__exit__(None, None, None)
            raise RuntimeError(
                "BF16 perturbation rounded entirely to zero; increase rho or use FP32."
            )
        return self.stats

    @torch.no_grad()
    def __exit__(self, exc_type, exc_value, traceback):
        for name, parameter in self.named_params:
            original = self.originals.get(name)
            if original is not None:
                parameter.copy_(original.to(parameter.device, parameter.dtype))
                if not torch.equal(parameter.detach().cpu(), original.cpu()):
                    raise RuntimeError(f"UAM failed to restore parameter {name} exactly.")
        self.originals.clear()
        return False
```

- [ ] **Step 7: Run replay/perturbation tests and formatting**

Run:

```bash
pytest -q tests/test_uam_replay.py
ruff check src/trainer/unlearn/uam.py tests/test_uam_replay.py
ruff format --check src/trainer/unlearn/uam.py tests/test_uam_replay.py
```

Expected: all pass.

- [ ] **Step 8: Commit replay and perturbation support**

```bash
git add src/trainer/unlearn/uam.py tests/test_uam_replay.py
git commit -m "feat: replay retain batches under UAM perturbations"
```

### Task 6: Implement effective-batch collection and perturbed retain replay

**Files:**
- Modify: `src/trainer/unlearn/uam.py`
- Modify: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write a failing one-call-per-update lifecycle test**

```python
def test_uam_replays_once_per_optimizer_update(tmp_path):
    dataset = unbatch(make_unlearn_batch(batch_size=8, sequence_length=6, seed=21))
    trainer, _ = make_uam_trainer(
        tmp_path,
        train_dataset=dataset,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=2,
    )
    trainer.train()
    assert trainer.uam_calls == trainer.replay_calls == trainer.state.global_step == 2
    assert trainer.component_buffers.empty
    assert trainer.replay_buffer.empty
```

- [ ] **Step 2: Verify lifecycle test RED**

Run: `pytest -q tests/test_uam_trainer.py -k replays_once`

Expected: missing training/replay state behavior.

- [ ] **Step 3: Implement base component collection**

In `training_step`:

```python
self._validate_uam_runtime()
model.train()
inputs = self._prepare_inputs(inputs)
with self.compute_loss_context_manager():
    forget_signal, _ = self.compute_uam_forget_signal(model, inputs["forget"])
    retain_loss = self.compute_retain_loss(model, inputs["retain"])

named_params = self._selected_named_parameters(model)
params = [parameter for _, parameter in named_params]
forget_grads = torch.autograd.grad(
    forget_signal, params, retain_graph=True, allow_unused=True
)
retain_grads = torch.autograd.grad(
    retain_loss, params, retain_graph=True, allow_unused=True
)
self.component_buffers.add("forget", named_params, forget_grads)
self.component_buffers.add("retain", named_params, retain_grads)
self.replay_buffer.append(inputs["retain"])
self._uam_microsteps += 1
del forget_grads, retain_grads
self.accelerator.backward(retain_loss)

if self.accelerator.sync_gradients or self._is_short_final_accumulation_step():
    self._finalize_uam_gradients(named_params)
return retain_loss.detach() / self.args.gradient_accumulation_steps
```

The finalizer divides all three selected component buffers by
`self._uam_microsteps` before nonlinear decisions, making them effective-batch
means rather than sums.

- [ ] **Step 4: Implement perturbed retain replay**

Add `_replay_perturbed_retain_gradients(named_params, deltas)` that:

1. enters `TemporaryParameterPerturbation`;
2. clears `perturbed_retain`;
3. iterates `self.replay_buffer.batches(self.accelerator.device)`;
4. computes NLL retain loss and `torch.autograd.grad` for selected parameters;
5. accumulates into `perturbed_retain` without calling standard backward;
6. exits the context, restoring parameters even when replay raises;
7. divides the accumulated gradients by replay batch count;
8. increments `replay_calls` once.

- [ ] **Step 5: Add a failure-path restoration test**

Monkeypatch `compute_retain_loss` to raise on the second replay batch. Assert
the exception propagates and every parameter remains bitwise equal to its
pre-perturbation value.

- [ ] **Step 6: Run lifecycle and restoration tests**

Run:

```bash
pytest -q tests/test_uam_trainer.py -k 'replays_once or replay_failure'
```

Expected: both pass.

- [ ] **Step 7: Commit effective-batch replay path**

```bash
git add src/trainer/unlearn/uam.py tests/test_uam_trainer.py
git commit -m "feat: replay effective retain batches for UAM"
```

### Task 7: Implement streaming pure-UAM finalization

**Files:**
- Modify: `src/trainer/unlearn/uam.py`
- Modify: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write a failing deterministic pure-UAM gradient test**

Seed component buffers manually for two parameter tensors, seed a nonuniform
Adam second moment, and monkeypatch replay to provide known perturbed-retain
gradients. Compute expected optimizer coordinates with
`adapter.sqrt_denominator`, call `_finalize_uam_gradients`, and assert exact
agreement with:

```python
expected_uam = perturbed_retain - (
    reflection_gamma
    * global_dot(perturbed_retain, forget)
    / global_dot(forget, forget)
) * forget
```

Also assert `uam_calls == 1` and the component/replay buffers are empty.

- [ ] **Step 2: Verify pure-UAM test RED**

Run: `pytest -q tests/test_uam_trainer.py -k pure_uam_gradient`

Expected: finalizer is absent or does not write UAM gradients.

- [ ] **Step 3: Implement the two-pass pure-UAM finalizer**

Pass A:

- obtain raw and optimizer-coordinate forget norms;
- call `decide_perturbation`;
- materialize CPU FP32 parameter deltas one tensor at a time;
- replay perturbed retain gradients;
- compute global `dot(z_f, z_tilde_r)` and `||z_f||^2`;
- call `decide_uam`.

Pass B:

```python
forget = self._mean_component_coordinate("forget", name, parameter, sqrt_denom)
perturbed = self._mean_component_coordinate(
    "perturbed_retain", name, parameter, sqrt_denom
)
uam = apply_uam_tensor(perturbed, forget, uam_decision)
raw_gradient = self._from_adam_coordinates(uam, sqrt_denom)
if parameter.grad is None:
    parameter.grad = torch.zeros_like(parameter)
parameter.grad.copy_(raw_gradient.to(parameter.grad.dtype))
```

Wrap finalization in `try/finally` so component, replay, perturbation-delta, and
microstep state are cleared after successful writeback or a raised error.

- [ ] **Step 4: Add a paper-vs-metric perturbation test**

With the same raw forget gradient and nonuniform Adam denominator, assert
`fixed_loss` and `metric_trust` generate different parameter deltas and that
each satisfies its own invariant.

- [ ] **Step 5: Run pure-UAM and geometry regressions**

Run:

```bash
pytest -q tests/test_uam_trainer.py -k 'pure_uam or perturbation'
pytest -q tests/test_uam_geometry.py
```

Expected: all pass.

- [ ] **Step 6: Commit pure UAM**

```bash
git add src/trainer/unlearn/uam.py tests/test_uam_trainer.py
git commit -m "feat: finalize optimizer-aware UAM gradients"
```

### Task 8: Implement Residual-GU-UAM and the sign gate

**Files:**
- Modify: `src/trainer/unlearn/uam.py`
- Modify: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write failing residual orthogonality and gate tests**

Create manual component buffers whose UAM residual has both retain-tangent and
retain-normal components. After finalization assert:

```python
diagnostics = trainer.last_uam_diagnostics
assert diagnostics["mode"] == "uam_gu"
assert diagnostics["residual_gate_kept"] is True
assert diagnostics["relative_residual_orthogonality"] < 1e-6
assert diagnostics["residual_forget_gate_dot"] < 0.0
assert diagnostics["predicted_retain_directional_derivative"] < 0.0
assert diagnostics["predicted_forget_correction_derivative"] > 0.0
```

Add a second fixture with positive gate dot; assert final selected gradients
equal ordinary retain gradients and `residual_gate_kept is False`.

- [ ] **Step 2: Verify residual tests RED**

Run: `pytest -q tests/test_uam_trainer.py -k residual`

Expected: missing residual diagnostics and wrong final gradients.

- [ ] **Step 3: Add global residual scalar pass**

When `uam_mode == "uam_gu"`, compute over all selected tensors:

```python
residual = uam - retain
residual_retain_dot += (residual * retain).sum()
residual_forget_dot += (residual * forget).sum()
forget_retain_dot += (forget * retain).sum()
retain_sq += retain.square().sum()
```

Call `decide_residual_gu` once globally.

- [ ] **Step 4: Write residual final gradients**

In the streaming writeback pass:

```python
final_coordinates, normal = apply_residual_gu_tensor(
    uam,
    retain,
    residual_decision,
)
```

Accumulate normal-retain dot, normal norm, forget-normal dot, and final
directional derivatives for diagnostics before converting back to raw
gradient.

- [ ] **Step 5: Add actual-update safety test**

Run one optimizer step on `TinyCausalLM`, record parameters before and after,
and compare the correction relative to a retain-only step. Assert the
first-order retain dot of the correction is below `1e-6` relative error and the
forget-signal dot of the correction is positive whenever the gate is kept.

- [ ] **Step 6: Run Residual-GU-UAM tests**

Run: `pytest -q tests/test_uam_trainer.py -k 'residual or actual_update'`

Expected: all pass.

- [ ] **Step 7: Commit UAM+GU**

```bash
git add src/trainer/unlearn/uam.py tests/test_uam_trainer.py
git commit -m "feat: project the UAM residual through GU"
```

### Task 9: Prove gradient-accumulation equivalence

**Files:**
- Modify: `tests/test_uam_trainer.py`
- Modify: `src/trainer/unlearn/uam.py`

- [ ] **Step 1: Write failing GAS equivalence tests for both modes**

Parameterize `mode` over `uam` and `uam_gu`. Use identical initial
`TinyCausalLM` copies and an eight-example dataset. Train one update with
`batch=8,GAS=1` and `batch=2,GAS=4`, with identical seeded Adam states.
Compare:

```python
assert full.last_uam_diagnostics["replay_microsteps"] == 1
assert accumulated.last_uam_diagnostics["replay_microsteps"] == 4
assert full.last_uam_diagnostics["uam_coefficient"] == pytest.approx(
    accumulated.last_uam_diagnostics["uam_coefficient"], rel=1e-5, abs=1e-6
)
for left, right in zip(full.model.parameters(), accumulated.model.parameters()):
    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Verify GAS tests RED**

Run: `pytest -q tests/test_uam_trainer.py -k gas_equivalence`

Expected: mismatch in coefficients or parameter updates.

- [ ] **Step 3: Correct mean scaling and replay ordering**

Ensure all component accesses divide raw accumulated tensors by the same
`_uam_microsteps`, all replay batches use the one shared perturbation, and the
Adam adapter state is read before any optimizer step. Do not project or reflect
inside a microstep.

- [ ] **Step 4: Add short-window lifecycle coverage**

Use two examples with `GAS=4`, one training step, and assert one finalization,
one replay, two cached microsteps, empty buffers, and finite diagnostics.

- [ ] **Step 5: Run GAS and existing GU GAS suites**

Run:

```bash
pytest -q tests/test_uam_trainer.py -k 'gas or short_window'
pytest -q tests/test_geometric_adam.py -k 'gradient_accumulation or short_final'
```

Expected: all pass.

- [ ] **Step 6: Commit GAS correctness**

```bash
git add src/trainer/unlearn/uam.py tests/test_uam_trainer.py
git commit -m "fix: apply UAM to effective-batch gradients"
```

### Task 10: Persist UAM diagnostics and actual parameter deltas

**Files:**
- Modify: `src/trainer/unlearn/uam.py`
- Modify: `src/trainer/unlearn/gu_diagnostics.py`
- Modify: `tests/test_uam_trainer.py`

- [ ] **Step 1: Write a failing diagnostics-schema test**

Train one update with a JSONL path and `actual_delta_mode=full`. Assert record
types `uam_geometry` then `actual_delta`, and require these geometry keys:

```python
required = {
    "update_step", "mode", "forget_signal", "perturbation_normalization",
    "rho", "requested_perturbation_norm", "effective_perturbation_norm",
    "effective_perturbation_ratio", "forget_norm", "retain_norm",
    "perturbed_retain_norm", "uam_coefficient", "residual_tangent_norm",
    "residual_normal_norm", "relative_residual_orthogonality",
    "residual_forget_gate_dot", "residual_gate_kept",
    "predicted_forget_directional_derivative",
    "predicted_retain_directional_derivative",
    "identity_fallback_parameters", "replay_microsteps",
    "uam_calls", "replay_calls",
}
assert required <= records[0].keys()
assert records[0]["uam_calls"] == records[0]["replay_calls"] == 1
```

- [ ] **Step 2: Verify diagnostics test RED**

Run: `pytest -q tests/test_uam_trainer.py -k diagnostics_schema`

Expected: missing keys or wrong record type.

- [ ] **Step 3: Emit the UAM geometry record**

Set `record_type="uam_geometry"`, populate every required scalar with finite
Python floats, assign both `last_uam_diagnostics` and
`last_surgery_diagnostics`, and write through `SurgeryDiagnosticsWriter`.

- [ ] **Step 4: Generalize summary aggregation without breaking PCGrad**

Change `aggregate_surgery_records` so geometry records include both
`record_type in {"geometry", "uam_geometry"}`. Preserve existing GU summary
keys and add optional UAM means only when values are present:

```python
"mean_effective_perturbation_ratio": mean_present(
    geometry_records, "effective_perturbation_ratio"
),
"residual_gate_rate": mean_present(
    geometry_records, "residual_gate_kept"
),
```

- [ ] **Step 5: Prepare the actual-delta callback before clearing buffers**

Call:

```python
self.actual_delta_callback.prepare_step(
    self.uam_calls,
    named_params,
    self.component_buffers,
)
```

after final gradients and diagnostics are ready but before component buffers
are cleared. The callback continues to interpret `forget` as the ascent signal
and `retain` as base retain NLL.

- [ ] **Step 6: Run diagnostic and GU regression tests**

Run:

```bash
pytest -q tests/test_uam_trainer.py -k 'diagnostics or actual_delta'
pytest -q tests/test_geometric_pcgrad.py -k 'diagnostics or actual_delta'
```

Expected: all pass and no checkpoint directories under the test output.

- [ ] **Step 7: Commit diagnostics**

```bash
git add src/trainer/unlearn/uam.py src/trainer/unlearn/gu_diagnostics.py \
  tests/test_uam_trainer.py
git commit -m "feat: record UAM perturbation and update diagnostics"
```

### Task 11: Add BF16, FlashAttention 2, and PagedAdamW32 integration

**Files:**
- Modify: `tests/test_gu_gpu_integration.py`
- Create: `tests/test_uam_gpu_integration.py`

- [ ] **Step 1: Write a failing BF16 eager-vs-Flash replay test**

Reuse `make_llama` and fixed synthetic batches from
`test_gu_gpu_integration.py`. Starting from identical weights, compute NLL
forget gradients and perturbed retain gradients with eager and
`flash_attention_2`; assert finite values and BF16-appropriate agreement.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_uam_gpu_integration.py -k flash`

Expected: missing UAM GPU fixture/helper failures.

- [ ] **Step 3: Implement test helpers through the public trainer path**

Create two-step `UAMUnlearn` fixtures with `save_strategy="no"`,
`use_reentrant=false`, representable test `rho`, and mode parameterization.
Do not bypass `_finalize_uam_gradients` when checking gradients.

- [ ] **Step 4: Add a Torch AdamW BF16 update test**

Assert two optimizer updates complete, `uam_calls == replay_calls == 2`, all
diagnostics are finite, effective perturbation ratio is positive, and no
checkpoint payload exists.

- [ ] **Step 5: Add a real PagedAdamW32 BF16 update test**

Use `pytest.importorskip("bitsandbytes")`. Instantiate the same
`bitsandbytes.optim.adamw.AdamW` class and kwargs emitted by Transformers for
`paged_adamw_32bit`, validate `state2` FP32 after initialization, and run the
second update through UAM.

- [ ] **Step 6: Run GPU tests**

Run:

```bash
pytest -q tests/test_gu_gpu_integration.py tests/test_uam_gpu_integration.py
```

Expected: all GPU tests pass; hardware/library skips must be explicit rather
than silently counted as passes.

- [ ] **Step 7: Commit GPU support**

```bash
git add tests/test_gu_gpu_integration.py tests/test_uam_gpu_integration.py
git commit -m "test: validate UAM on BF16 Flash and PagedAdamW32"
```

### Task 12: Add no-checkpoint four-arm smoke launchers and analyzer

**Files:**
- Create: `scripts/uam_smoke_arm.sh`
- Create: `scripts/uam_smoke_matrix.sh`
- Create: `scripts/analyze_uam_smoke.py`
- Create: `tests/test_uam_scripts.py`

- [ ] **Step 1: Write failing launcher-contract tests**

Read launcher text and assert it contains:

```python
for token in (
    "trainer=UAMUnlearn",
    "trainer.args.optim=paged_adamw_32bit",
    "trainer.args.bf16=true",
    "model.model_args.attn_implementation=flash_attention_2",
    "trainer.args.gradient_checkpointing_kwargs.use_reentrant=false",
    "save_model_after_train=false",
    "trainer.args.save_strategy=no",
    "trainer.args.max_steps=10",
):
    assert token in arm_text
```

Assert the matrix contains exactly the four approved method specifications
and the audit searches for safetensors, model/training `.bin`, optimizer,
scheduler, RNG, trainer state, and `.ckpt` payloads.

- [ ] **Step 2: Verify launcher tests RED**

Run: `pytest -q tests/test_uam_scripts.py -k launcher`

Expected: scripts do not exist.

- [ ] **Step 3: Implement `uam_smoke_arm.sh`**

Accept `<method> <gpu> <timestamp>`, where method maps exactly as follows:

```bash
case "${method}" in
    uam_nll)
        mode=uam; signal=nll; normalization=fixed_loss ;;
    uam_simnpo)
        mode=uam; signal=simnpo; normalization=fixed_loss ;;
    uam_gu_nll)
        mode=uam_gu; signal=nll; normalization=metric_trust ;;
    uam_gu_simnpo)
        mode=uam_gu; signal=simnpo; normalization=metric_trust ;;
    *) exit 2 ;;
esac
```

Use TOFU forget01, Llama-3.2-1B, local `/tmp/uam_smoke` staging, one GPU,
BF16, FlashAttention 2, PagedAdamW32, GAS 8, ten updates, diagnostics at steps
1 and 10, evaluation enabled, and `save_model_after_train=false`.

- [ ] **Step 4: Implement atomic artifact audit and persistence**

Copy only run log, Hydra config, JSONL/summary diagnostics, and
`TOFU_SUMMARY.json` to `saves/exp/UAM_SMOKE/<timestamp>`. Run the checkpoint
audit both before and after persistence.

- [ ] **Step 5: Implement the two-GPU matrix scheduler**

Start from the tested scheduler mechanically:

```bash
cp scripts/pcgrad_smoke_matrix.sh scripts/uam_smoke_matrix.sh
```

Then set `matrix_root="saves/exp/UAM_SMOKE/${timestamp}"`, replace the arm
construction with:

```bash
arms=(uam_nll uam_simnpo uam_gu_nll uam_gu_simnpo)
```

and make `launch_arm` execute:

```bash
local command="bash scripts/uam_smoke_arm.sh ${method} ${slot} ${timestamp}"
bash scripts/uam_smoke_arm.sh "${method}" "${slot}" "${timestamp}" &
```

Retain the tested fail-fast rule: after any nonzero exit code, do not schedule
a new arm, wait for every already-active PID, append all completed rows, and
exit nonzero. The manifest header is exactly:

```text
pid\tgpu\tmethod\tstart_utc\tend_utc\texit_code\tcommand
```

- [ ] **Step 6: Write failing analyzer tests**

Create fake four-arm outputs and assert the analyzer:

- requires four successful manifest rows;
- requires one TOFU summary and UAM diagnostics per arm;
- rejects metric-key drift;
- reports perturbation ratios, gate rate, replay/finalizer counts, and metric
  deltas relative to `uam_nll`;
- records zero checkpoint payloads;
- emits single-newline-terminated Markdown and sorted JSON.

- [ ] **Step 7: Implement the analyzer**

Define `METHODS = ("uam_nll", "uam_simnpo", "uam_gu_nll",
"uam_gu_simnpo")` and implement top-level functions named `read_manifest`,
`find_summary`, `read_diagnostics`, `checkpoint_payloads`, `analyze_matrix`,
`render_markdown`, and `write_outputs`.

`read_manifest` requires the seven declared columns, exactly one successful
row for each method, and no extra method. `find_summary` requires exactly one
`TOFU_SUMMARY.json`. `read_diagnostics` requires ten
`record_type="uam_geometry"` records and validates every key listed in Task
10. `checkpoint_payloads` scans the same payload patterns as the launcher.
`analyze_matrix` checks identical metric keys, stores raw metrics, computes
every method minus `uam_nll`, and records mean perturbation ratio, mean
residual gate rate, maximum residual orthogonality, finalizer count, replay
count, manifest provenance, and payload count. `render_markdown` emits those
values in method and metric tables. `write_outputs` uses sorted, indented JSON
and exactly one final newline for both files.

- [ ] **Step 8: Run script tests and Bash validation**

Run:

```bash
pytest -q tests/test_uam_scripts.py
bash -n scripts/uam_smoke_arm.sh scripts/uam_smoke_matrix.sh
```

Expected: all pass.

- [ ] **Step 9: Commit smoke infrastructure**

```bash
git add scripts/uam_smoke_arm.sh scripts/uam_smoke_matrix.sh \
  scripts/analyze_uam_smoke.py tests/test_uam_scripts.py
git commit -m "feat: add no-checkpoint UAM smoke matrix"
```

### Task 13: Run the four-arm smoke matrix and generate the report

**Files:**
- Create: `UAM_SMOKE_REPORT.md`
- Runtime artifacts: `saves/exp/UAM_SMOKE/<timestamp>/`

- [ ] **Step 1: Verify GPUs and caches without changing external state**

Run:

```bash
conda activate unlearning
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
test -d /root/.cache/huggingface
```

Expected: two usable GPUs and an accessible Hugging Face cache.

- [ ] **Step 2: Launch the matrix**

Run:

```bash
timestamp=$(date -u +%m%d%H%M%S)
bash scripts/uam_smoke_matrix.sh "${timestamp}"
```

Expected: four manifest rows with exit code zero.

- [ ] **Step 3: Audit outputs**

Run the arm script's `--audit-only` mode against both the persistent matrix
root and `/tmp/uam_smoke/<timestamp>`. Assert four summaries, four Hydra
configs, four diagnostics JSONL files, and zero checkpoint payloads.

- [ ] **Step 4: Generate analysis and report**

Run:

```bash
python scripts/analyze_uam_smoke.py \
  "saves/exp/UAM_SMOKE/${timestamp}" \
  --markdown UAM_SMOKE_REPORT.md \
  --json "saves/exp/UAM_SMOKE/${timestamp}/analysis.json"
```

- [ ] **Step 5: Inspect the mechanism criteria**

Require:

- one UAM finalization and replay per optimizer update;
- finite, nonzero effective perturbation ratios;
- residual orthogonality below `1e-5` when the UAM+GU gate is kept;
- actual-delta forget and retain directional derivatives present at steps 1
  and 10;
- no superiority statement because seed count is one.

- [ ] **Step 6: Commit the generated report**

```bash
git add UAM_SMOKE_REPORT.md
git commit -m "docs: report UAM and residual GU smoke results"
```

### Task 14: Final verification and independent review

**Files:**
- All task files above

- [ ] **Step 1: Run the full CPU regression suite**

Run:

```bash
pytest -q \
  tests/test_uam_geometry.py \
  tests/test_uam_replay.py \
  tests/test_uam_trainer.py \
  tests/test_uam_scripts.py \
  tests/test_component_buffers.py \
  tests/test_optimizer_geometry.py \
  tests/test_gradient_surgery.py \
  tests/test_geometric_adam.py \
  tests/test_geometric_pcgrad.py \
  tests/test_analyze_pcgrad_smoke.py \
  tests/test_pcgrad_scripts.py
```

Expected: zero failures.

- [ ] **Step 2: Run the full GPU integration suite**

Run:

```bash
pytest -q tests/test_gu_gpu_integration.py tests/test_uam_gpu_integration.py
```

Expected: zero failures; any skip must identify a genuinely unavailable GPU
feature.

- [ ] **Step 3: Run static and artifact checks**

Run:

```bash
ruff check src/trainer/unlearn/uam.py src/trainer/unlearn/uam_geometry.py \
  src/trainer/unlearn/component_buffers.py src/trainer/unlearn/gu_diagnostics.py \
  tests/test_uam_geometry.py tests/test_uam_replay.py tests/test_uam_trainer.py \
  tests/test_uam_gpu_integration.py tests/test_uam_scripts.py \
  scripts/analyze_uam_smoke.py
ruff format --check src/trainer/unlearn/uam.py src/trainer/unlearn/uam_geometry.py \
  src/trainer/unlearn/component_buffers.py src/trainer/unlearn/gu_diagnostics.py \
  tests/test_uam_geometry.py tests/test_uam_replay.py tests/test_uam_trainer.py \
  tests/test_uam_gpu_integration.py tests/test_uam_scripts.py \
  scripts/analyze_uam_smoke.py
bash -n scripts/uam_smoke_arm.sh scripts/uam_smoke_matrix.sh
git diff --check abfe95f HEAD
```

Expected: all checks pass.

- [ ] **Step 4: Request independent review**

Review `abfe95f..HEAD` against the approved design. Require explicit checks of:

- NLL/SimNPO sign convention;
- fixed-loss normalization mismatch labeling;
- frozen-Adam coordinate transforms;
- effective-batch replay and scaling;
- parameter restoration on success and exception;
- residual-only projection and gate inequality;
- PagedAdamW32 state timing;
- BF16 perturbation quantization;
- actual update diagnostics and zero-checkpoint audit.

- [ ] **Step 5: Fix every Critical/Important finding test-first**

For each valid finding, add a failing regression, verify RED, apply the minimal
fix, verify GREEN, and commit. Re-request focused review until no
Critical/Important item remains.

- [ ] **Step 6: Re-run final verification after review fixes**

Repeat Steps 1-3 and regenerate the report/analysis to byte-compare against
the committed report and stored JSON.

- [ ] **Step 7: Hand off without merging or pushing**

Report the branch, worktree, commit range, test counts, smoke root, report,
geometry conclusions, limitations, and zero checkpoint payload count. Keep
`dev0` and its user-owned dirty files unchanged.
