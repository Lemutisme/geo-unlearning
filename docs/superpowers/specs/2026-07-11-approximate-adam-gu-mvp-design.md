# Approximate Adam/AdamW Geometric Unlearning MVP Design

**Date:** 2026-07-11
**Status:** Approved design; awaiting written-spec review

## 1. Goal

Replace the unreachable GU implementation on `dev0` with a reachable,
testable Stage-A implementation for standard Torch Adam and AdamW. The new
path must obtain the forget and retain gradients independently, accumulate
them over the effective batch, project them globally in a frozen
preconditioner-induced geometry, write the result to `p.grad`, and leave
gradient clipping and `optimizer.step()` to Transformers 4.45.1.

This version is deliberately labelled **approximate optimizer-aware GU**. A
later, separate optimizer implementation will address the denominator update,
momentum, weight decay, and strict safety of the realized parameter delta.

## 2. Verified Root Cause

Transformers 4.45.1 calls `training_step()`, performs gradient clipping, and
then calls `self.optimizer.step()` directly. It does not define or call a
Trainer `optimizer_step()` hook. The current `GeometricUnlearn.optimizer_step()`
therefore never participates in training. If called manually, it also reaches
an undefined `project_current_grads()` method or a nonexistent parent method.

The current code additionally:

- reconstructs a forget gradient from a total gradient and a separately
  computed retain gradient;
- differentiates the same retain graph twice;
- projects each parameter tensor independently rather than using one global
  parameter-space coefficient;
- applies a full Adam-like whitening map instead of square-root coordinate
  maps;
- uses a reversed sign-selective condition; and
- silently inherits unsupported DeepSpeed, distributed, PagedAdamW, and
  reentrant-checkpointing configurations.

## 3. Scope

### In scope

- Current `dev0` only. The existing `codex/correct-geometric-unlearning`
  worktree and branch remain untouched.
- Standard `torch.optim.Adam` and `torch.optim.AdamW`, including an
  Accelerate wrapper around either optimizer.
- A frozen, pre-step second-moment metric with square-root transforms.
- `beta1=0`, `weight_decay=0`, single-process BF16 or FP32 training, and
  non-reentrant gradient checkpointing.
- Independent component gradients and correct gradient accumulation.
- One global rank-one retain constraint over all selected tensors.
- SimNPO with NLL retain loss as the primary validation objective.
- A geometry-disabled compatibility path for baseline recovery.
- A dedicated single-GPU validation entry point that does not modify the
  existing two-GPU `scripts/gu_eval.sh` workflow.

### Out of scope

- Strict guarantees for the realized Adam/AdamW parameter delta.
- Custom optimizer behavior, momentum correction, or freezing the denominator
  inside the optimizer step.
- Nonzero Adam first-moment momentum or weight decay.
- bitsandbytes PagedAdamW, DeepSpeed, FSDP, DDP, FP16, Apex, or reentrant
  checkpointing.
- Historical low-rank retain bases, per-parameter bases, sign-selective
  tangents, and optimizer-aware Stage B.
- KL retain-objective changes. Existing objective selection remains explicit;
  the SimNPO validation uses NLL.

## 4. Approaches Considered

### Frozen square-root Adam geometry

Read the optimizer's pre-step `exp_avg_sq`, construct square-root coordinate
maps, project in those coordinates, map back to a raw gradient, and let the
standard optimizer proceed. Enforce `beta1=0` and `weight_decay=0`. This is the
selected approach.

### Raw Euclidean projection under Adam

Project raw gradients directly and then call AdamW. This is simpler but does
not represent the optimizer-induced update geometry. Rejected.

### Standard Adam momentum with heuristic correction

Keep `beta1>0` and try to compensate for historical momentum before the step.
Without owning the optimizer update, this cannot establish a reliable safety
claim. Deferred to the custom-optimizer phase.

## 5. Component Losses and Configuration

`GeometricUnlearn` resolves the selected method configuration exactly once
during initialization. For SimNPO validation the resolved values are:

```text
gamma = 0.125
alpha = 1.0
retain_loss_type = NLL
beta = 4.5
delta = 0.0
```

The resolved values are passed through the existing `GradDiff` initialization
path so reference-model creation, when required by another configured
objective, observes the final retain-loss type. `compute_loss()` must never
mutate these coefficients.

The trainer exposes:

```python
compute_component_losses(model, inputs)
```

which returns the unscaled forget loss, unscaled retain loss, and forget
outputs. `compute_loss()` combines them only as

```python
gamma * forget_loss + alpha * retain_loss
```

for evaluation and compatibility paths. The SimNPO forget component preserves
the existing sequence-normalized loss formula, and its retain component is the
configured NLL objective.

`geometric_config.gu_enabled=false` skips all projection logic while retaining
the same component implementation and optimizer. A one-step comparison against
the native `SimNPO` trainer must recover the same scalar loss, gradients, and
parameter update.

## 6. Selected Parameters

Selection uses `geometric_config.trainable_params_regex`. A parameter is
selected only when it requires gradients and its full name matches at least one
configured expression. An empty selection is a configuration error.

The validation configuration uses `.*`, so every trainable parameter belongs
to the one global geometry. Unselected parameters, when a future experiment
uses a narrower expression, retain the ordinary total-loss gradient produced
by the standard backward pass. Any safety diagnostic and claim is limited to
the selected parameters.

## 7. Training-Step Data Flow

For every microbatch, `GeometricUnlearn.training_step()` performs the following
operations:

1. Validate the runtime before doing model work.
2. Prepare inputs and compute the forget and retain component losses once.
3. Average losses for `n_gpu>1` only for signature compatibility; runtime
   validation rejects multi-process execution.
4. Use `torch.autograd.grad()` to obtain selected-parameter forget and retain
   gradients independently, retaining the graph until the normal backward.
5. Scale both component gradients by
   `1 / gradient_accumulation_steps`, matching Accelerate 0.34.2, and add
   detached FP32 copies to raw-gradient buffers.
6. Call `self.accelerator.backward(total_loss)`. This populates ordinary
   `.grad` values for all parameters and preserves the normal behavior for
   unselected parameters.
7. When `self.accelerator.sync_gradients` is true, transform the accumulated
   selected gradients, project and combine them globally, overwrite selected
   `.grad` values, increment the projection counter, record diagnostics, and
   clear both buffers.
8. Return the detached loss divided by the configured accumulation count, as
   Transformers 4.45.1 expects.

The outer Trainer loop then performs its normal global gradient clipping,
callbacks, and optimizer step. Projection is never performed independently on
each microbatch because projection does not commute with accumulation.

## 8. Accumulation Buffers

The trainer owns two dictionaries keyed by selected parameter name:

```python
_gu_forget_buffer: dict[str, torch.Tensor]
_gu_retain_buffer: dict[str, torch.Tensor]
```

Each stored tensor is detached FP32 on the parameter's device. A buffer entry
is created lazily when the corresponding component gradient is not `None`.
Buffers are cleared immediately after successful finalization. FP16 is not
supported, so there is no GradScaler overflow path in this MVP.

An empty forget or retain buffer at finalization is an error. A zero retain
gradient is valid and leaves the forget component unchanged after the
epsilon-protected coefficient calculation.

## 9. Frozen Adam/AdamW Geometry

For a parameter in optimizer group `j`, let the pre-step optimizer state contain
second moment `v` and completed-step count `s`. When `s>0`, define

$$
\hat v=\frac{v}{1-\beta_2^s},
\qquad
h=\sqrt{\hat v}+\epsilon,
\qquad
P=\operatorname{Diag}(h^{-1}).
$$

The square-root maps are

$$
z=P^{1/2}g=\frac{g}{\sqrt h},
\qquad
g=P^{-1/2}z=\sqrt h\,z.
$$

If a parameter has no initialized state, its map is the identity and the
fallback is included in diagnostics. The implementation uses each optimizer
group's own `beta2` and `eps`, rejects AMSGrad, and rejects missing,
shape-incompatible, or non-finite initialized state.

After transforming the accumulated raw component gradients, compute one
coefficient across every selected tensor:

$$
c=\frac{\sum_\ell\langle z_f^{(\ell)},z_r^{(\ell)}\rangle}
        {\max\!\left(\sum_\ell\lVert z_r^{(\ell)}\rVert_2^2,10^{-12}\right)}.
$$

Then

$$
z_f^\perp=z_f-cz_r,
\qquad
z_{\mathrm{GU}}=\gamma z_f^\perp+\alpha z_r,
\qquad
g_{\mathrm{GU}}=P^{-1/2}z_{\mathrm{GU}}.
$$

The frozen-preconditioner forget update obeys

$$
g_r^\top(-\eta P g_f^\star)
=-\eta z_r^\top z_f^\perp
\approx0,
$$

while the retain term is non-increasing to first order. The standard optimizer
subsequently updates its second moment using `g_GU`, so its actual denominator
differs from the frozen one. This is the precise reason the implementation and
diagnostics use the label `approximate`.

## 10. Runtime Validation

Projection-enabled training fails before its first forward pass unless all of
the following hold:

- DeepSpeed and FSDP are disabled;
- `world_size == 1`;
- FP16 and Apex are disabled;
- the underlying optimizer is `torch.optim.Adam` or `torch.optim.AdamW`;
- every optimizer group has `beta1 == 0`, `weight_decay == 0`, and
  `amsgrad == false`;
- gradient checkpointing is disabled or explicitly configured with
  `use_reentrant=false`;
- `gamma > 0`, `alpha >= 0`, and selected parameters are nonempty.

The wrapper is unwrapped only far enough to inspect the underlying standard
Torch optimizer. PagedAdamW and other state layouts are rejected rather than
guessed.

The dedicated validation run uses one A100, FP32 model parameters,
`adamw_torch`, `adam_beta1=0`, `weight_decay=0`, non-reentrant checkpointing,
and all trainable parameters selected.

## 11. Diagnostics

Every finalized effective batch records at least:

- projection call count;
- global projection coefficient;
- transformed forget and retain norms;
- transformed dot product before and after projection;
- relative orthogonality residual;
- number of parameters using identity-metric fallback; and
- the explicit mode label `approximate_adam_stage_a`.

The projection call count must equal `trainer.state.global_step` after training.
No diagnostic field may describe the realized standard-Adam delta as certified
or exact.

## 12. Test Strategy

All implementation changes follow red-green-refactor.

### Pure geometry

- Square-root forward/inverse closure for nonuniform denominators.
- Identity fallback for uninitialized optimizer state.
- Global decomposition and transformed-space orthogonality.
- A two-tensor counterexample proving blockwise coefficients differ from the
  required global coefficient.
- The frozen-preconditioner forget update has retain directional derivative
  below `1e-6` relative error in FP32.

### Trainer behavior

- The unused `optimizer_step()` method and undefined projection call are gone.
- One projection occurs per actual optimizer update.
- GAS=1 with batch 8 and GAS=4 with batch 2 produce equivalent accumulated
  GU gradients and parameter updates under a frozen test metric.
- Selected gradients are overwritten before Trainer clipping; unselected
  gradients retain the ordinary total-loss value.
- Component gradients are obtained independently and are never reconstructed
  from a total gradient.
- Buffers clear after each finalized effective batch.

### Configuration and compatibility

- Initialization resolves SimNPO coefficients once.
- `gu_enabled=false` matches the native SimNPO one-step loss and update.
- DeepSpeed, FSDP, multi-process, FP16, PagedAdamW, nonzero `beta1`, nonzero
  weight decay, AMSGrad, reentrant checkpointing, and empty parameter selection
  each fail with a targeted message.
- A toy standard AdamW step records the approximate label and does not assert a
  false strict-safety threshold on the realized delta.

## 13. SimNPO Validation Protocol

Run two new arms from
`open-unlearning/tofu_Llama-3.2-1B-Instruct_full` on TOFU `forget01` and
`retain99`:

1. `GeometricUnlearn` with `gu_enabled=false`;
2. `GeometricUnlearn` with `gu_enabled=true`.

Both arms use seed 0, FP32, Torch AdamW, `beta1=0`, `weight_decay=0`, ten
optimizer updates, and effective batch size 32. The same base checkpoint,
example order, objective coefficients, clipping threshold, and scheduler apply
to both. Outputs use new task and experiment directories and never overwrite
the existing saved models or evaluations.

Each trained model receives the existing full TOFU evaluation. The resulting
summaries are compared with each other and displayed alongside these historical
files:

- `saves/exp/baseline07102348/tofu_Llama-3.2-1B-Instruct_forget01_base_SimNPO/TOFU_SUMMARY.json`
- `saves/exp/GU/07102323/tofu_Llama-3.2-1B-Instruct_forget01_GU_SimNPO/TOFU_SUMMARY.json`

The historical runs are contextual only. They used two-process DeepSpeed,
PagedAdamW, BF16, and `weight_decay=0.01`; moreover, the historical baseline
used SimNPO `gamma=0.125` while the historical GU run used `gamma=0.2`. Their
metric differences therefore cannot be attributed to geometric projection.

The new control-versus-GU comparison is the valid experimental contrast.
Metric improvement is an empirical observation, not a prerequisite for the
mathematical and lifecycle tests to pass.

## 14. Completion Criteria

- All new unit and integration tests pass in the exact dependency versions
  pinned by the repository.
- The reachable training path projects exactly once per optimizer update.
- GAS and baseline-recovery tests pass.
- Frozen-metric transformed-space orthogonality meets the FP32 test tolerance.
- Unsupported configurations fail before training.
- Both ten-step SimNPO arms and their full TOFU evaluations complete in new
  output directories.
- The final report distinguishes verified frozen-metric properties, observed
  realized AdamW behavior, historical results, and deferred strict-optimizer
  work.
