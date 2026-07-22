# Minimal Correct GU Design

**Status:** Approved for implementation planning  
**Branch:** `feat/GU`  
**Base:** `dev0`

## Goal

Provide one readable implementation of Geometric Unlearning that constrains the
real parameter update produced by AdamW. GU must work with every existing
`UnlearnTrainer` objective without reimplementing or dispatching objective
losses.

The feature is correct relative to an explicit protected functional: the
answer-masked retain NLL on the current retain batch and an optional FIFO of
previous retain covectors.

## Non-Goals

- RMU-, UAM-, PCGrad-, OrthoGrad-, or PDU-specific GU variants.
- Adaptive layer selection, Fisher geometry, or a general QP framework.
- DDP, FSDP, DeepSpeed, FP16, or arbitrary optimizers.
- Finite-step safety claims when the first-order mode is selected.

Existing methods and performance knobs remain available. The legacy
`GeometricUnlearn` trainer remains for PCGrad and its other shipped behavior,
but its GU mode must fail with a direct migration message so there is only one
authoritative GU path.

## User-Facing Configuration

GU is an optional branch of the common `UnlearnTrainer`; the selected objective
trainer remains unchanged.

```yaml
trainer: NPO  # Any existing UnlearnTrainer subclass.

trainer:
  method_args:
    gu:
      enabled: true
      parameter_regex:
        - 'model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight'
      retain_history_rank: 8
      projection_eps: 1.0e-12
      retain_filter: first_order  # first_order | finite_step
      retain_budget: 1.0e-4
      backtracking_scales: [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
      diagnostics_path: gu_diagnostics.jsonl
```

`enabled: false` must produce the same one-step update as the objective trainer
without GU. `parameter_regex` is required and must select at least one optimizer
parameter. Parameters outside the selected scope are frozen.

The protected functional is fixed to answer-masked retain NLL in this minimal
version. It does not inherit an objective-specific retain surrogate.

## Traceable Execution Path

```text
Hydra trainer=<existing objective>, method_args.gu.enabled=true
  -> TRAINER_REGISTRY[objective trainer]
  -> UnlearnTrainer.training_step
  -> accumulate current retain covector over GAS microsteps
  -> AdamW step pre-hook snapshots selected parameters
  -> ordinary AdamW.step
  -> AdamW step post-hook reads realized parameter delta
  -> SafeProx correction in frozen Adam geometry
  -> optional actual retain-loss backtracking
  -> corrected parameters and one JSONL record
```

The GU branch lives directly in `UnlearnTrainer`. It does not dynamically wrap
trainer classes, delegate objective computation, or introduce a runtime,
manager, projector, solver, or callback abstraction.

Framework-required `training_step` and optimizer hook methods are execution
entry points, not convenience helpers. Other helpers are allowed only when they
contain nontrivial logic and have at least three production call sites.

## Candidate Update

Let the selected parameters immediately before and after ordinary AdamW be
`theta_before` and `theta_adam`. The proposal is

```math
\Delta_t^0 = \theta_{\mathrm{adam}} - \theta_{\mathrm{before}}.
```

It therefore includes the effects of gradient clipping, momentum, adaptive
normalization, weight decay, and optimizer history. GU does not infer this
proposal from gradients.

The optimizer state remains the ordinary AdamW proposal state, including when
finite-step filtering scales the parameter delta to zero. Each subsequent
realized parameter delta is constrained again, so safety is defined on applied
parameter updates rather than on an imagined modified Adam trajectory. The
diagnostic record names this state policy explicitly.

Supported optimizers are Torch AdamW and shipped 32-bit paged AdamW variants
whose selected state exposes finite `exp_avg_sq`. Eight-bit optimizer state is
rejected.

## Retain Constraints

During `training_step`, GU computes the answer-masked retain NLL covector for
the selected parameters using `autograd.grad`. FP32 covectors are accumulated
over the effective batch and finalized only when
`accelerator.sync_gradients` is true.

The current covector is always included. Up to `retain_history_rank` previous
globally normalized covectors are stored in FIFO order. Positive normalization
does not change a half-space constraint. Historical covectors are not
orthogonalized because arbitrary basis rotations do not preserve a cone.

The default history rank is 8. Rank 0 disables history but still enforces the
current retain constraint. Values above 8 are rejected because the chosen
small active-set solve is intentionally bounded.

## Adam-Metric SafeProx

Let `G_t` contain the current and historical retain covectors. The safety set is

```math
G_t^T \Delta \preceq 0.
```

After AdamW updates its second moment, freeze

```math
M_t = \operatorname{Diag}(\sqrt{v_t} + \epsilon).
```

GU solves

```math
\Delta_t^\star = \arg\min_\Delta
\frac12\lVert\Delta-\Delta_t^0\rVert_{M_t}^2
\quad\text{subject to}\quad G_t^T\Delta\preceq0.
```

For active multipliers `lambda`,

```math
\Delta_t^\star = \Delta_t^0 - M_t^{-1}G_t\lambda,
\qquad
K_t = G_t^TM_t^{-1}G_t.
```

There are at most nine constraints. The implementation enumerates active sets,
solves the corresponding FP64 Gram systems, and selects the feasible solution
with minimum objective. It then verifies primal feasibility, nonnegative duals,
complementarity, and a bounded KKT residual before changing parameters.

If the proposal is already safe, it is applied unchanged and recorded with a
zero correction ratio.

## Retain Filter Modes

### First-order mode (default)

Apply the SafeProx result directly. The supported claim is only

```math
G_t^T\Delta_t^\star\preceq0.
```

No finite-step retain-loss guarantee is made.

### Finite-step mode

Evaluate the actual retain NLL at the configured scales in descending order and
apply the largest scale satisfying

```math
u(\theta_t+s\Delta_t^\star)-u(\theta_t)\le B.
```

`B` is `retain_budget`. If no scale is feasible, restore the pre-step selected
parameters, apply a zero step, and record the reason. This mode uses actual
forward evaluations rather than an HVP approximation.

There is no exact closed-form line search for a general neural-network retain
loss. Under a local quadratic model with
`a = grad(u)^T Delta` and `c = Delta^T H_u Delta`, the positive-curvature root
would be

```math
s = \min\left(1,\frac{-a+\sqrt{a^2+2cB}}{c}\right),
```

but this is not a finite-step certificate without a curvature upper bound.
The minimal implementation therefore does not add an HVP path.

## Diagnostics

Each optimizer update writes exactly one JSON object containing:

- optimizer step and objective trainer name;
- selected parameter and constraint counts;
- proposal norm, corrected norm, and correction ratio;
- active constraints and maximum violation before and after correction;
- KKT residual;
- selected finite-step scale;
- retain loss before and after when finite-step mode is enabled;
- zero-step status and reason;
- projection and filter wall time.

Diagnostics are observability, not control flow. Missing or unwritable requested
diagnostics fail before training begins.

## Runtime Validation

Initialization rejects:

- an optimizer other than supported 32-bit AdamW variants;
- DDP, FSDP, DeepSpeed, Apex, or FP16;
- reentrant gradient checkpointing;
- an empty parameter selection;
- selected parameters absent from optimizer groups;
- trainable parameters outside the selected scope;
- invalid history rank, epsilon, budget, scales, or diagnostics path.

A step fails without changing selected parameters when retain covectors,
optimizer state, Gram systems, corrected deltas, or KKT checks are nonfinite or
invalid.

## Tests

One focused test module must cover:

1. Hydra flag to executed GU branch and diagnostic record.
2. Every registered `UnlearnTrainer` subclass without objective reimplementation.
3. Exact baseline recovery when GU is disabled.
4. Realized deltas containing momentum, adaptive denominator, and weight decay.
5. Actual parameter-delta safety for current and historical constraints.
6. Safe proposals remaining unchanged.
7. Active-set and KKT correctness, including dependent constraints.
8. FIFO history and mandatory current constraint.
9. Equivalent effective batches under GAS 1 and GAS greater than 1.
10. Finite-step selection of the largest actually feasible scale.
11. Safe zero step when no scale is feasible.
12. Fail-closed unsupported modes and invalid state.
13. One diagnostic record per optimizer update.
14. No model, optimizer, checkpoint, or trainer-state artifacts.

A single-GPU BF16 integration test verifies the complete Trainer execution path
with Torch AdamW. Unit tests use actual parameter changes, not only projected
intermediate tensors.

## Code Standards

- Human readability is the first priority; end-to-end traceability from CLI to
  branch, delta, metric, and test is mandatory.
- Production implementation should add no more than 600 lines and must be
  redesigned before exceeding 750 lines.
- Control-flow nesting is at most three levels.
- Prefer deletion and direct code over adapters, factories, or one-call wrappers.
- Existing features, performance knobs, and observability are not disabled to
  reduce line count.
- Comments explain only non-obvious reasons in two to four lines. They contain
  no run identifiers, commit hashes, single-run metrics, or internal paths.
- A defensive branch is added only when a shipped recipe can reach it.

## Acceptance Criteria

- The pre-change `dev0` baseline remains green.
- All new focused and integration tests pass in `conda activate unlearning`.
- Shipped GU recipes use the single `gu.enabled` path.
- Legacy `gradient_surgery=gu` fails with a direct migration message.
- The diff contains no duplicate objective loss implementation.
- A reader can trace one optimizer update in source order without following a
  callback or runtime framework.
