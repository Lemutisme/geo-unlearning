# PCGrad Multi-Dataset and Systems Compatibility Design

**Date:** 2026-07-11
**Branch:** `dev0`
**Status:** Approved design; awaiting written-spec review

## 1. Goal

Extend the reachable GeometricUnlearn training path on `dev0` with a
retain-prioritized PCGrad mode, PagedAdamW32 optimizer-state support, and BF16
FlashAttention 2 execution. Use matched no-checkpoint experiments on TOFU and
MUSE to determine whether the current GU result is specific to one split or is
caused by persistent forget-retain gradient conflict.

The implementation must preserve the existing component-gradient data flow:

```text
component forward
-> independent effective-batch forget/retain gradients
-> frozen optimizer-coordinate gradient surgery
-> write selected parameter.grad
-> Trainer clipping
-> optimizer.step
```

## 2. Branch and Scope Boundaries

### In scope on `dev0`

- Existing `none` and `gu` paths plus a new `pcgrad` surgery mode.
- SimNPO with NLL retain loss on:
  - TOFU forget01/retain99/holdout01;
  - TOFU forget05/retain95/holdout05;
  - TOFU forget10/retain90/holdout10;
  - MUSE News;
  - MUSE Books.
- Torch Adam/AdamW and bitsandbytes PagedAdamW32.
- FP32 eager reference execution.
- BF16 model execution with FlashAttention 2.
- FP32 component buffers and frozen optimizer-coordinate geometry.
- Single-process, single-GPU execution with non-reentrant gradient
  checkpointing.
- In-memory evaluation without model or Trainer checkpoint payloads.

### Out of scope on `dev0`

- UAM, UAM-paper, UAM-code, Residual-GU-UAM, and constrained residual QP.
  These will receive a separate specification and branch.
- WMDP/RMU representation objectives. They belong to the future UAM branch.
- PagedAdamW8 or any quantized optimizer state.
- A nonexistent PagedAdamW16 state format. In this design, "16-bit" means
  BF16 model computation, while PagedAdamW state remains FP32.
- FP16 GradScaler, DeepSpeed, FSDP, DDP, Apex, momentum, and weight decay.
- Strict safety claims for the realized standard Adam/PagedAdamW parameter
  delta. Both remain approximate optimizer-aware modes.

## 3. Mathematical Definition

Let both configured components be minimization objectives:

\[
g_f=\nabla J_f(w),\qquad g_r=\nabla L_r(w).
\]

For SimNPO, \(J_f\) is the existing sequence-normalized SimNPO forget loss,
not the raw forget-data NLL. Let \(P_t\) be the frozen pre-step diagonal Adam
preconditioner and define square-root coordinates

\[
z_f=P_t^{1/2}g_f,\qquad z_r=P_t^{1/2}g_r.
\]

Retain-prioritized PCGrad modifies only the auxiliary forget gradient:

\[
z_f^{\mathrm{PC}}=
\begin{cases}
z_f-
\dfrac{z_f^\top z_r}{\lVert z_r\rVert_2^2+\varepsilon}z_r,
& z_f^\top z_r<0,\\[2mm]
z_f,&z_f^\top z_r\ge0.
\end{cases}
\]

The optimizer-equivalent combined coordinate gradient is

\[
z_{\mathrm{final}}=\gamma z_f^{\mathrm{PC}}+\alpha z_r,
\]

which is mapped back with \(P_t^{-1/2}\) before being copied to
`parameter.grad`.

This is deliberately asymmetric: retain is the primary objective and is never
projected. The original symmetric randomized PCGrad algorithm is not used.

## 4. Exact Relationship to Current GU

Current rank-one GU always removes the retain component:

\[
z_f^{\mathrm{GU}}
=z_f-
\frac{z_f^\top z_r}{\lVert z_r\rVert_2^2+\varepsilon}z_r.
\]

Therefore,

\[
z_f^\top z_r<0
\quad\Longrightarrow\quad
z_f^{\mathrm{PC}}=z_f^{\mathrm{GU}}.
\]

All ten projection coefficients in the first TOFU forget01 run were negative.
PCGrad would consequently have produced the same effective gradient as GU on
every update in that run. PCGrad is not assumed to improve that result. Its
scientific purpose is to test whether other splits and domains enter a
non-conflicting regime in which conditional surgery differs from conservative
GU.

The implementation and report must not describe PCGrad as an improvement if
the measured PCGrad-GU update distance is numerically zero.

## 5. Code Architecture

### 5.1 Surgery strategy

Move gradient combination behind a small strategy interface selected by:

```yaml
geometric_config:
  gradient_surgery: none  # none | gu | pcgrad
```

The strategy receives effective-batch FP32 coordinate dictionaries and
returns the final coordinate gradient plus diagnostics. Component loss
calculation, GAS buffers, optimizer-coordinate transforms, and gradient
write-back remain shared.

`none` must delegate to the native Trainer training step to preserve exact
SimNPO baseline recovery. `gu` preserves the current unconditional global
rank-one projection. `pcgrad` uses the conditional global projection defined
above.

### 5.2 Optimizer-state adapter

Replace direct Torch-state access with adapters exposing:

```python
step(parameter) -> int | None
second_moment(parameter) -> Tensor | None
prefetch(parameter) -> None
validate_group(group) -> None
```

Adapters:

- Torch Adam/AdamW: `exp_avg_sq`, Tensor or scalar `step`.
- bitsandbytes PagedAdamW32: FP32 `state2`, integer `step`, and optimizer
  `prefetch_state(parameter)` when the paged buffer requires it.

Only a genuinely empty state receives the first-step identity fallback. A
nonempty partial, shape-incompatible, negative, non-finite, or quantized state
is rejected. Paged `state2.dtype == torch.uint8` is an explicit PagedAdamW8
error.

Bias correction uses each parameter group's beta2 and step. Every group must
have beta1 zero, weight decay zero, AMSGrad disabled, and positive finite eps.

### 5.3 Memory-bounded component buffers

All-trainable-parameter MUSE runs use a 7B model. Two full FP32 component
buffers cannot remain on an 80GB GPU alongside model weights, ordinary
gradients, activations, and optimizer state. The buffer implementation must
therefore support:

```yaml
geometric_config:
  component_buffer_device: cpu  # cpu | parameter
```

For MUSE, detached component gradients are accumulated in pinned CPU FP32
buffers. Forget gradients are accumulated and released before retain
gradients are computed, so the two autograd-gradient tuples are never live
simultaneously. Final global dots and write-back use parameter-wise streaming
rather than materializing a second full-model transformed copy.

The host currently has about 1.7 TiB available memory, which is sufficient for
two concurrent MUSE arms. Each launcher must nevertheless estimate two FP32
selected-parameter buffers plus 20% headroom and fail before model loading if
available host memory is lower than that estimate.

TOFU-1B may use parameter-device buffers for the FP32 reference, but the
algorithm must be numerically equivalent across buffer devices within the
declared tolerance.

## 6. BF16 and FlashAttention 2

FlashAttention 2 is a forward/backward kernel choice and does not change the
gradient-surgery definition. Supported production execution is:

```text
BF16 model parameters and autocast
+ FlashAttention 2
+ FP32 component/surgery buffers
+ PagedAdamW32 or Torch AdamW
```

The model loader already rejects FlashAttention 2 with FP32. FP16 remains
unsupported because the current component-gradient path has no coordinated
GradScaler overflow handling. Non-reentrant checkpointing remains mandatory
because independent `autograd.grad` calls must share the forward graph.

## 7. Runtime Validation and Failure Behavior

Projection-enabled runs fail before the first model forward unless:

- exactly one process and at most one CUDA device are active;
- DeepSpeed, FSDP, DDP, Apex, and FP16 are disabled;
- checkpointing is disabled or `use_reentrant=false`;
- the optimizer is Torch Adam/AdamW or PagedAdamW32;
- beta1 and weight decay are zero in every group;
- selected trainable parameters are present in the optimizer;
- buffer-device configuration is supported;
- FlashAttention 2 is paired with BF16 model dtype;
- the surgery mode is one of `none`, `gu`, or `pcgrad`.

An empty retain gradient is an error. A numerically zero retain norm leaves the
forget gradient unchanged and records `zero_retain_norm=true`. No fallback may
silently switch PagedAdamW to Euclidean geometry after its state is
initialized.

## 8. Diagnostics

Every optimizer update in GU or PCGrad mode records:

- surgery mode and call count;
- global transformed dot and cosine before surgery;
- conflict decision (`dot < 0`);
- projection coefficient;
- transformed dot and cosine after surgery;
- forget, retain, and final coordinate norms;
- relative surgery magnitude
  \(\lVert z_f^{\mathrm{surgery}}-z_f\rVert/
    (\lVert z_f\rVert+\varepsilon)\);
- relative PCGrad-GU distance when PCGrad mode is active;
- identity-fallback count and optimizer-adapter name;
- component buffer device;
- frozen-metric predicted retain/forget directional derivatives.

Actual post-optimizer delta diagnostics require a pre-step parameter snapshot
and therefore are not collected over every full-model MUSE update. Full actual
delta checks run in toy tests and the TOFU systems-isolation run at steps 1 and
10. MUSE uses a fixed, configuration-recorded probe of parameter elements at
steps 1 and 10; those results are labelled sampled diagnostics rather than
full-model guarantees.

Run summaries aggregate conflict rate, maximum residual, mean relative surgery
magnitude, and mean PCGrad-GU distance. If conflict rate is one and distance is
below tolerance, the report states that PCGrad empirically degenerated to GU.

## 9. Experiment Design

### 9.1 Algorithm smoke matrix

Use the same model, data order, seed, precision, optimizer, batch, GAS,
learning-rate schedule, and evaluation implementation within each dataset.
Run ten optimizer updates with seed 0:

| Dataset setting | Control | GU | PCGrad |
|---|---:|---:|---:|
| TOFU forget01 | yes | yes | yes |
| TOFU forget05 | yes | yes | yes |
| TOFU forget10 | yes | yes | yes |
| MUSE News | yes | yes | yes |
| MUSE Books | yes | yes | yes |

All algorithm-matrix arms use the production system configuration: BF16,
FlashAttention 2, PagedAdamW32, beta1 zero, weight decay zero, and FP32 surgery
buffers. Control uses the same GeometricUnlearn objective with surgery disabled
and is already unit-checked against native SimNPO.

### 9.2 Systems isolation

On TOFU forget01 with PCGrad and seed 0, compare ten updates under:

1. FP32, eager attention, Torch AdamW;
2. BF16, FlashAttention 2, Torch AdamW;
3. BF16, FlashAttention 2, PagedAdamW32.

This isolates numerical/kernel and optimizer-adapter changes from the
cross-method comparison.

### 9.3 Advancement rule

Advance a dataset setting to a full ten-epoch, three-seed run only if its smoke
run satisfies at least one:

- PCGrad conflict rate is below 0.95; or
- mean relative PCGrad-GU update distance exceeds `1e-4`.

If all five settings have conflict rate one and distance at most `1e-6`, stop
the PCGrad expansion and report a cross-dataset degeneracy result instead of
spending the full experiment budget.

Full runs, when triggered, use seeds 0, 1, and 2 and retain the repository's
ten-epoch dataset budgets.

## 10. Evaluation

TOFU reports the full existing suite, with primary interpretation based on:

- forget QA probability and ROUGE;
- extraction strength and exact memorization;
- MIA AUC distance from 0.5;
- model utility and retain extraction strength.

MUSE reports its full existing suite, with primary interpretation based on:

- forget knowledge/verb-memory ROUGE;
- retain knowledge-memory ROUGE;
- extraction and exact memorization;
- MIA metrics and privacy leakage.

Every result table includes raw values, GU/control and PCGrad/control deltas,
PCGrad/GU deltas, configuration differences, and the number of seeds. A
single-seed smoke result is mechanism evidence only, not a performance claim.

## 11. Test Strategy

All behavior changes use red-green-refactor.

### Pure geometry

- Negative dot: PCGrad is transformed-space orthogonal and exactly equals GU.
- Nonnegative dot: PCGrad leaves the forget gradient unchanged while GU does
  not generally do so.
- Global multi-tensor coefficient differs from blockwise surgery.
- Zero retain gradient is finite and unchanged.
- CPU and parameter-device buffer results agree.

### Optimizer adapters

- Torch Adam and AdamW initialized/noninitialized states.
- PagedAdamW32 state2 mapping after a real CUDA optimizer step.
- Paged state prefetch and parameter-group lookup.
- Quantized, negative, non-finite, partial, wrong-shape, and non-positive-step
  states fail closed.
- Heterogeneous optimizer groups use their own beta2 and eps.

### Trainer lifecycle

- One surgery call per optimizer update, including short GAS windows.
- GAS 1 and GAS 4 agree under initialized nonuniform metrics.
- Component gradients are accumulated independently.
- Selected gradients are replaced before Trainer clipping.
- Unselected parameters preserve ordinary total-loss gradients.
- Buffers clear after every finalized update and on handled exceptions.
- Full actual parameter deltas satisfy the declared toy frozen-metric checks.
- `none` recovers the native SimNPO update.

### GPU integration

- BF16 eager and BF16 FlashAttention 2 component gradients agree within
  documented BF16 tolerance.
- Torch AdamW and PagedAdamW32 adapters produce finite surgery and the expected
  mode labels.
- A short MUSE-7B run stays within A100-80GB memory with CPU component buffers.

## 12. Artifact Policy

No experiment writes model weights, optimizer state, scheduler state, RNG
state, or Trainer state. Evaluation runs on the in-memory trained model. A
post-arm audit fails the experiment if any checkpoint payload is found.

Evaluation directories may retain JSON, logs, Hydra configuration snapshots,
and evaluator namespaces named `checkpoint-<step>`; those namespaces must not
contain checkpoint payloads.

## 13. Claims and Non-Claims

This work may claim that:

- PCGrad executes on the real Trainer path;
- conditional surgery differs from GU only outside the conflict regime;
- the measured conflict regime does or does not generalize across the selected
  TOFU and MUSE settings;
- PagedAdamW32 and FlashAttention 2 run under the declared approximate mode.

It may not claim that:

- PCGrad improves GU when their recorded updates are equal;
- PagedAdamW provides strict optimizer-aware retain invariance;
- one seed or ten updates establish benchmark superiority;
- MUSE results imply WMDP/RMU compatibility;
- this branch implements or validates UAM.
