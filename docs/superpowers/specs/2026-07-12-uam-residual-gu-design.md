# UAM and Residual-GU-UAM Design

**Date:** 2026-07-12  
**Branch:** `feat/uam-gu`  
**Base:** `dev0@abfe95f`  
**Status:** Approved for implementation

## 1. Goal

Implement two checkpoint-free unlearning update rules on the real Transformers
4.45.1 training path:

1. `uam`: Unlearning-Aware Minimization using the efficient projected retain
   gradient from Kim et al., NeurIPS 2025.
2. `uam_gu`: retain descent plus a GU-projected UAM residual correction.

The implementation must support Torch Adam/AdamW and bitsandbytes
PagedAdamW32, FP32/BF16, gradient accumulation, and FlashAttention 2. The
first version remains single-process and uses the same frozen-preconditioner
approximation as the existing GU implementation.

## 2. Scope and Non-Goals

### In scope

- token-NLL and SimNPO-derived forget signals;
- the paper's fixed-linearized-loss-increase perturbation normalization;
- an optimizer-metric trust-region perturbation normalization;
- efficient UAM without Hessian construction;
- residual-only GU projection with a forget-benefit sign gate;
- effective-batch gradient accumulation through retain-batch replay;
- exact parameter restoration after temporary perturbation;
- diagnostics for geometry, perturbation quantization, and actual updates;
- ten-step Llama-3.2-1B smoke runs with no checkpoint payloads.

### Out of scope

- the exact Hessian term in UAM equation (12);
- optimizer momentum correction or a custom Adam implementation;
- DeepSpeed, FSDP, DDP, FP16, or reentrant gradient checkpointing;
- a historical low-rank retain basis;
- WMDP/RMU representation losses, which require a separate strict-WMDP
  extension after this trainer is validated;
- superiority claims from single-seed smoke runs.

## 3. Architecture

The work is isolated on `feat/uam-gu`, while `dev0` remains the completed
PCGrad branch. The UAM branch inherits the tested component-buffer,
optimizer-geometry, diagnostics, PagedAdamW32, and FlashAttention support from
`dev0`, but introduces a separate trainer and configuration.

### New components

- `src/trainer/unlearn/uam_geometry.py`
  - pure tensor/global-vector operations;
  - perturbation normalization;
  - UAM projected/reflected retain gradient;
  - residual GU projection and sign gate.
- `src/trainer/unlearn/uam.py`
  - component forward/gradient collection;
  - accumulation-window retain input replay;
  - temporary parameter perturbation and restoration;
  - optimizer-coordinate conversion and final gradient writeback;
  - runtime validation and diagnostics.
- `configs/trainer/UAMUnlearn.yaml`
  - defaults for both update modes and both forget signals.

`UAMUnlearn` subclasses `GeometricUnlearn` only to reuse objective parsing,
selected-parameter logic, optimizer geometry, diagnostics callbacks, and
runtime infrastructure. It overrides the training/update path rather than
adding UAM branches to the existing GU/PCGrad finalizer.

## 4. Configuration Contract

```yaml
handler: UAMUnlearn

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

Allowed values:

- `mode`: `uam` or `uam_gu`;
- `forget_signal`: `nll` or `simnpo`;
- `perturbation_normalization`: `auto`, `fixed_loss`, or `metric_trust`;
- `replay_device`: `cpu` or `parameter`.

`auto` resolves to:

- `fixed_loss` for `uam`;
- `metric_trust` for `uam_gu`.

The defaults do not silently reuse `gamma` or `alpha` from SimNPO.
`reflection_gamma` is UAM's projection/reflection coefficient and
`residual_lambda` controls only the residual correction. UAM v1 requires an
NLL retain objective.

## 5. Loss and Sign Convention

Let `F` be the forget signal, always defined so that a larger value means more
forgetting:

### NLL signal

\[
F(w)=L_{\mathrm{NLL}}(w;D_f),
\qquad g_f=\nabla F(w).
\]

The inner perturbation follows `+g_f`, increasing forget NLL.

### SimNPO signal

The existing SimNPO loss is minimized to unlearn. Therefore UAM uses

\[
F(w)=-L_{\mathrm{SimNPO}}(w;D_f),
\qquad g_f=-\nabla L_{\mathrm{SimNPO}}(w;D_f).
\]

This makes the perturbation and sign gate use the same “larger is more
forgotten” convention as NLL.

The retain objective is

\[
R(w)=L_{\mathrm{NLL}}(w;D_r),
\qquad g_r=\nabla R(w).
\]

## 6. Frozen Optimizer Geometry

For one optimizer update, freeze the pre-step diagonal Adam preconditioner
`P_t`. Define optimizer coordinates

\[
z=P_t^{1/2}g.
\]

For a raw Adam denominator `D_t`, `P_t=D_t^{-1}` and the existing adapter
computes

\[
z=D_t^{-1/2}g.
\]

An empty optimizer state uses the identity as a documented first-step
fallback. The approximation does not account for momentum, so runtime
validation requires `beta1=0` and `weight_decay=0`.

## 7. Inner Perturbation

Two normalizations are supported.

### Fixed linearized loss increase

The UAM paper writes

\[
\delta_{\mathrm{fixed}}
=\rho\frac{g_f}{\lVert g_f\rVert_2^2+\epsilon}.
\]

This keeps the first-order loss increase approximately equal to `rho`. It is
not the solution of the stated Euclidean radius-constrained problem; the
implementation and diagnostics call it `fixed_loss`, not `l2_radius`.

### Optimizer-metric trust region

The optimizer-aware trust-region perturbation is

\[
\delta_{\mathrm{metric}}
=\rho\frac{P_tg_f}
{\sqrt{g_f^\top P_tg_f+\epsilon}}.
\]

It satisfies a radius constraint in the metric induced by `P_t^{-1}`.

If the forget norm is zero, UAM cannot define an inner maximizer and fails
closed. It does not silently become retain-only fine-tuning.

## 8. UAM Update

Temporarily evaluate retain data at `w + delta` and define

\[
\widetilde g_r=\nabla R(w+\delta),
\qquad
z_f=P_t^{1/2}g_f,
\qquad
\widetilde z_r=P_t^{1/2}\widetilde g_r.
\]

The efficient UAM update in frozen optimizer coordinates is

\[
z_{\mathrm{UAM}}
=\widetilde z_r
-\gamma
\frac{\langle\widetilde z_r,z_f\rangle}
{\lVert z_f\rVert^2+\epsilon}z_f.
\]

`reflection_gamma=1` removes the forget component. The paper default
`reflection_gamma=2` is a Householder reflection. Because this branch applies
the transformation in frozen Adam coordinates, it is an optimizer-aware
approximation of the paper's Euclidean equation (15), not the exact paper
optimizer dynamics.

For `mode=uam`, selected parameters receive `z_UAM`. Non-selected parameters
receive the ordinary retain gradient from the standard backward path.

## 9. Residual-GU-UAM Update

Define the UAM-specific gradient residual

\[
q=z_{\mathrm{UAM}}-z_r,
\qquad z_r=P_t^{1/2}g_r.
\]

Only this residual is projected:

\[
q^\perp
=q-
\frac{\langle q,z_r\rangle}
{\lVert z_r\rVert^2+\epsilon}z_r.
\]

The candidate correction step is `-q_perp`. It is retained only when it
increases the forget signal:

\[
\langle z_f,q^\perp\rangle<-\tau.
\]

The final gradient coordinate is

\[
z_{\mathrm{final}}
=z_r+\lambda q^\perp
\]

when the gate passes, and `z_final = z_r` otherwise. Therefore the correction
does not change retain loss to first order:

\[
\langle z_r,-\lambda q^\perp\rangle=0.
\]

This design intentionally does not project the complete UAM update. Directly
projecting the total update would delete ordinary retain descent and can
reverse the forgetting direction when the UAM alignment is already negative.

If `z_r` has zero norm in `uam_gu` mode, the retain constraint is undefined
and the trainer fails closed.

## 10. Gradient Accumulation and Replay

UAM is nonlinear in the effective-batch forget gradient. Per-microbatch UAM
followed by summation is not equivalent to UAM on the accumulated gradient.
The trainer therefore uses one replay operation per optimizer update:

1. At each microstep at the unchanged parameters `w`:
   - compute `F` and `R`;
   - accumulate selected `g_f` and `g_r` in FP32;
   - execute the standard retain backward for non-selected parameters;
   - cache a detached copy of the retain inputs.
2. At the optimizer boundary:
   - form the effective-batch mean `g_f` and `g_r`;
   - freeze `P_t`;
   - construct and temporarily apply `delta` once;
   - replay every cached retain microbatch at the same `w + delta`;
   - accumulate the effective-batch mean `tilde_g_r`;
   - restore parameters in a `finally` block;
   - compute UAM or Residual-GU-UAM and overwrite selected `param.grad`.
3. Clear component and replay buffers before Trainer clipping and
   `optimizer.step()`.

The first equivalence contract uses equal-sized microbatches, matching the
current GU tests. Unequal final microbatches retain standard Trainer weighting
semantics and are reported separately rather than claimed to be identical to
a concatenated batch.

## 11. Temporary Perturbation and BF16

FP32 parameters apply the requested perturbation directly. BF16 parameters
apply the nearest representable perturbation. For every step, diagnostics
record

\[
\frac{\lVert\delta_{\mathrm{effective}}\rVert}
{\lVert\delta_{\mathrm{requested}}\rVert+\epsilon}.
\]

If all selected BF16 perturbations round to zero, the trainer raises an error
with instructions to increase `rho` or use FP32. Parameter restoration is
preflight-checked tensor by tensor and verified after replay. A restoration
mismatch is a hard error.

Retain replay inputs default to CPU storage to bound GPU memory. The
`parameter` option is available for small tests and performance comparisons.

## 12. Diagnostics

Each optimizer step records:

- update step, mode, forget signal, and perturbation normalization;
- requested and effective perturbation norms and their ratio;
- `g_f`, `g_r`, and `tilde_g_r` norms;
- UAM coefficient and pre/post alignment;
- residual tangent and normal norms;
- residual-retain orthogonality;
- sign-gate dot product and gate decision;
- predicted forget and retain directional derivatives;
- identity-preconditioner fallback count;
- replay microstep count and surgery/replay call counters.

The existing actual-delta callback measures post-optimizer parameter changes
for selected steps. No diagnostic may require a model checkpoint.

## 13. Runtime Validation

UAM v1 rejects:

- DeepSpeed and FSDP;
- more than one process or GPU;
- FP16 and Apex;
- reentrant gradient checkpointing;
- optimizers other than Torch Adam/AdamW and PagedAdamW32;
- Adam `beta1 != 0`, nonzero weight decay, or AMSGrad;
- PagedAdamW state whose `state2` is not FP32;
- non-NLL retain objectives;
- invalid mode, signal, normalization, or replay-device values;
- nonpositive `rho`, negative `residual_lambda`, or negative `sign_tau`.

Trainer clipping and optimizer stepping remain owned by Transformers after
`training_step()` returns.

## 14. Test Contract

### Geometry unit tests

1. Fixed-loss perturbation has first-order increase `rho`.
2. Metric perturbation has optimizer-metric norm `rho`.
3. `reflection_gamma=2` performs the expected Householder reflection.
4. The residual projection is globally orthogonal to `z_r`.
5. The sign gate retains only a correction whose actual step increases `F`.
6. Direct total-update projection deletes retain descent, while the residual
   construction preserves it.
7. Zero forget and zero retain norms fail according to the runtime contract.

### Trainer tests

1. One perturb/replay/finalization call occurs per optimizer update.
2. `batch=8,GAS=1` and `batch=2,GAS=4` produce equivalent selected gradients
   and updates for equal-sized microbatches.
3. Parameters are bitwise equal before perturbation and after restoration.
4. NLL and sign-converted SimNPO signals point in the documented directions.
5. UAM and UAM+GU write the expected selected gradients while non-selected
   parameters follow retain descent.
6. Actual parameter deltas satisfy the predicted directional signs.
7. No `checkpoint-*` payload is created.

### GPU integration tests

1. Torch AdamW FP32/eager.
2. Torch AdamW BF16/FlashAttention 2.
3. PagedAdamW32 BF16/FlashAttention 2.
4. BF16 perturbation diagnostics are finite and nonzero for the test `rho`.

## 15. Smoke Matrix

After unit and GPU tests pass, run four ten-step Llama-3.2-1B arms:

| Arm | Mode | Forget signal | Perturbation |
|---|---|---|---|
| UAM-NLL | `uam` | `nll` | `fixed_loss` |
| UAM-SimNPO | `uam` | `simnpo` | `fixed_loss` |
| UAM-GU-NLL | `uam_gu` | `nll` | `metric_trust` |
| UAM-GU-SimNPO | `uam_gu` | `simnpo` | `metric_trust` |

All arms use one GPU, BF16, PagedAdamW32, FlashAttention 2,
`use_reentrant=false`, `beta1=0`, and `weight_decay=0`. Save strategy is `no`.
The launch script audits for `*.safetensors`, `*.bin`, `*.pt`, and `*.pth`
payloads and fails if any are present.

These runs are execution and mechanism checks. They do not establish
statistical superiority.

## 16. Acceptance Criteria

The implementation is complete when:

- all existing GU/PCGrad tests still pass on the UAM branch;
- every new geometry and trainer test has passed through a red-green cycle;
- CPU and GPU integration suites pass;
- independent review reports no unresolved Critical or Important finding;
- all four smoke arms exit successfully;
- diagnostics contain exactly one UAM finalization per optimizer update;
- the artifact audit finds zero checkpoint payloads;
- the report labels fixed-loss normalization and frozen-Adam geometry
  honestly and does not claim exact arbitrary-optimizer guarantees.
