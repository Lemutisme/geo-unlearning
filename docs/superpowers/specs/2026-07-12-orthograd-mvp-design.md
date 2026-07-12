# Full-Parameter OrthoGrad MVP and GU Overhead Design

**Date:** 2026-07-12
**Branch:** `feat/orthograd`
**Base:** `dev0@abfe95f`
**Paper:** Shamsian et al., “Go Beyond Your Means: Unlearning with
Per-Sample Gradient Orthogonalization,” arXiv:2503.02312v2
**Status:** Approved design; implementation plan pending

## 1. Goal

Implement a full-parameter OrthoGrad minimum viable path for TOFU and compare
its feasibility, compute cost, GPU memory, and ten-step behavior with the
repository’s SimNPO control and GU implementation.

The comparison answers two narrow questions:

1. Can exact per-sample retain-gradient orthogonalization run on the
   Llama-3.2-1B TOFU stack without LoRA on one A100-80GB?
2. What wall-clock, per-step, projection, and memory overhead do GU and
   OrthoGrad add over a matched SimNPO control?

The experiment is a single-seed, ten-update feasibility study. It must not be
used to claim performance superiority.

## 2. Paper Semantics and Repository Mapping

For an unlearn gradient `g_u` and per-sample retain gradients
`G_r = [g_r^1, ..., g_r^k]`, the paper computes an orthonormal basis `Q` for
the column space of `G_r` and projects

\[
g_u^\perp = g_u - QQ^\top g_u.
\]

Its optimizer gradient is

\[
g = \alpha \bar g_r - (1-\alpha)g_u^\perp,
\qquad
\bar g_r = \frac{1}{k}\sum_i g_r^i,
\]

where `g_u` is the gradient of a loss that is ascended.

The controlled GU comparison uses the existing SimNPO objective, whose
gradient is already a descent-direction unlearning candidate. Define that
candidate as `g_f`. The implemented update is therefore

\[
g_{\text{OrthoGrad}}
= g_r + 0.125\,P_{\operatorname{span}\{g_r^i\}}^\perp g_f.
\]

Its direction corresponds to the paper’s `alpha = 8/9`; the global factor
`1.125` is absorbed into the update scale so that the raw retain and forget
weights exactly match the repository’s GU weights `alpha=1` and
`gamma=0.125`. This isolates average-retain rank-one projection versus
per-sample retain-subspace projection.

This MVP uses the paper’s Euclidean geometry. It is labelled
`orthograd_euclidean_approx_adam` because PagedAdamW32 does not preserve an
exact Euclidean projection after updating optimizer state. Existing GU remains
labelled as its frozen-Adam approximation. The report must not conflate their
guarantees.

## 3. Scope

### Supported

- one process on one CUDA device;
- BF16 or FP32;
- FlashAttention 2 or eager attention;
- Torch AdamW or PagedAdamW32 with beta1 0 and weight decay 0;
- non-reentrant gradient checkpointing;
- all trainable parameters selected;
- retain batch size from 1 through 4;
- gradient accumulation equal to 1;
- SimNPO forget candidate and NLL retain loss.

### Rejected at runtime

- FP16, DeepSpeed, FSDP, Apex, DDP, or multiple visible training processes;
- reentrant checkpointing;
- nonzero Adam momentum or weight decay;
- gradient accumulation greater than 1;
- partial parameter selection;
- retain batches above the configured maximum;
- empty answers, non-finite gradients, or an empty retain basis.

The GAS restriction is intentional. Summing microbatch-wise projections is not
equivalent to projecting the effective-batch gradient against the union of all
per-sample retain gradients. Full GAS support is outside the MVP.

## 4. Architecture

### 4.1 Trainer isolation

Add `OrthogradUnlearn` as a separate subclass of `GeometricUnlearn`. It reuses
component-loss definitions, optimizer validation, selected-parameter naming,
and diagnostics infrastructure, but owns its training step and per-sample
basis lifecycle. No Orthograd branch is added to the GU/PCGrad finalizer.

Add `configs/trainer/OrthogradUnlearn.yaml` with safe defaults and
`save_strategy: "no"`. Register the trainer without changing existing trainer
selection behavior.

### 4.2 Per-sample retain gradients

For one batch of size `k`:

1. compute the batch SimNPO candidate gradient once;
2. slice the retain input tree into `k` batch-size-one inputs;
3. compute one NLL retain gradient per sample with a separate forward and
   `torch.autograd.grad`;
4. convert each selected gradient to FP32 on the parameter device;
5. accumulate the arithmetic mean retain gradient;
6. insert the sample gradient into a global retain basis.

Tests must prove that the mean of the per-sample gradients reconstructs the
ordinary batch retain gradient under the repository’s answer masking and loss
normalization.

### 4.3 Global orthonormalization

Represent each basis direction as `dict[str, Tensor]`, spanning every selected
parameter tensor. Use two-pass global modified Gram–Schmidt:

\[
v \leftarrow v - \sum_j \langle v,q_j\rangle q_j
\]

with dots and norms summed over all parameter tensors. Drop a direction when
its post-orthogonalization norm is below a relative FP32 rank tolerance. Log
the requested retain rank, effective rank, dropped directions, and maximum
pairwise basis residual.

The implementation must not compute independent blockwise coefficients and
must not flatten the full parameter vector into one contiguous tensor.

### 4.4 Projection and writeback

Compute one global coefficient per basis vector,

\[
c_j = \langle g_f,q_j\rangle,
\qquad
g_f^\perp = g_f - \sum_j c_j q_j,
\]

then stream across parameter tensors and write

\[
p.grad \leftarrow g_r + 0.125g_f^\perp.
\]

The basis, sample gradient, forget gradient, retain mean, and partial
writeback state are transactional. Every success or exception path clears
them. If a failure occurs after any parameter gradient has been written, the
method clears all selected `parameter.grad` values before re-raising.

Trainer clipping and `optimizer.step()` remain outside `training_step()`.

## 5. Resource and Time Profiling

Profiling is opt-in and applied identically to Control, GU, and OrthoGrad arms.
Synchronization used for timing is therefore part of all profiled runs.

### 5.1 End-to-end measurements

A process-tree resource wrapper records:

- command wall-clock seconds and exit code;
- maximum aggregate resident CPU memory for the training process tree;
- baseline and maximum GPU device memory from NVML polling;
- polling interval and sample count.

The trainer records:

- Hugging Face `train_runtime`;
- CUDA maximum allocated and reserved bytes, reset immediately before train;
- update count and per-update synchronized wall time;
- first update separately from steps 2–10;
- p50, mean, and p95 step times.

### 5.2 Method phase measurements

GU records component-gradient collection, projection/finalization, and
writeback time. OrthoGrad records candidate-gradient collection, per-sample
retain-gradient collection, orthonormalization, projection/writeback, and
total training-step time. CUDA events are resolved once at the end of a step
to avoid a synchronization after every tensor operation.

Control has no surgery phases; its end-to-end and per-update timings are the
baseline for overhead ratios.

### 5.3 Analytic resource diagnostics

Every OrthoGrad update records:

- selected parameter count `d`;
- requested and effective rank;
- theoretical FP32 basis bytes `4*d*effective_rank`;
- peak live vector count;
- estimated dot/AXPY FLOPs for two-pass Gram–Schmidt and projection;
- preserved-signal ratio `||g_f^perp|| / ||g_f||`;
- maximum relative orthogonality residual against every retained basis
  direction.

The measured report distinguishes PyTorch allocator memory, NVML device
memory, CPU RSS, and analytic basis storage. These quantities must not be
presented as interchangeable.

## 6. Matched Experiment

Run sequentially on GPU 0 to avoid cross-device timing variance:

1. `control`: ordinary SimNPO with GU disabled;
2. `gu`: existing frozen-Adam GU;
3. `orthograd`: full-parameter Euclidean per-sample OrthoGrad.

Common configuration:

| Setting | Value |
|---|---|
| Model | Llama-3.2-1B-Instruct |
| Dataset | TOFU forget01 / retain99 / holdout01 |
| Seed | 0 |
| Optimizer updates | 10 |
| Batch / GAS | 4 / 1 |
| Precision | BF16 |
| Attention | FlashAttention 2 |
| Optimizer | PagedAdamW32 |
| Learning rate | `1e-5` |
| Adam beta1 / weight decay | 0 / 0 |
| Warmup epochs | 0 |
| Gradient checkpointing | non-reentrant |
| Checkpoint/model saving | disabled |

The timing comparison uses the same profiler and synchronization policy for
all arms. Evaluation runs after training and is excluded from training wall
time. Each arm persists only its resolved Hydra config, TOFU summary,
diagnostics, resource profile, and logs.

## 7. Outputs and Analysis

The matrix root is `saves/exp/ORTHOGRAD_MVP/<timestamp>`. Generate
`ORTHOGRAD_MVP_REPORT.md` containing:

1. raw TOFU metrics and paired deltas;
2. train runtime, step-time distribution, and phase breakdown;
3. peak allocated, reserved, NVML, and CPU RSS values;
4. GU and OrthoGrad time/memory overhead relative to Control;
5. effective retain rank and preserved-signal diagnostics;
6. projection residuals and actual parameter-delta probes;
7. an explicit feasibility verdict and scope limitations.

The analyzer rejects missing arms, failed exits, metric-key drift, non-finite
telemetry, update-count drift, checkpoint payloads, and inconsistent resolved
configs.

## 8. Tests

### Geometry

- global projection is orthogonal to every independent retain sample;
- rank-deficient and duplicate retain gradients are handled deterministically;
- decomposition and preserved-signal ratio are correct;
- a blockwise counterexample differs from the global result;
- combination weights equal `retain + 0.125*projected_forget`;
- the mean per-sample retain gradient equals the batch retain gradient.

### Trainer lifecycle

- exactly one basis and finalization per optimizer update;
- retain batch sizes 1 and 4 work;
- all buffers and gradients clear on success and injected exceptions;
- unsupported GAS and distributed/precision/optimizer modes fail closed;
- Trainer clipping and optimizer step remain reachable after writeback;
- the final raw gradient satisfies the measured Euclidean residual before the
  approximate Adam step, while the realized parameter delta is recorded
  separately without claiming exact Euclidean invariance.

### GPU and scripts

- BF16 eager and FlashAttention 2 produce finite, close tiny-model updates;
- real PagedAdamW32 runs two public updates;
- memory counters are positive and internally ordered;
- the three-arm launcher uses the exact matched configuration;
- analysis and reports are deterministic;
- no model, optimizer, scheduler, RNG, Trainer-state, or checkpoint payload is
  persisted.

## 9. Success Criteria

The MVP is complete only if:

- all CPU and GPU tests pass without hiding skips;
- all three real arms finish ten updates and evaluation;
- OrthoGrad’s maximum retain-basis residual is below `1e-5`;
- GU and OrthoGrad timing and all requested memory quantities are present;
- overhead ratios are computed from matched successful arms;
- no checkpoint payload exists;
- the report limits conclusions to feasibility, resource cost, and observed
  single-run behavior.

An OrthoGrad OOM is also a valid feasibility result only after the measured
peak, analytic basis requirement, attempted configuration, and failure path
are recorded. It does not satisfy the three-arm endpoint comparison and must
not be reported as an implementation success.
