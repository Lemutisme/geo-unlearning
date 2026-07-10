# Correct Geometric Unlearning Design

**Date:** 2026-07-10
**Status:** Approved architecture; written specification pending author review

## 1. Goal

Replace the unreachable and mathematically inconsistent GU implementation with a tested two-stage system:

1. **Stage A:** faithfully implement the paper's optimizer-induced geometry on the current pre-momentum candidate;
2. run a checkpoint-free Llama-3.2-1B quick validation and analyze theory-versus-observation;
3. **Stage B:** constrain the realized AdamW parameter delta, including momentum and weight decay;
4. repeat the paired quick validation and update the theoretical interpretation from measured diagnostics.

The selection objective is **Pareto-safe forgetting**: satisfy retain, privacy, and utility floors first, then maximize forgetting among feasible candidates.

## 2. Scope

### In scope

- The eight submitted objectives: GradDiff, CEU, DPO, NPO, SatImp, SimNPO, UNDIAL, and WGA.
- Single-process and native Accelerate DDP execution.
- BF16 and FP16 with explicit AMP tests.
- Torch Adam/AdamW and the repository's 32-bit bitsandbytes PagedAdamW state layout.
- Correct gradient accumulation and distributed averaging.
- A paper-faithful projector and a separately labelled safe-cone experimental projector.
- A Stage-B realized-delta correction layer.
- Small JSON/JSONL metrics and diagnostic outputs.

### Out of scope for this cycle

- DeepSpeed ZeRO, FSDP, SageMaker MP, Apex, fused optimizers with unsupported AMP contracts, and 8-bit optimizer states.
- Submission-grade multi-seed experiments or recreation of Tables 1 and 5--8.
- Saving trained checkpoints during quick validation.
- Claiming that the corrected implementation generated the submitted paper's existing numbers.

Unsupported distributed modes must fail before training with an actionable message.

## 3. Approaches Considered

### Patch the current `geometric.py`

This has the smallest diff but retains duplicated objective formulas, ambiguous gradient provenance, stale basis coordinates, and an invalid lifecycle. Rejected.

### Modular objective/projector/runtime system

This separates objective semantics, pure geometry, and Trainer/optimizer integration. It supports unit-level mathematical tests and lets Stage B reuse the same constraints. Selected.

### Rewrite the complete Trainer and optimizer

This offers maximum lifecycle control and could eventually support ZeRO, but it copies a large version-specific training loop and increases regression risk. Deferred.

## 4. Components and Responsibilities

### `src/trainer/unlearn/objectives.py`

Defines `ObjectiveParts` and one source of truth for each objective:

- unscaled `forget_loss`;
- optional unscaled `retain_loss`;
- `gamma` and `alpha`;
- model outputs;
- reference-model requirement;
- validated input schema.

Every baseline trainer and GU consume the same component implementation. With geometry disabled, GU and the matching baseline must have equal scalar losses and parameter gradients.

### `src/trainer/unlearn/geometric_projector.py`

Contains pure Torch code with no Trainer dependency:

- optimizer denominator and square-root coordinate maps;
- raw-gradient FIFO and current-metric basis reconstruction;
- FP32 re-orthogonalization;
- tangent/normal decomposition;
- paper sign/cap rule;
- safe-cone projection and convergence diagnostics;
- Stage-B realized-delta projection;
- structured projection diagnostics.

The empty tangent projection returns zero. All zero norms and non-finite values have explicit behavior.

### `src/trainer/unlearn/geometric_runtime.py`

Owns runtime state:

- per-accumulation-window component-gradient buffers;
- DDP mean reduction;
- optimizer state adapters for `exp_avg_sq` and bitsandbytes `state2`;
- idempotent raw optimizer pre/post hook installation;
- preservation of the user's clipping threshold while disabling Trainer's earlier clip, followed by clipping after Stage-A gradient replacement;
- Stage-B parameter snapshots and corrections;
- hook counts, a window identifier, callback/next-window cleanup, and AMP-overflow handling.

### `src/trainer/unlearn/geometric.py`

Becomes orchestration only:

- validates configuration and supported execution mode;
- chooses projected parameters exactly;
- invokes shared objective components;
- captures component gradients for selected tensors;
- configures Stage A, Stage A+, and Stage B;
- exposes diagnostics to Trainer logging.

It must not define an unused `optimizer_step()` method.

### Configuration

`configs/trainer/GeometricUnlearn.yaml` receives explicit fields for:

- `projection_mode: paper | cone`;
- `stage_b: false | true`;
- rank, refresh period, residual and coverage tolerances;
- projected layer selection and `freeze_others`;
- `kappa`, `tau`, and repair coefficient;
- objective name plus its exact objective arguments;
- anchor-loss type and mask;
- first-step metric fallback;
- unsupported-mode policy.

A native-DDP Accelerate configuration is added. The existing DeepSpeed default is not silently reused.

## 5. Stage-A Data Flow

1. The objective registry validates the batch schema and returns `ObjectiveParts`.
2. For each microbatch, GU differentiates the forget and objective-retain components separately for selected parameters, scales them exactly as Accelerate scales the main loss, and accumulates detached FP32 buffers.
3. On scheduled refresh windows, GU separately differentiates the configured retain anchor. Forward KL means `KL(p_theta || p_ref)`, not the reverse-KL expression in the old implementation.
4. The ordinary Trainer backward remains responsible for unselected parameter gradients and standard logging.
5. A raw optimizer pre-hook runs exactly once at every actual native optimizer update, including short final accumulation windows.
6. The pre-hook reduces auxiliary buffers across DDP ranks, reconstructs the current basis, reads the actual optimizer denominator, and replaces selected gradients with the optimizer-equivalent projected gradients.
7. GU stores the user's `max_grad_norm`, sets Trainer's earlier clipping threshold to zero, and has the pre-hook clip the final complete gradient set using the stored threshold.
8. Runtime buffers are cleared after a successful step; if AMP skips the raw optimizer call, an optimizer-event callback or the next window transition clears the stale window before any new gradient is accumulated.

No gradient is recovered by subtracting a KL gradient from a total containing an NLL gradient. The implementation keeps separately named objective-retain and anchor-retain gradients.

## 6. Basis and Metric Policy

- Store recent raw retain directions in a bounded FIFO.
- At each update, transform them with the current `P^(1/2)` and rebuild an FP32 orthonormal basis.
- Insert the current accumulated objective-retain direction before historical anchors so the current safety target is represented.
- Use the optimizer group's epsilon and bias-correction convention.
- Read Torch `exp_avg_sq` or floating-point bitsandbytes `state2` only when shape and dtype are valid.
- Use a logged identity metric on the first step when no optimizer state exists.
- Expand rank or projected layer coverage only when coverage or harmful-unprotected-mass diagnostics exceed tolerance.

## 7. Stage-A Projection Modes

### `paper`

Implements the submitted normal projection, same-sign condition `a_i*b_i > tau`, tangent cap, and optional retain repair. This is the rebuttal-safe reference implementation.

### `cone`

Projects the candidate onto the intersection of first-order retain-safe halfspaces in the low-rank tangent coordinates. It reports convergence and maximum violation. Non-convergence falls back to the pure normal direction. This mode is an explicitly new experimental extension.

## 8. Stage-B Data Flow

1. The Stage-A pre-hook snapshots protected parameters and the frozen metric/basis immediately before the raw optimizer step.
2. AdamW performs its normal state update and proposes a parameter delta containing momentum, bias correction, and weight decay.
3. A raw optimizer post-hook computes the realized delta.
4. The delta is whitened and projected onto the retain-safe cone.
5. Protected parameters are replaced with `theta_before + corrected_delta`.
6. Logs record the raw and corrected retain derivatives and the correction ratio.

Optimizer moments remain proposal state. Since every realized selected-parameter delta is corrected, the first-order selected-parameter retain condition applies step by step. A global guarantee requires projecting all trainable tensors or setting `freeze_others=true`.

## 9. Error Handling

Training stops before the first update when any of the following holds:

- DeepSpeed, ZeRO, FSDP, reentrant checkpointing, or an unsupported optimizer/AMP contract is active;
- `gamma` is zero for an objective that requests a forget projection;
- GU has no retain input or no selected trainable parameter;
- objective coefficients change within one accumulation window;
- DDP auxiliary reduction cannot be performed;
- a configured explicit parameter regex matches nothing.

Non-finite component gradients skip the geometric update and emit a diagnostic; they are never silently inserted into the basis. Safe-cone non-convergence uses the pure normal fallback and records the event.

## 10. Testing Strategy

All production changes follow red-green-refactor.

### Pure mathematical tests

- `P^(1/2)` mapping and inverse closure;
- Euclidean and metric orthogonality;
- projector decomposition, idempotence, and empty-basis behavior;
- correct sign examples and cap bound;
- retain directional derivative for paper and cone modes;
- stale-metric basis reconstruction;
- safe-cone feasibility and fallback;
- realized-delta safety for momentum and weight-decay counterexamples.

### Objective parity tests

- Scalar loss and gradient parity for all eight objectives with geometry off;
- exact YAML coefficients and schemas;
- NLL/KL separation and reference-model lifecycle;
- `alpha=0` behavior for CEU and UNDIAL.

### Runtime tests

- one hook per update for accumulation 1, 2, and 4;
- a dataset shorter than one accumulation window;
- DDP two-rank equality with a single-process equivalent batch;
- BF16 and FP16 scale/unscale/clip order;
- optimizer recreation and hook idempotence;
- Torch AdamW and bitsandbytes `state2` adapters;
- explicit DeepSpeed/FSDP rejection;
- no graph is differentiated after it is freed;
- Stage-B correction includes momentum and weight decay.

## 11. Checkpoint-Free 1B Quick Validation

### Environment

- cached `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`;
- TOFU `forget05` / `retain95`;
- SimNPO objective, seed 0;
- one A100-80GB initially; native DDP is covered separately by a small integration test;
- 32 fixed paired training examples, eight optimizer updates, 32 forget and 32 retain evaluation examples, and 16 examples for extraction diagnostics;
- one permitted escalation to 24 optimizer updates if all four arms remain within numerical noise after eight updates.

### Arms

1. SimNPO baseline;
2. Stage-A `paper`;
3. Stage-A `cone`;
4. Stage-A `cone` plus Stage B.

Each arm reloads the same base model and uses identical data order and objective coefficients.

### No-checkpoint contract

The validation driver calls Trainer without `save_model()` or `save_state()`, sets `save_strategy=no` and `report_to=none`, and verifies that no model/state checkpoint files exist afterward. Only compact configuration, JSON metrics, and JSONL diagnostics may be retained.

### Metrics

- held-out forget loss and extraction-strength subset;
- held-out retain NLL and forward KL;
- small utility subset when the evaluator can run within the quick budget;
- basis coverage, entanglement ratio, harmful tangent fraction, cap activation;
- predicted Stage-A first-order retain derivative;
- realized finite-step retain change;
- raw and corrected Stage-B retain derivatives and correction ratio;
- wall time and peak allocated CUDA memory.

### Decision rule

The quick-run defaults are:

- retain NLL no more than 2% above the baseline;
- forward retain KL no more than `baseline + 0.01`;
- when computed, ES-Re, Privacy, and MU no more than `0.02` below the baseline.

All tolerances are logged and may not be changed after inspecting an arm's outcome. First reject arms that violate any available floor. Among feasible arms, choose the lowest forgetting score. A tiny-subset result is diagnostic only and will be labelled accordingly.

## 12. Result-Driven Theory Analysis

- If Stage-A predicted drift is safe but finite retain loss increases, quantify the second-order gap and reduce step size/trust radius.
- If Stage B materially reduces drift, the momentum/weight-decay limitation is empirically active.
- If Stage-B correction is negligible, Stage A already aligns the optimizer proposal and B may be unnecessary overhead.
- If cone mode improves forgetting at equal retain floors, the paper sign rule is empirically conservative.
- If basis coverage is poor, increase rank or projected layers before changing objective weights.
- If coverage is high but retention still fails, inspect unselected tensors and finite-step curvature rather than increasing rank blindly.

## 13. Completion Criteria

- Mathematical, objective-parity, lifecycle, and Stage-B tests pass in the available environment.
- Stage A executes exactly once per optimizer update and satisfies its represented first-order invariant numerically.
- Stage B satisfies the represented invariant on the realized selected-parameter delta numerically.
- The 1B quick validation completes without writing checkpoints.
- The report distinguishes verified facts, diagnostic observations, and untested claims.
