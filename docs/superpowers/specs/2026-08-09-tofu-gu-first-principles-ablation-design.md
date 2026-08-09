# TOFU GU First-Principles Ablation Design

**Status:** Approved for implementation planning  
**Date:** 2026-08-09  
**Branch:** `feat/GU`  
**Execution device:** physical GPU 0 only

## Goal

Identify why common GU forgets less than the method-native baseline on TOFU,
separating trainable-parameter scope, parameter storage precision, current-step
projection, historical constraints, objective coefficients, and training budget.
Use the causal findings to construct and validate a less conservative GU
candidate from constrained-optimization principles.

The study focuses on TOFU `forget10` with four representative objectives:

- CEU: forget-only objective with high observed GU correction;
- GradDiff: explicit retain loss plus medium/high GU correction;
- SimNPO: low forget coefficient and nearly inactive GU correction;
- UNDIAL: no explicit retain loss, strong forgetting, low GU correction.

All experiments are checkpoint-free and use live in-memory evaluation.

## Causal Model

The applied update is treated as a composition of interventions:

```text
objective and training budget
  -> trainable parameter subspace
  -> selected-parameter storage precision
  -> current retain half-space projection
  -> historical retain constraints
  -> applied optimizer delta
```

The existing baseline-versus-GU comparison changes several of these at once.
The ablation estimates the sequential effects in the order in which the
interventions enter the training system:

```math
E_scope     = Y(scope_bf16) - Y(native),
E_precision = Y(scope_fp32) - Y(scope_bf16),
E_current   = Y(gu_rank0) - Y(scope_fp32),
E_history   = Y(gu_rank8) - Y(gu_rank0).
```

This decomposition is intentionally ordered rather than presented as a
permutation-invariant attribution. It answers which implementation stage moves
the endpoint under the actual pipeline.

## Stage A: Five-Arm Mechanism Decomposition

Run all five arms for each of CEU, GradDiff, SimNPO, and UNDIAL with seed 0,
the pinned TOFU `forget10` model/data revisions, and the method's current
learning rate, objective coefficients, effective batch, and 10-epoch budget.

### A0: Native

- method-native trainable parameters;
- native BF16 parameter storage;
- no common GU collection or projection.

### A1: Scope-BF16

- train only `model.layers.(29|30|31).mlp.down_proj.weight`;
- keep selected parameters in BF16;
- no retain covector, projection, history, or GU diagnostics.

### A2: Scope-FP32

- same three selected matrices;
- promote selected storage to FP32;
- no retain covector, projection, or history.

### A3: GU-rank0

- current common GU implementation;
- three selected FP32 matrices;
- current retain constraint only;
- `retain_history_rank=0`.

### A4: GU-rank8

- current production GU configuration;
- current retain constraint plus eight historical constraints.

Stage A contains 20 fresh GPU0 jobs. Existing baseline/GU results are context,
not substitutes, so all five arms share the same launch environment and
resource monitor.

## Scope-Control Implementation

Add an optional common `scope_control` mapping to `UnlearnTrainer` for the two
no-projection controls:

```yaml
scope_control:
  enabled: true
  parameter_regex:
    - model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight
  selected_storage: native  # native | float32
```

When absent or disabled, existing trainers are bitwise unchanged. When enabled,
it freezes every unselected parameter and optionally promotes selected BF16
storage to FP32 before creating the ordinary optimizer. It registers no
optimizer hooks and computes no auxiliary retain forward. Combining
`scope_control.enabled=true` with `gu.enabled=true` is rejected so each arm has
one authoritative owner of scope and precision.

The mapping has an exact public contract and receives focused tests for scope,
dtype, optimizer membership, disabled equivalence, invalid configuration, and
absence of GU transient state.

## Stage B: Capacity, Budget, and Objective Tests

Run the following seed-0 arms after Stage A. All retain production GU safety
unless explicitly noted.

### Common arms for all four methods

1. six-layer scope: layers 26--31;
2. twelve-layer scope: layers 20--31;
3. doubled training budget: 20 epochs, preserving the method LR and data order.

### Objective-specific arms

- GradDiff:
  - `gamma=1, alpha=0`, allowing GU alone to protect retain behavior;
  - `gamma=2, alpha=0`, testing additional forget drive without duplicate
    retain regularization.
- SimNPO:
  - `gamma=0.5, alpha=1`;
  - `gamma=1.0, alpha=1`;
  - `gamma=0.5, alpha=0`, testing GU-only retain protection.

CEU has no forget coefficient, and UNDIAL already uses `alpha=0`; they receive
only the common scope and budget arms. Stage B therefore contains 17 jobs.

## First-Principles Improvement Candidate

The current strict constraint solves a nearest-update problem:

```math
\min_\Delta \frac12\lVert\Delta-\Delta_0\rVert_M^2
\quad\text{s.t.}\quad g_{r,j}^T\Delta\le0.
```

This protects represented retain directions but does not directly maximize
forgetting under an allowed utility loss. The proposed improvement is a
budgeted lexicographic SafeProx:

```math
\max_\Delta -g_f^T\Delta
\quad\text{s.t.}\quad
L_r(\theta+\Delta)-L_r(\theta)\le B,
\quad \lVert\Delta-\Delta_0\rVert_M\le R.
```

The implementable form is:

1. build a forget-only optimizer proposal when the objective's retain term is
   redundant with GU;
2. project only genuinely harmful current/history constraints;
3. keep the raw current retain-gradient norm so a shifted first-order budget
   remains in retain-loss units despite row normalization;
4. use actual finite-step retain NLL as the authoritative filter;
5. backtrack to the largest feasible scale;
6. expand the trainable subspace only when the restricted scope lacks forget
   capacity, as shown by Stage A/B.

No budgeted mode is promoted merely because it forgets more. It must satisfy
the endpoint floors below.

### Stage C budget sweep

For each method, select the Stage-B seed-0 configuration with the lowest ES-Un
among configurations that already satisfy the promotion retain/privacy/utility
floors. If no Stage-B configuration satisfies the floors, use production
GU-rank8 as the anchor rather than weakening the floors.

Run the anchor with actual finite-step retain budgets:

```text
B = 1e-4, 5e-4, 1e-3 mean answer-masked retain NLL
backtracking scales = 1, 0.5, 0.25, 0.125, 0.0625
trust radius = the ordinary optimizer proposal norm (scale never exceeds 1)
```

Production `B=0` is represented by the existing strict GU arm and is not rerun.
There are at most 12 Stage-C jobs. The largest budget satisfying all endpoint
floors is the method's candidate; a larger budget cannot be selected when a
smaller feasible budget forgets at least as much.

## Metrics and Diagnostics

Each arm records:

- TOFU `extraction_strength` (ES-Un, lower is better);
- `retain_extraction_strength` (ES-Re, higher is better);
- `privleak` (higher is better under the current metric convention);
- `model_utility` (higher is better);
- every available TOFU endpoint for audit;
- wall-clock time and peak NVML MiB;
- selected parameter count/numel and storage dtype;
- projection count, current/history active-set counts, correction ratio, maximum
  violation, and zero-step count when GU is enabled.

The analyzer reports causal endpoint deltas between adjacent Stage-A arms. It
does not average forgetting, privacy, and utility into one score.

## Promotion Rule

A Stage-B or budgeted candidate advances to three-seed validation only if, on
seed 0 relative to production GU-rank8:

- ES-Un improves by at least `0.05` absolute;
- ES-Re decreases by no more than `0.02` absolute;
- model utility decreases by no more than `0.02` absolute;
- privleak decreases by no more than `2.0` points;
- all GU numerical and artifact validation passes;
- no erroneous zero step or nonfinite state occurs.

If no candidate satisfies every floor, the result is “no safe improvement”;
the study does not relax the floors post hoc.

Every advancing candidate and its matched Scope-FP32/no-projection control run
seeds 1 and 2. Final conclusions require all three seeds and report mean plus
sample standard deviation.

## Execution and Outputs

Use physical GPU0 exclusively and a fresh root:

```text
/root/tofu_gu_ablation_20260809_gpu0/
```

The root contains a manifest, durable queue state, exact command/environment
sidecars, per-job live-evaluation endpoints, resource evidence, and:

```text
ABLATION_RAW.json
ABLATION_TABLES.md
ABLATION_REPORT.md
```

Model, optimizer, scheduler, RNG, Trainer state, and checkpoint payloads are
forbidden. Infrastructure failures may be retried once with the identical
command. Scientific failures remain terminal and are never replaced.

## Acceptance Criteria

- All 20 Stage-A jobs reach durable terminal states.
- The four adjacent causal effects are computable for every valid method.
- All 17 Stage-B jobs reach durable terminal states.
- Every promoted candidate receives seeds 1 and 2, or the report explicitly
  states that no candidate passed the floors.
- Every child PID is observed on physical GPU0 only.
- No checkpoint or model payload is written.
- The final report distinguishes established causal effects, endpoint
  correlations, and proposed mechanisms.
