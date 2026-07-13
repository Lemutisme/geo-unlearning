# WMDP-Cyber UAM and Residual-GU W2 Design

**Date:** 2026-07-13  
**Branch:** `feat/uam-gu`  
**Base implementation:** `UAMUnlearn` and `GeometricUnlearn`  
**Status:** Approved for implementation

## 1. Goal

Produce the reviewer-requested, directly matched WMDP-Cyber comparison:

| Method | WMDP-Cyber ↓ | MMLU ↑ | Relative runtime |
|---|---:|---:|---:|
| Base | measured | measured | — |
| RMU | measured | measured | measured |
| RMU + GU | measured | measured | measured |
| UAM | measured | measured | measured |
| UAM + GU | measured | measured | measured |

The experiment tests whether residual GU changes the forget-retain frontier of
a paper-derived UAM candidate update. It does not assume that the result is
positive. Every completed arm is reported, and complementarity is claimed only
when the measured endpoints support it.

## 2. Scientific Scope and Nomenclature

This experiment is a **Cyber-only, paper-derived port** of UAM Algorithm 2. It
is not a reproduction of the paper's joint WMDP-Bio/WMDP-Cyber training because
the authorized workspace contains only the WMDP-Cyber forget corpus.

The authors' released repository contains a generic PyTorch/vision UAM
minimizer, but no Zephyr, WMDP, RMU, or WikiText pipeline. The implementation
will therefore be described as:

> a paper-faithful WMDP-Cyber representation-objective port, cross-checked
> against the authors' released perturb-and-restore implementation.

It must not be described as “using the authors' released WMDP code.”

The external references used for the port are pinned in the implementation
notes:

- UAM paper: NeurIPS 2025 proceedings paper
  `86cca1b77a808b7f9ef5a987bcae6803`;
- released UAM repository commit
  `9ebd70de6bd069add730ffde8f0deb39dbb1436b`;
- WMDP RMU reference commit
  `bc5e1ba0367ea826caeeeaa50656336a1e87acfb`.

The following two fidelity axes are kept separate:

1. **Objective fidelity:** representation losses, Gaussian initialization
   noise, layer selection, learning rate, reflection coefficient, and primary
   perturbation radius follow the paper's WMDP specification.
2. **Systems matching:** every trainable arm uses the same BF16,
   FlashAttention 2, PagedAdamW32, gradient-accumulation, and frozen
   preconditioner settings required for a controlled GU comparison.

The optimizer setup is therefore a matched systems comparison rather than a
claim that the paper used the identical PagedAdamW configuration.

## 3. In-Scope and Out-of-Scope Work

### In scope

- untouched Zephyr-7B-β Base evaluation;
- a matched Cyber-only RMU implementation;
- RMU with global rank-one GU;
- UAM Algorithm 2 over WMDP representation losses;
- residual-only GU over the UAM-specific correction;
- selected-parameter training for layers 5–7;
- WMDP-Cyber and MMLU evaluation;
- wall-clock and peak-memory profiling;
- geometry and actual-parameter-delta diagnostics;
- single-GPU, checkpoint-free execution on GPU 0.

### Out of scope

- WMDP-Bio or joint Bio/Cyber training;
- a claim of exact reproduction of the paper's Table 4;
- DDP, FSDP, DeepSpeed, FP16, or reentrant checkpointing;
- historical low-rank retain bases;
- custom momentum-aware Adam;
- multi-seed superiority claims in the first W2 matrix;
- automatic inclusion of a result in the rebuttal before audit.

## 4. Architecture

The implementation is isolated from the existing token-NLL/SimNPO UAM path.

### 4.1 Shared representation helper

Add a focused WMDP representation-objective helper responsible for:

- locating the selected transformer block by exact module name;
- capturing the block output with a transaction-safe forward hook;
- removing the hook even if the forward pass raises;
- computing a label-mask-aware representation MSE;
- evaluating the frozen reference model under `torch.no_grad()`;
- injecting seeded Gaussian forget noise;
- validating activation shapes, masks, and finite values.

This helper contains no gradient projection or optimizer logic.

### 4.2 `WMDPUAMUnlearn`

`WMDPUAMUnlearn` subclasses `UAMUnlearn` and changes the objective hooks,
WMDP runtime contract, and the coordinate split described below. It reuses the
existing:

- component-gradient buffers;
- effective-batch retain replay;
- temporary parameter perturbation and restoration;
- UAM reflection finalizer;
- residual-GU finalizer and sign gate;
- optimizer-coordinate adapter;
- actual-delta and resource diagnostics.

The existing token-UAM finalizer performs both UAM reflection and residual GU
in frozen optimizer coordinates. W2 instead requires two explicitly separated
geometries:

- the UAM reflection is Euclidean, matching paper Algorithm 2;
- only the residual-GU projection uses the frozen PagedAdamW metric.

The finalizer must expose this as an explicit WMDP configuration rather than
silently reusing the token-UAM coordinate convention.

It creates a frozen reference model, uses layer 7 representations, and
restricts selected parameters to:

```text
model.layers.(5|6|7).mlp.down_proj.weight
```

The restriction is checked both against the optimizer parameter groups and the
named parameters returned to the geometry finalizer.

### 4.3 `WMDPRMUUnlearn`

`WMDPRMUUnlearn` subclasses `GeometricUnlearn` and exposes explicit RMU forget
and retain components:

- `gu_enabled=false`: matched RMU baseline;
- `gu_enabled=true`: global rank-one RMU+GU.

The control vector is generated once from the configured seed and reused for
the run. The matched RMU gradient is

\[
g_{\mathrm{RMU}}=g_f+\alpha g_r,
\qquad \alpha=100.
\]

RMU+GU writes

\[
g_{\mathrm{RMU+GU}}
=P_{g_r}^{\perp,H}g_f+\alpha g_r.
\]

Disabling GU must recover the matched RMU parameter update exactly within the
configured numerical tolerance.

## 5. WMDP Representation Objectives

Let `h_7(x; w)` be the output of transformer layer 7 for the trainable model,
and let `h_7(x; w_0)` be the output of the frozen original model.

### 5.1 Retain objective

For a WikiText retain batch `x_r`, define

\[
L_r(w)=
\operatorname{MaskedMSE}
\left(h_7(x_r;w),h_7(x_r;w_0)\right).
\]

Only non-padding, supervised tokens contribute to the loss. An empty mask
fails closed.

### 5.2 RMU forget objective

For a WMDP-Cyber forget batch `x_f`, define

\[
L_f^{\mathrm{RMU}}(w)=
\operatorname{MaskedMSE}
\left(h_7(x_f;w),c u\right),
\]

where `u` is a seeded random unit vector and `c=20`.

### 5.3 UAM forget objective

The unperturbed UAM representation loss is zero at initialization, so the
paper injects Gaussian noise. With

\[
\xi\sim\mathcal N(0,0.01^2),
\]

define

\[
L_f^{\mathrm{UAM}}(w)=
\operatorname{MaskedMSE}
\left(h_7(x_f;w)+\xi,h_7(x_f;w_0)\right).
\]

Noise is generated from a run-local seeded generator. Replaying or evaluating
retain data must not advance this generator.

## 6. UAM and Residual-GU Updates

### 6.1 Paper-derived UAM

Let

\[
g_f=\nabla L_f^{\mathrm{UAM}}(w).
\]

The primary paper-derived perturbation is

\[
\delta=
\rho\frac{g_f}{\lVert g_f\rVert_2^2+\varepsilon},
\qquad \rho=5\times10^{-5}.
\]

At the perturbed parameters, compute

\[
\widetilde g_r=\nabla L_r(w+\delta).
\]

The raw-gradient UAM candidate is

\[
g_{\mathrm{UAM}}
=\left(I-\gamma P_f\right)\widetilde g_r,
\quad
P_f=\frac{g_fg_f^\top}{\lVert g_f\rVert_2^2+\varepsilon},
\quad \gamma=2.
\]

This reflection is computed in raw Euclidean gradient coordinates. It is not
whitened by the optimizer denominator.

The perturbation is temporary: every selected parameter must be restored
bitwise to its pre-perturbation value before final-gradient writeback.

### 6.2 Residual UAM+GU

Compute the ordinary retain gradient at the unperturbed parameters:

\[
g_r=\nabla L_r(w),
\qquad
r_{\mathrm{UAM}}=g_{\mathrm{UAM}}-g_r.
\]

Only the UAM-specific residual is constrained. The raw `r_UAM` and `g_r` are
mapped into frozen optimizer coordinates, projected there, and mapped back to
an equivalent raw gradient:

\[
n=P_{g_r}^{\perp,H}r_{\mathrm{UAM}}.
\]

The existing forget-benefit sign gate retains `n` only when its predicted
forget derivative is beneficial and its relative orthogonality residual is at
most `1e-6`. The applied gradient is

\[
g_{\mathrm{UAM+GU}}=
\begin{cases}
g_r+n,&\text{if the gate passes},\\
g_r,&\text{otherwise}.
\end{cases}
\]

This preserves ordinary retain descent rather than projecting away the whole
UAM update.

## 7. Optimizer Geometry and Claim Boundary

All **GU projections** use the same frozen diagonal PagedAdamW32
preconditioner adapter as the existing GU implementation. The UAM reflection
itself remains Euclidean as specified in Section 6. Runtime validation
requires:

- `adam_beta1=0`;
- `weight_decay=0`;
- BF16 or FP32;
- one process on GPU 0;
- `gradient_checkpointing.use_reentrant=false`.

The implementation is an approximate optimizer-aware GU because the standard
optimizer updates its denominator after final-gradient writeback. It is not
called exact optimizer-aware GU.

The constraint is a global rank-one, current-effective-batch constraint. There
is no history rank or refresh schedule in this W2 experiment. The response and
paper text must not claim otherwise.

## 8. Experimental Matrix

### 8.1 Shared configuration

| Setting | Value |
|---|---|
| Model | `HuggingFaceH4/zephyr-7b-beta` |
| Forget data | authorized WMDP-Cyber forget corpus |
| Retain data | WikiText retain corpus |
| Representation | layer 7 |
| Trainable tensors | layers 5–7 `mlp.down_proj.weight` |
| Effective batch | 4 |
| Physical batch / GAS | 1 / 4 |
| Updates | 80 |
| Learning rate | `5e-5` |
| Precision | BF16 |
| Attention | FlashAttention 2 |
| Optimizer | PagedAdamW32 |
| Adam beta1 / weight decay | 0 / 0 |
| Seed | 42 |
| Gradient clipping | standard Trainer clipping after final writeback |
| Checkpoints | disabled and audited |

### 8.2 Arms

1. **Base:** untouched checkpoint evaluation only.
2. **RMU:** steering coefficient 20, retain weight 100.
3. **RMU+GU:** matched RMU with global rank-one GU on the forget component.
4. **UAM:** paper-derived UAM objective and reflection.
5. **UAM+GU:** ordinary retain descent plus gated, projected UAM residual.

Every trainable arm starts from the same untouched checkpoint and uses the same
data ordering, effective batch, number of updates, optimizer, and evaluator.
The model revision, tokenizer revision, lm-eval version, and Cyber-corpus
provenance hash are recorded once and asserted identical across arms.

### 8.3 Perturbation-radius policy

The primary W2 comparison uses `rho=5e-5`, the best value stated by the paper's
LLM sensitivity discussion. The paper and released code disagree about both
the radius normalization and parts of the search grid. Therefore:

- the primary result uses the paper equation and `rho=5e-5`;
- no data-dependent radius substitution is allowed during a run;
- a later sensitivity appendix may evaluate `5e-6`, `5e-5`, `5e-4`, and
  `5e-3` after the five-arm matrix passes mechanism audits;
- the released-code `rho*g/||g||` normalization is an optional ablation, not
  the primary W2 result.

## 9. Base-Evaluation Gate

Before launching any trainable arm, evaluate the untouched checkpoint using
the exact evaluator that will be reused for all arms.

The paper reports:

| Metric | Paper Base |
|---|---:|
| MMLU | 0.5810 |
| WMDP-Cyber | 0.4400 |

The training matrix may proceed only if:

\[
|\mathrm{MMLU}-0.5810|\le 0.02,
\qquad
|\mathrm{Cyber}-0.4400|\le 0.03.
\]

If the gate fails, stop and diagnose tokenizer revision, chat-template use,
lm-eval version, task configuration, and checkpoint identity. Do not launch
the trainable arms or replace the paper numbers with a post-hoc tolerance.

Passing the gate establishes evaluation compatibility only; it does not make
the Cyber-only training a reproduction of the paper's joint experiment.

## 10. Diagnostics and Resource Measurement

Each trainable arm records:

- optimizer-update count and finalizer count;
- forget, retain, perturbed-retain, and final gradient norms;
- requested and BF16-effective perturbation norms;
- UAM reflection coefficient;
- GU projection coefficient;
- sign-gate pass rate and failure reason;
- maximum relative optimizer-metric orthogonality residual;
- actual selected-parameter delta at steps 1 and 80;
- actual forget and retain first-order derivatives;
- training wall-clock time;
- end-to-end wall-clock time;
- CUDA allocator peak, process-level NVML peak, and CPU RSS.

Relative runtime is normalized to matched RMU training wall time. Base has no
relative training runtime.

## 11. Tests

Implementation follows test-driven development. Required tests include:

1. layer 7 activations and frozen-reference outputs are captured correctly;
2. hooks are removed after successful and failing forwards;
3. masked representation MSE excludes padding and rejects an empty mask;
4. seeded Gaussian noise makes the initial UAM forget gradient non-zero;
5. only the three intended `down_proj.weight` tensors are selected;
6. temporary perturbations are restored after success and every exception;
7. UAM reflection matches a hand-computed tensor example;
8. changing the frozen Adam denominator does not change the Euclidean UAM
   candidate before residual GU is applied;
9. UAM+GU residual projection is globally orthogonal in frozen optimizer
   coordinates;
10. a failed sign gate writes exactly the ordinary retain gradient;
11. RMU with GU disabled matches the ordinary component-sum update;
12. RMU+GU matches a hand-computed global projection;
13. effective batch 4 and physical batch 1 with GAS 4 agree within tolerance;
14. BF16, FlashAttention 2, and PagedAdamW32 complete two real GPU updates;
15. finalizer and projection call counts equal optimizer-update counts;
16. no model, optimizer, scheduler, RNG, or Trainer checkpoint payload is
    persisted.

## 12. Failure Handling

The matrix fails closed when any of the following occurs:

- Base evaluation misses the compatibility gate;
- an objective, gradient, perturbation, or final gradient is non-finite;
- a selected tensor is missing, duplicated, or outside layers 5–7;
- the perturbation is not restored exactly;
- a finalizer count differs from the requested 80 updates;
- optimizer-metric residual exceeds `1e-6` on a kept correction;
- an arm exits non-zero or lacks a required summary/profile;
- an output directory already exists;
- a symlink or checkpoint payload appears in staging or persistent artifacts.

An arm failure prevents later arms from being scheduled. Completed artifacts
are retained for diagnosis but are not promoted into the final W2 table as a
successful matrix.

## 13. Execution Order and Budget

Only GPU 0 is used.

1. run untouched Base evaluation;
2. audit the Base compatibility gate;
3. run RMU;
4. run RMU+GU;
5. run UAM;
6. run UAM+GU;
7. audit artifacts and recompute the table directly from raw JSON;
8. run the optional rho sensitivity only if the main matrix passes.

The expected cost of the primary matrix is approximately 2–3 GPU-hours on the
available A100 80 GB, dominated by repeated MMLU/WMDP evaluation. The optional
rho grid is budgeted separately and cannot delay the primary five-arm table.

## 14. Reporting and Claims

The final table reports raw point estimates, deltas against RMU and UAM,
standard errors supplied by lm-eval, and relative training runtime.

The result text follows these rules:

- If UAM improves Cyber while preserving MMLU, state the exact measured
  changes without claiming significance from one seed.
- If UAM+GU improves or preserves both endpoints relative to UAM, state that
  the result is consistent with complementarity.
- If one endpoint improves and the other worsens, report a trade-off frontier.
- If UAM+GU is dominated, report that residual GU did not help in this setting.
- Numerical orthogonality is mechanism evidence, not endpoint or retention
  proof.
- The Cyber-only result is not generalized to WMDP-Bio.

The rebuttal sentence “using the authors' released code” is replaced by the
paper-derived-port wording in Section 2.

## 15. Acceptance Criteria

The W2 experiment is complete only when:

- the untouched Base passes the compatibility gate;
- all four trainable arms finish 80 optimizer updates;
- every arm has WMDP-Cyber, MMLU, runtime, memory, config, and diagnostics;
- raw JSON values reproduce every final table cell;
- checkpoint and symlink audits pass;
- the full unit/GPU test suite and formatting checks pass;
- the report labels the experiment Cyber-only and approximate-Adam-aware;
- no conclusion exceeds the measured evidence.
