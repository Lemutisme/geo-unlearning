# Derivation Package

## Target

Derive two implementable forms of Geometric Unlearning (GU):

1. **Stage A:** a paper-faithful projection of the current optimizer-preconditioned candidate direction, before Adam/AdamW momentum and decoupled weight decay;
2. **Stage B:** a projection of the realized optimizer parameter delta, so the first-order retain condition applies to the actual selected-parameter update.

The empirical objective is lexicographic Pareto safety: first satisfy retain, privacy, and utility floors relative to the corresponding baseline, then choose the candidate with the strongest forgetting.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

Stage A is coherent when the preconditioner is frozen during projection and the protected retain gradients are represented by the current low-rank basis. Stage B removes the momentum/weight-decay gap by correcting the realized delta, but its optimizer state remains an internal proposal state rather than the state of an unconstrained AdamW trajectory.

## Invariant Object

The invariant object is the first-order retain change caused by the parameter delta:

\[
D L_r(\theta)[\Delta\theta]=\bar g_r^\top\Delta\theta,
\qquad \bar g_r:=\nabla_\theta L_r.
\]

All coordinate changes, bases, and projections must preserve the meaning of this scalar. A metric is used to choose the nearest or steepest safe direction; it does not change the retain derivative itself.

## Assumptions

- Losses are differentiable at the current parameters.
- During one Stage-A projection, the diagonal preconditioner is frozen and strictly positive.
- Auxiliary component gradients use the same microbatches, reductions, accumulation scaling, and distributed averaging as the training objective they represent.
- The selected parameter tensors are ordinary replicated parameters under single-process training or native Accelerate DDP. DeepSpeed ZeRO and FSDP are outside the first implementation.
- Gradient checkpointing, when enabled, is non-reentrant so `torch.autograd.grad` is supported.
- A full-model safety statement requires either projecting every trainable tensor or freezing tensors outside the projected set. Otherwise, the guarantee is explicitly restricted to the selected tensors.
- Stage B observes and corrects every realized optimizer delta on the protected tensors.

## Notation

- \(\bar g_f=\nabla_\theta L_f\): raw forget gradient.
- \(\bar g_r=\nabla_\theta L_r\): raw retain-objective gradient.
- \(h=\sqrt{\hat v}+\epsilon\): frozen Adam denominator.
- \(P=\operatorname{Diag}(h^{-1})\): local preconditioner.
- \(H=P^{-1}=\operatorname{Diag}(h)\): optimizer-induced metric.
- \(S=P^{1/2}=\operatorname{Diag}(h^{-1/2})\).
- \(C=H^{1/2}=P^{-1/2}=S^{-1}\).
- \(z=S\bar g\): a raw gradient represented in whitened update coordinates.
- \(Q\): an orthonormal basis in whitened coordinates.
- \(R=[z_{r,1},\ldots,z_{r,m}]\): current protected retain directions in whitened coordinates.
- \(\gamma,\alpha\): the exact coefficients of the selected baseline objective.
- \(\kappa,\tau\): paper-compatible tangent cap and sign threshold.

## Derivation Strategy

Start from the coordinate-free retain derivative, derive the optimizer-induced whitening map, then formulate three progressively stronger projections:

1. the submitted paper's normal plus sign-selected tangent rule;
2. the nearest direction in the convex first-order-safe cone;
3. the same cone projection applied to the realized AdamW parameter delta.

## Derivation Map

1. The Adam denominator defines \(P\), hence the metric \(H=P^{-1}\).
2. The square-root transform \(S=P^{1/2}\) converts the metric problem to Euclidean coordinates.
3. Retain safety becomes a Euclidean dot-product constraint in those coordinates.
4. Stage A projects raw objective gradients and maps the result to an optimizer-equivalent raw gradient.
5. The cone formulation keeps the maximum amount of candidate signal subject to retain non-increase.
6. Stage B applies the same invariant directly to the observed optimizer delta, including momentum and weight decay.

## Main Derivation

### Step 1: Correct optimizer-induced coordinates

For a raw gradient \(\bar g\), the frozen preconditioned direction is

\[
d=P\bar g.
\]

Whitening an update-space direction uses \(C=H^{1/2}=P^{-1/2}\), hence

\[
z=Cd=P^{-1/2}P\bar g=P^{1/2}\bar g=S\bar g.
\]

The inverse maps are

\[
d=Sz,
\qquad
\bar g_{\mathrm{eq}}=Hd=Cz.
\]

Therefore \(P\bar g_{\mathrm{eq}}=Sz=d\). Using \(P\bar g\) as the whitened vector, as the current implementation does, is not equivalent.

### Step 2: Retain derivative in whitened coordinates

Let \(z_r=S\bar g_r\) and let a positive candidate direction \(z\) induce the parameter update \(\Delta\theta=-\eta Sz\). Then

\[
\bar g_r^\top\Delta\theta
=-\eta\bar g_r^\top Sz
=-\eta z_r^\top z.
\]

Thus first-order retain non-increase is exactly

\[
z_r^\top z\ge 0.
\]

For several protected retain directions, the constraint is

\[
R^\top z\succeq 0.
\]

Exact invariance replaces the inequalities with equalities.

### Step 3: Paper-faithful Stage A

Let \(Q^\top Q=I\), with columns spanning the protected retain directions. For

\[
z_f=S\bar g_f,
\qquad
z_r=S\bar g_r,
\]

define

\[
a=Q^\top z_f,
\qquad b=Q^\top z_r,
\qquad z_f^\perp=z_f-Qa.
\]

The submitted sign-aware rule keeps component \(i\) precisely when

\[
a_i b_i>\tau,
\]

because the applied update is the negative direction. Define

\[
z_{\mathrm{keep}}^0=\sum_{i:a_i b_i>\tau}a_iq_i
\]

and

\[
s=\min\!\left(1,
\frac{\kappa\lVert z_f^\perp\rVert_2}
{\lVert z_{\mathrm{keep}}^0\rVert_2}\right),
\qquad
z_{\mathrm{keep}}=s z_{\mathrm{keep}}^0,
\]

with the zero-norm cases defined to return zero tangent motion. If \(z_r\) is represented by \(Q\), the practical candidate is

\[
z_A=
\gamma(z_f^\perp+z_{\mathrm{keep}})
+\beta_{\mathrm{rep}}QQ^\top z_r.
\]

Its first-order retain derivative is non-positive:

\[
\bar g_r^\top(-\eta Sz_A)
=-\eta\left[
\gamma s\sum_{i:a_i b_i>\tau}a_i b_i
+\beta_{\mathrm{rep}}\lVert Q^\top z_r\rVert_2^2
\right]\le0.
\]

The optimizer-equivalent raw gradient written to `p.grad` is

\[
\bar g_A=Cz_A=P^{-1/2}z_A.
\]

This is a pre-momentum candidate guarantee only.

### Step 4: Theory-guided safe-cone Stage A+

The componentwise sign rule is sufficient but not generally the least destructive safe correction. The nearest first-order-safe direction is the convex projection

\[
z_{A+}
=\arg\min_z\frac12\lVert z-z_f\rVert_2^2
\quad\text{subject to}\quad
R^\top z\succeq0.
\]

The feasible set is a closed convex cone containing zero, so the projection exists and is unique. Projection onto this cone is non-expansive:

\[
\lVert z_{A+}\rVert_2\le\lVert z_f\rVert_2.
\]

It retains all candidate tangent combinations that are jointly safe, whereas hard null projection discards the entire tangent component and the paper sign rule treats coordinates independently. Because all constraints lie in a rank-\(k\) subspace, solve only for the tangent coefficients and leave \(z_f^\perp\) unchanged. A deterministic half-space projection solver must report its maximum constraint violation; failure to converge falls back to the pure normal direction.

This is a new experimental mode and must not be presented as the submitted algorithm.

### Step 5: Basis coverage and refresh

The current-step safety residual is controlled by

\[
\epsilon_{\mathrm{cov}}
=\frac{\lVert z_r-QQ^\top z_r\rVert_2}
{\max(\lVert z_r\rVert_2,\varepsilon)}.
\]

To avoid stale coordinates as \(P\) changes, store a FIFO of raw retain gradients, transform them with the current \(S\), and re-orthogonalize in FP32. Always include the current accumulated retain gradient before older anchors. Increase rank or projected-layer coverage when the observed residual or unprotected harmful contribution exceeds its configured tolerance.

### Step 6: Stage B on the realized AdamW delta

Let the optimizer, including momentum, bias correction, and decoupled weight decay, propose

\[
\Delta\theta_{\mathrm{opt}}
=\theta_{\mathrm{after}}-\theta_{\mathrm{before}}.
\]

Whiten the realized delta directly:

\[
y=C\Delta\theta_{\mathrm{opt}}.
\]

Since \(\bar g_r=S^{-1}z_r=Cz_r\),

\[
\bar g_r^\top\Delta\theta_{mathrm{opt}}
=z_r^\top y.
\]

The safe realized delta is therefore

\[
y_B
=\arg\min_y\frac12\lVert y-y_{\mathrm{opt}}\rVert_2^2
\quad\text{subject to}\quad
R^\top y\preceq0,
\]

and

\[
\Delta\theta_B=S y_B.
\]

Writing

\[
\theta_{\mathrm{after}}
\leftarrow\theta_{\mathrm{before}}+\Delta\theta_B
\]

ensures

\[
\bar g_{r,j}^\top\Delta\theta_B\le0
\]

for every represented retain constraint, including the effects of historical momentum and weight decay in the proposed delta. Stage A should run before Stage B so the optimizer proposal is already close to the safe set; the Stage-B correction ratio then diagnoses the practical momentum/decay gap.

## Remarks and Interpretation

- The hard null-space projection is appropriate when exact first-order invariance is required. The cone projection is appropriate for the selected Pareto-safe goal because it permits retain-helpful motion.
- The best empirical configuration is not chosen by averaging unrelated metrics. First enforce floors for ES-Re, Privacy, and MU; among feasible runs, minimize ES-Un.
- Useful online diagnostics are basis coverage, harmful directional mass, cap activation, predicted Stage-A retain derivative, realized Stage-B retain derivative, and Stage-B correction ratio.
- Rank and projected layers should be increased only when diagnostics show uncovered harmful motion. This preserves more forgetting capacity and limits overhead.
- A trust-region continuation is safer than maximizing tangent motion immediately: begin with \(\kappa=0\), then increase it only while the Pareto floors remain satisfied.

## Boundaries and Non-Claims

- Stage A does not guarantee the complete AdamW parameter delta.
- Stage B guarantees only the tensors it corrects. A global statement requires all trainable tensors to be corrected or all other tensors to be frozen.
- Low-rank and minibatch bases approximate the full retain set. The guarantee is exact only for represented constraints.
- Finite-step retain loss can still increase because the guarantee is first order. The quick validation must compare predicted directional change with observed finite-step change.
- Correct code does not establish the paper's reported numbers. Those require new controlled experiments.
- DeepSpeed ZeRO and FSDP require separate sharding-aware implementations.

## Open Risks

- Repeated Stage-B parameter correction leaves optimizer moments as proposal state; this is safe per corrected step but may reduce long-horizon efficiency.
- A forward-KL anchor and an NLL retain objective define different gradients. They must remain separate, and both must be identified in logs.
- The first optimizer step has no historical second moment. The implementation must use an explicit, logged identity-metric fallback rather than silently substituting an unrelated EMA.
- FP16 scaling, native DDP reduction, optimizer recreation, and short accumulation windows require integration tests in the pinned environment.
- Pareto floors estimated on a tiny quick-validation subset are diagnostic, not submission-grade evidence.
