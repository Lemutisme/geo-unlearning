---
{"id":"I-tofu-gu-adaptive-controller","kind":"idea","links":[{"ref":"K-tofu-gu-stage-a-causal-decomposition","rel":"derived_from"},{"ref":"Q-tofu-gu-adaptive-tuning","rel":"addresses"}],"status":"proposed"}
---

# Objective-adaptive, budgeted GU controller

## Principle

Choose the least restrictive intervention that satisfies explicit endpoint
floors. Do not hard-code scope, precision, history rank, and objective retain
weight as one universal configuration.

## Diagnostics

1. **Update-to-ULP ratio**

```math
q_i=|\Delta_{0,i}|/\operatorname{ulp}(\theta_i).
```

Promote selected storage to FP32 when a material fraction of proposed BF16
updates has `q_i<1`.

2. **Forget-capacity coverage**

```math
C_f(k)=\frac{\lVert P_kM^{-1/2}g_f\rVert_2^2}
{\lVert M^{-1/2}g_f\rVert_2^2}.
```

Increase the trainable scope only when the smaller scope lacks forget-gradient
capacity and endpoint floors remain satisfied.

3. **Constraint shadow cost**

Use active multipliers and correction ratio to estimate the marginal forget
cost of current and historical constraints. Add a historical covector only when
it is nonredundant and predicted to be violated; retire persistently inactive
or redundant history.

## Candidate update

Construct a forget-driven optimizer proposal, then solve

```math
\min_\Delta\frac12\lVert\Delta-\Delta_0\rVert_M^2
\quad\text{s.t.}\quad
g_{r,j}^T\Delta\le b_j,
\quad \lVert\Delta\rVert_M\le R.
```

Current-gradient normalization must retain its raw norm so `b_j` remains in
retain-loss units. The final authority is an actual finite-step retain-NLL
filter with budgets `1e-4`, `5e-4`, and `1e-3`, backtracking scales
`1, 0.5, 0.25, 0.125, 0.0625`, and no scale above the ordinary proposal.

## Control policy

1. select FP32 when update-to-ULP evidence requires it;
2. select the smallest scope meeting forget-capacity coverage;
3. remove objective retain loss only when GU alone maintains endpoint floors;
4. start with current-only projection;
5. add the minimum active history rank required by retain endpoints;
6. allow a nonzero retain-loss budget only after strict candidates fail to
   recover forgetting;
7. choose candidates lexicographically and validate with three seeds.

## Falsifiable predictions

- CEU/GradDiff should exhibit a rank-dependent forget/utility frontier.
- GradDiff `alpha=0` should reduce correction burden or improve ES-Un at a
  feasible intermediate rank.
- SimNPO should respond primarily to scope/gamma, not history.
- UNDIAL should show little benefit from additional restriction.

If these predictions fail, the controller is rejected rather than tuned around
the negative result.

