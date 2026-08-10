---
{"id":"Q-tofu-gu-adaptive-tuning","kind":"question","links":[{"ref":"K-tofu-gu-stage-a-causal-decomposition","rel":"grounded_by"},{"ref":"I-tofu-gu-adaptive-controller","rel":"tests"}],"status":"open"}
---

# Can objective-adaptive GU improve forgetting without violating endpoint floors?

## Key research question

Can method-specific choices of history rank, trainable scope, forget/retain
objective ratio, and finite retain-loss budget achieve lower TOFU ES-Un than
production GU-rank8 while preserving its ES-Re, privacy, and model utility?

## Stage-B falsification matrix

### CEU

- `history_rank = 1, 2, 4` with the production three-layer FP32 scope.

Prediction: intermediate ranks trace a monotone forgetting/utility frontier
between rank0 and rank8. Failure of monotonicity falsifies rank count as a
sufficient history-control statistic.

### GradDiff

- `history_rank = 1, 2, 4` with `gamma=1, alpha=1`;
- at the best feasible rank, `alpha=0, gamma=1` and `alpha=0, gamma=2`.

Prediction: removing the explicit retain term recovers forget drive because GU
already enforces retain constraints. If endpoint retain floors fail despite
active GU, objective retain regularization is not redundant.

### SimNPO

- scope layers `26--31` and `20--31` at production coefficients;
- `gamma=0.5` and `gamma=1.0` at the three-layer scope;
- `gamma=0.5, alpha=0` and 20 epochs;
- combine the best feasible scope and gamma.

Prediction: scope/gamma changes move ES-Un materially while history rank does
not. Failure falsifies the Stage-A capacity/drive explanation.

### UNDIAL

Use production GU-rank8 as a healthy reference; do not spend tuning budget
unless another arm challenges its Pareto position.

## Lexicographic admission rule

Relative to production GU-rank8, a candidate advances only if:

1. ES-Un improves by at least `0.05` absolute;
2. ES-Re decreases by at most `0.02`;
3. model utility decreases by at most `0.02`;
4. privleak decreases by at most `2.0` points;
5. KKT, endpoint, provenance, GPU, and no-checkpoint validation pass.

Among feasible candidates choose the lowest ES-Un. Do not average unrelated
metrics and do not relax floors after observing results. Promoted candidates
must run seeds 1 and 2 against their matched control.

