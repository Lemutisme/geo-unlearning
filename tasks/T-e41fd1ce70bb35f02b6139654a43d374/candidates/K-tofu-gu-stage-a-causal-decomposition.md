---
{"id":"K-tofu-gu-stage-a-causal-decomposition","kind":"knowledge","links":[{"ref":"Q-tofu-gu-adaptive-tuning","rel":"motivates"},{"ref":"I-tofu-gu-adaptive-controller","rel":"supports"}],"status":"bounded"}
---

# Bounded causal decomposition of TOFU GU Stage A

## Scope of the claim

The evidence is one seed on TOFU forget10 with Llama-3.1-8B-Instruct, four
objectives, and five ordered interventions: native, Scope-BF16, Scope-FP32,
GU-rank0, and GU-rank8. The four adjacent differences are ordered pipeline
effects, not permutation-invariant attribution.

## Findings

- **CEU:** restricting the scope and changing selected storage precision did
  not move ES-Un, ES-Re, or model utility at displayed precision. Current-only
  projection changed ES-Un by `+0.002480` and MU by `+0.132074`. Adding rank-8
  history changed ES-Un by a further `+0.034527` and MU by `+0.490432`.
  History, not the three-layer scope, explains most of CEU's production-GU
  forgetting/utility shift.
- **GradDiff:** Scope-BF16 changed ES-Un from `0.032509` to `0.929643`; promoting
  the same three tensors to FP32 returned ES-Un to `0.032509` while MU fell to
  zero. The small selected subspace at LR `1e-5` is numerically sensitive to
  parameter-storage precision. Current projection then raised MU by `0.227874`
  at ES-Un cost `+0.013426`; rank-8 history added MU `+0.242745` at ES-Un cost
  `+0.098765`.
- **SimNPO:** the three-layer scope caused ES-Un `+0.280671`; FP32 recovered
  `-0.056413`. Current projection had exactly zero endpoint and correction
  effect, while history changed ES-Un only `+0.001449`. SimNPO weakness is a
  capacity/forget-drive problem, not a projection problem in this setting.
- **UNDIAL:** scope changed ES-Un only `+0.010830` while increasing MU by
  `0.497274`; FP32 recovered ES-Un `-0.007664`. Current projection and history
  slightly improved forgetting and utility. UNDIAL is counterevidence against
  a universal claim that GU projection necessarily weakens unlearning.

## Mechanistic interpretation

Write the ordinary proposal as

```math
\Delta_0(k,d,\gamma,\alpha,T)
=P_k\,Q_d\,O(\gamma,\alpha,T),
```

where `k` is trainable-subspace capacity, `d` is storage precision, and `O` is
the objective/optimizer/training-budget map. GU applies the Adam-metric
projection

```math
\Delta^*=\arg\min_\Delta\frac12\lVert\Delta-\Delta_0\rVert_M^2
\quad\text{s.t.}\quad G^T\Delta\le0.
```

The endpoint loss of forgetting can arise before projection (`P_k`, `Q_d`, or
weak forget drive) or from the shadow cost of current/history constraints. The
Stage-A data demonstrate all of these regimes, so one fixed history rank and
one fixed parameter scope cannot be optimal across objectives.

## Counterevidence and limitations

- Identical data order, steps, coefficients, and source revisions rule out those
  variables as explanations of adjacent Stage-A effects, but do not establish
  generality beyond seed0/forget10.
- Scope-BF16 is a real intervention, not a clean mathematical scope effect when
  proposed updates approach BF16 ULP. Scope capacity should therefore be judged
  after controlling FP32 storage.
- Endpoint metrics are nonlinear and the sequential decomposition is
  order-dependent.
- This single-agent analysis cannot admit a high-impact universal mechanism
  without independent seed and reviewer evidence.

## Negative results

- A universal “rank8 history is the cause” explanation is rejected by SimNPO
  and UNDIAL.
- A projector-only interpretation of native-baseline versus GU deltas is
  rejected because scope and storage precision have large measured effects.
- A universal “more epochs” prescription is unsupported: different methods
  occupy distinct capacity-, precision-, and constraint-limited regimes.

