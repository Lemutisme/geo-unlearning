# GU vs. UAM and Residual-GU-UAM

Date: 2026-07-12

## Scope and conclusion

This note compares the existing approximate-Adam GU runs with the UAM and
Residual-GU-UAM smoke run on TOFU `forget01` using Llama-3.2-1B-Instruct.
Every result is a single-seed, ten-step mechanism check.

The strongest conclusion supported by the existing artifacts is:

- both GU and Residual-GU-UAM execute their intended optimizer-coordinate
  projection with residuals around `1e-9` to `1e-8`;
- GU alone has either negligible or mixed endpoint impact in its matched
  pairs;
- adding residual GU to NLL-UAM changes the endpoints more strongly and
  improves all four reported forget proxies, but also reduces model utility
  and retain extraction;
- adding residual GU to SimNPO-UAM is mixed: probability and exact
  memorization improve, while forget ROUGE, extraction, and retain extraction
  become worse.

These runs do **not** establish that UAM+GU is superior to GU. The most
systems-similar GU run used warmup while UAM explicitly disabled it, and no
comparison has multiple seeds.

## What is and is not matched

| Setting | Adam GU MVP | Production GU | UAM / UAM+GU |
|---|---|---|---|
| Model/data | Llama-3.2-1B, TOFU01 | Llama-3.2-1B, TOFU01 | Llama-3.2-1B, TOFU01 |
| Updates | 10 | 10 | 10 |
| Batch / GAS | 4 / 8 | 4 / 8 | 4 / 8 |
| Learning rate | `1e-5` | `1e-5` | `1e-5` |
| Precision | FP32 | BF16 | BF16 |
| Optimizer | Torch AdamW | PagedAdamW32 | PagedAdamW32 |
| Adam beta1 / weight decay | 0 / 0 | 0 / 0 | 0 / 0 |
| Attention | eager | FlashAttention 2 | FlashAttention 2 |
| Warmup epochs | 1 | 1 | 0 |
| Step-1 parameter delta | not recorded | zero | positive |
| Candidate being constrained | SimNPO forget gradient | SimNPO forget gradient | UAM residual correction |

The Production GU run is the closest systems match to UAM, but its first
parameter update is zero because its live learning rate starts at zero.
Consequently, raw Production-GU-vs-UAM endpoints are descriptive only. The
scientifically safer comparisons are the within-run paired deltas below.

## Raw endpoint table

Arrows state the desired direction for this comparison. Lower forget
probability, forget ROUGE, extraction, and exact memorization indicate
stronger forgetting; higher utility and retain extraction indicate better
retention.

| Arm | Utility ↑ | Forget prob ↓ | Forget ROUGE ↓ | Extraction ↓ | Exact mem ↓ | Retain extraction ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Adam GU control | 0.59810108 | 0.54925340 | 0.46035547 | 0.17449213 | 0.86395429 | 0.64105061 |
| Adam GU | 0.59810230 | 0.54958175 | 0.46035547 | 0.17449213 | 0.86395429 | 0.64105061 |
| Production control | 0.59763752 | 0.80246283 | 0.65090876 | 0.41234032 | 0.94391604 | 0.71956500 |
| Production GU | 0.59693684 | 0.80286219 | 0.66116937 | 0.39770796 | 0.94397661 | 0.71737996 |
| UAM NLL | 0.59977039 | 0.77953623 | 0.68803526 | 0.39998844 | 0.93711246 | 0.72652788 |
| UAM+GU NLL | 0.59816359 | 0.77012380 | 0.67904523 | 0.39743089 | 0.93363988 | 0.72373573 |
| UAM SimNPO | 0.59886567 | 0.76570623 | 0.61668112 | 0.38286624 | 0.93440222 | 0.73205127 |
| UAM+GU SimNPO | 0.59870161 | 0.75960643 | 0.65024587 | 0.39060631 | 0.93114177 | 0.72324654 |

## Paired incremental effects

Each row is `projected method − its matched unprojected method`. For the four
forget columns, a negative delta is favorable. For utility and retain
extraction, a positive delta is favorable.

| Pair | Utility ↑ | Forget prob ↓ | Forget ROUGE ↓ | Extraction ↓ | Exact mem ↓ | Retain extraction ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Adam GU − matched control | +0.00000122 | +0.00032835 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |
| Production GU − matched control | -0.00070068 | +0.00039936 | +0.01026060 | -0.01463235 | +0.00006057 | -0.00218504 |
| UAM+GU NLL − UAM NLL | -0.00160680 | -0.00941244 | -0.00899003 | -0.00255755 | -0.00347258 | -0.00279214 |
| UAM+GU SimNPO − UAM SimNPO | -0.00016406 | -0.00609980 | +0.03356475 | +0.00774007 | -0.00326046 | -0.00880472 |

### Reading the paired effects

1. **Adam GU MVP is numerically active but empirically almost inert.** Its
   maximum endpoint difference is `3.28e-4`, and five of the six selected
   metrics are unchanged at the shown precision. This agrees with the earlier
   conclusion that the repaired GU path works but the ten-step endpoint does
   not demonstrate a benefit.

2. **Production GU is mixed.** It reduces extraction by `0.01463`, but forget
   probability and ROUGE increase, while utility and retain extraction fall.
   This is not a consistent movement along a single forget-retain frontier.

3. **Residual GU contributes a coherent NLL-UAM forgetting correction.** All
   four forget proxies decrease relative to NLL-UAM. The cost is a utility
   decrease of `0.00161` and retain-extraction decrease of `0.00279`. Thus the
   correction is geometrically retain-orthogonal on the current batch, but it
   is not empirically retain-invariant over ten nonlinear optimizer steps.

4. **The SimNPO residual is signal-sensitive.** Forget probability and exact
   memorization improve, but ROUGE and extraction worsen. Retain extraction
   decreases by `0.00880`, the largest paired retain change in the table. The
   current SimNPO residual gate therefore does not yet give a robust empirical
   trade-off.

## Geometry and execution diagnostics

| Method | Finalizers | Gate/conflict rate | Maximum relative orthogonality residual | Interpretation |
|---|---:|---:|---:|---|
| Adam GU MVP | 10 | n/a | `3.44e-8` | Frozen-Adam global rank-one projection executes |
| Production GU | 10 | conflict `1.0` | `1.74e-9` | GU constraint is numerically satisfied |
| UAM+GU NLL | 10 | gate `0.4` | `3.54e-9` | Safe normal correction retained on 4/10 updates |
| UAM+GU SimNPO | 10 | gate `0.2` | `1.40e-9` | Safe normal correction retained on 2/10 updates |

The diagnostics separate mechanism correctness from endpoint quality. Near-zero
orthogonality establishes that the selected correction is normal to the
current retain direction in the frozen optimizer metric. It does not guarantee
retention on unseen retain examples, after Adam changes state, or after
higher-order effects accumulate.

## Interpretation

The current evidence suggests that GU needs a useful candidate direction.
Projecting the ordinary SimNPO forget gradient produced little or mixed change,
whereas projecting only UAM's residual correction produced a measurable normal
component on a minority of updates. With the NLL forget signal, that component
moved every selected forget proxy in the desired direction. This is consistent
with the residual-GU design: retain descent is preserved and only the
forget-aware UAM correction is constrained.

The retain degradation has three plausible, non-exclusive explanations:

- the constraint is rank one and batch-local rather than a broad per-example
  retain subspace;
- the geometry freezes the pre-step Adam denominator and is approximate after
  the optimizer updates its state;
- first-order orthogonality does not control curvature, cross-batch drift, or
  ten-step cumulative effects.

## Required matched follow-up

The next comparison should launch one matrix from the same commit and Hydra
base with `warmup_epochs=0`, BF16, FlashAttention 2, PagedAdamW32, beta1 0,
weight decay 0, GAS 8, and identical seeds:

1. SimNPO control;
2. GU;
3. UAM-NLL and UAM-SimNPO;
4. Residual-GU-UAM-NLL and Residual-GU-UAM-SimNPO.

At least three seeds and more than ten updates are required before making a
performance claim. Report paired deltas, confidence intervals, gate rate,
normal-component norm, actual parameter-delta retain dot, and both current-
batch and held-out-retain changes.

## Provenance

- Adam GU report: `GU_ADAM_MVP_REPORT.md`
- Production GU matrix: `saves/exp/PCGRAD_SMOKE/0711204008`
- UAM matrix: `saves/exp/UAM_SMOKE/0712145130`
- UAM mechanism report: `UAM_SMOKE_REPORT.md`
- UAM analysis JSON: `saves/exp/UAM_SMOKE/0712145130/analysis.json`

All endpoint numbers above were recomputed from the corresponding
`TOFU_SUMMARY.json` files. No averaging, rescaling, or multi-seed aggregation
was applied.
