# WMDP-Cyber UAM and Residual-GU W2 Report

Date: 2026-07-13

Branch: `feat/uam-gu`

Matrix: `/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2/w2-20260713-2`

## Result in one sentence

This single-seed WMDP-Cyber experiment does **not** establish UAM+GU
complementarity: relative to UAM, residual GU improves MMLU by `0.000712` but
worsens WMDP-Cyber accuracy by `0.001510`; both changes are smaller than the
corresponding single-evaluation standard errors.

## Scope and provenance

This is a **Cyber-only, paper-derived port of UAM Algorithm 2**, not a full
reproduction of the paper's WMDP experiment. The authors' repository contains
a generic PyTorch/vision UAM implementation, but no released Zephyr/WMDP/RMU
pipeline. We therefore implemented the paper's WMDP specification in this
repository and cross-checked the perturb-and-restore mechanism against the
released generic implementation.

- UAM paper: NeurIPS 2025 paper
  [`86cca1b77a808b7f9ef5a987bcae6803`](https://proceedings.neurips.cc/paper_files/paper/2025/file/86cca1b77a808b7f9ef5a987bcae6803-Paper-Conference.pdf).
- UAM repository:
  [`Harry24k/machine-unlearning-pytorch`](https://github.com/Harry24k/machine-unlearning-pytorch),
  commit `9ebd70de6bd069add730ffde8f0deb39dbb1436b`.
- RMU reference:
  [`centerforaisafety/wmdp`](https://github.com/centerforaisafety/wmdp),
  commit `bc5e1ba0367ea826caeeeaa50656336a1e87acfb`.
- Model and tokenizer: `HuggingFaceH4/zephyr-7b-beta`.
- Retain data: `wikitext/wikitext-2-raw-v1`, split `train`.
- Evaluator: `lm-eval==0.4.8`.
- WMDP-Cyber corpus SHA-256:
  `b5d339ed7f42a9e0dfc00708e516b288363a87512ec9cbdad8703f0ea8f5ea9a`.

No WMDP-Bio data were used. Consequently, these results must not be described
as reproducing the paper's joint Bio/Cyber result or its complete WMDP table.

## Method

The WMDP objective uses the output of transformer layer 7. Only the following
three tensors are trainable:

```text
model.layers.(5|6|7).mlp.down_proj.weight
```

For UAM, the forget representation signal includes seeded Gaussian noise with
standard deviation `0.01`. The temporary parameter perturbation is

\[
\delta=\rho\frac{g_f}{\lVert g_f\rVert_2^2+\varepsilon},
\qquad \rho=5\times10^{-5},
\]

and the retain gradient at the perturbed point is transformed with the
paper-specified Euclidean Householder reflection (`gamma=2`). The ordinary
retain gradient is evaluated at the unperturbed parameters.

For composition, GU is applied only to the UAM-specific residual:

\[
g_r=\nabla L_r(w),\qquad
r_{\mathrm{UAM}}=g_{\mathrm{UAM}}-g_r,
\]

\[
g_{\mathrm{UAM+GU}}
=g_r+P_{g_r}^{\perp,H}r_{\mathrm{UAM}},
\]

when the projected correction passes the forget-benefit sign gate; otherwise
the selected tensors receive `g_r`. This preserves ordinary retain descent and
does not project the entire UAM update.

The UAM reflection is Euclidean. The GU projection is global rank one over the
three selected tensors and uses the current effective batch in an approximate
frozen-PagedAdamW metric. It is not a historical low-rank basis and is not an
exact momentum-aware Adam method.

## Experimental protocol

| Setting | Value |
|---|---|
| GPU | GPU 0, single process |
| Seed | 42 |
| Physical batch / GAS | 1 / 4 |
| Effective batch | 4 |
| Optimizer updates | 80 |
| Learning rate | `5e-5` |
| Precision | BF16 |
| Attention | FlashAttention 2 |
| Optimizer | PagedAdamW32 |
| Adam beta1 / weight decay | 0 / 0 |
| Gradient checkpointing | enabled, `use_reentrant=false` |
| Checkpoints | disabled; output tree audited |

All trainable arms start from the same untouched Zephyr checkpoint and use the
same optimizer, data ordering, effective batch, update count, and evaluator.
RMU uses steering coefficient `20` and retain weight `100`. UAM uses the
paper-derived primary radius `rho=5e-5`; no post-hoc radius selection was
performed.

## Base compatibility gate

The untouched checkpoint was evaluated before training. The fixed gate compares
against the paper's Base values (`0.4400` WMDP-Cyber and `0.5810` MMLU):

| Metric | Measured Base | Absolute delta | Allowed delta | Pass |
|---|---:|---:|---:|:---:|
| WMDP-Cyber | 0.443382 | 0.003382 | 0.030000 | yes |
| MMLU | 0.585387 | 0.004387 | 0.020000 | yes |

This establishes evaluator compatibility only; it does not turn the
Cyber-only port into a full paper reproduction.

## Endpoint results

Relative runtime is training time normalized to RMU. Wall-clock includes both
training and evaluation. Peak memory is process-level NVML memory sampled at
`0.2 s`; CUDA allocator peak was unavailable and is recorded as `null` in the
raw artifacts.

| Method | WMDP-Cyber ↓ | MMLU ↑ | Relative runtime | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|
| Base | 0.4434 | 0.5854 | — | 758.4s | 16486 MiB |
| RMU | 0.2416 | 0.5763 | 1.000x | 979.2s | 33918 MiB |
| RMU + GU | 0.2406 | 0.5784 | 3.149x | 1205.7s | 34710 MiB |
| UAM | 0.4283 | 0.5830 | 6.868x | 1592.0s | 34926 MiB |
| UAM + GU | 0.4298 | 0.5837 | 8.732x | 1799.4s | 37166 MiB |

The unrounded point estimates and standard errors are:

| Method | WMDP-Cyber | stderr | MMLU | stderr |
|---|---:|---:|---:|---:|
| Base | 0.44338198 | 0.01114751 | 0.58538670 | 0.00394982 |
| RMU | 0.24157021 | 0.00960483 | 0.57627119 | 0.00394492 |
| RMU + GU | 0.24056366 | 0.00959116 | 0.57840763 | 0.00394561 |
| UAM | 0.42828384 | 0.01110367 | 0.58296539 | 0.00395402 |
| UAM + GU | 0.42979366 | 0.01110853 | 0.58367754 | 0.00395075 |

## Direct paired comparisons

| Pair | Delta WMDP-Cyber ↓ | Delta MMLU ↑ | Train-time ratio | End-to-end ratio | Peak-memory delta |
|---|---:|---:|---:|---:|---:|
| RMU + GU minus RMU | -0.001007 | +0.002136 | 3.149x | 1.231x | +792 MiB |
| UAM + GU minus UAM | +0.001510 | +0.000712 | 1.271x | 1.130x | +2240 MiB |

For RMU, GU moves both point estimates in the desired direction, but the
changes are smaller than the individual evaluation standard errors. For UAM,
GU produces a trade-off: slightly better MMLU and slightly worse forgetting.
The desired claim that UAM+GU both reduces WMDP-Cyber accuracy and preserves or
improves MMLU is therefore **not supported** by this matrix.

UAM and UAM+GU also forget substantially less than the matched RMU arms at the
primary `rho=5e-5`, although they preserve more MMLU. This is a descriptive
single-seed comparison, not a superiority or significance claim.

## Geometry diagnostics

| Arm | Geometry updates | Projection/replay calls | Gate/conflict | Maximum relative orthogonality residual | Actual-delta probes |
|---|---:|---:|---:|---:|---:|
| RMU | 80 | 80 finalizers | — | — | 2 |
| RMU + GU | 80 | 80 projections | 61 conflict steps | `3.7789e-9` | 2 |
| UAM | 80 | 80 replays | — | — | 2 |
| UAM + GU | 80 | 80 replays | 45 corrections kept | `1.1848e-8` | 2 |

RMU+GU encountered one zero-retain-gradient first-step fallback. All 80
UAM+GU records passed the orthogonality safety check, including gated-off
steps. The maximum residual is well below the configured `1e-6` threshold.

The candidate update's predicted forget directional derivative was positive
on `43/80` UAM steps and `70/80` UAM+GU steps. Its median changed from
`1.9597e-9` for UAM to `0.7311` for UAM+GU. Thus the residual projection and
sign gate materially changed the local candidate geometry, but that local
diagnostic did not translate into a better final WMDP-Cyber endpoint.

This distinction is important: near-zero current-batch orthogonality verifies
the implemented projection, not retain invariance on unseen examples or
endpoint improvement after many nonlinear optimizer steps.

## Interpretation

The analyzer classifies this matrix as `trade_off`.

1. **Observation:** RMU achieves much stronger Cyber forgetting than UAM at
   the tested primary radius. **Interpretation:** the implemented UAM
   candidate is weak in this Cyber-only representation setting. **Implication:**
   GU cannot recover a strong endpoint merely by filtering a weak candidate.
2. **Observation:** UAM+GU improves the local predicted forget direction on
   many steps but ends with Cyber accuracy `0.001510` above UAM.
   **Interpretation:** current-effective-batch rank-one, first-order geometry
   does not control cross-batch drift, curvature, or the final benchmark.
   **Implication:** mechanism correctness and endpoint benefit must be claimed
   separately.
3. **Observation:** residual GU adds `27.1%` to UAM training time and
   `2240 MiB` to peak NVML memory in this run. **Interpretation:** replay plus
   separate component gradients are plausible sources of the overhead, but
   this run did not profile their individual contributions. **Implication:**
   the current implementation demonstrates feasibility with PagedAdamW32 and
   FlashAttention 2, but not a favorable efficiency trade-off.

## Claim boundary and next experiments

The evidence supports the following statement:

> We implemented a Cyber-only, paper-derived UAM Algorithm 2 port and a
> residual-only UAM+GU composition. The projection satisfies its intended
> current-batch optimizer-coordinate constraint, but at `rho=5e-5` the
> single-seed endpoint is a small trade-off rather than evidence of
> complementarity.

It does not support a claim of statistically significant improvement,
generalization to WMDP-Bio, exact optimizer-aware GU, or complete reproduction
of the UAM paper. The most informative follow-ups are a predeclared rho sweep,
at least three seeds, joint WMDP-Bio/Cyber evaluation, and a held-out or
historical retain subspace. These should be reported regardless of whether
they favor UAM+GU.

## Artifact audit

- Manifest: 5/5 arms exited with code 0.
- Each trainable arm recorded exactly 80 geometry updates and two full
  selected-parameter actual-delta probes.
- No checkpoint directories, model weights, optimizer state, scheduler state,
  RNG state, Trainer state, or symlinks are present in the matrix tree.
- Raw generated table:
  `/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2/w2-20260713-2/W2_TABLE.md`.
- Machine-readable summary:
  `/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2/w2-20260713-2/W2_SUMMARY.json`.

All endpoint values in this report were read from the raw lm-eval summaries;
no rescaling, averaging across seeds, or post-hoc arm removal was applied.
