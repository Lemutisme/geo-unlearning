# SimNPO, OrthoGrad-style, and GU MVP comparison

## Scope

This is a single-seed, ten-update MVP comparison, not a convergence or
state-of-the-art claim. All matched arms use seed 0, BF16, FlashAttention 2,
gradient checkpointing with `use_reentrant=false`, `paged_adamw_32bit`,
`adam_beta1=0`, zero weight decay, and no saved model/optimizer checkpoint.
Runs used GPU 0 only.

The primary-table `peak mem` value is the process-level NVML peak above the
pre-launch baseline. Allocator peak and CPU RSS are reported separately below,
because paged AdamW can saturate device memory while moving method-specific
state between CPU and GPU.

## TOFU Forget01

| Method | ES Re. ↑ | ES Un. ↓ | Priv. ↑ | MU ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|
| SimNPO (base) | 0.699751 | 0.515041 | -99.881936 | 0.595549 | 93.004s | 51.042 GiB |
| + OrthoGrad-style (per-sample QR, Euclidean) | 0.696759 | 0.523791 | -99.881936 | 0.594409 | 116.070s | 78.616 GiB |
| + GU, Euclidean metric (Table 5) | 0.698959 | 0.523791 | -99.881936 | 0.596339 | 96.384s | 67.071 GiB |
| + GU, Adam metric (ours) | 0.693896 | 0.534257 | -99.645809 | 0.595739 | 264.828s | 70.007 GiB |

Training/resource detail:

| Method | train wall | train × base | allocator peak | allocator × base | CPU RSS |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 7.058s | 1.00× | 5.902 GiB | 1.00× | 3.433 GiB |
| + OrthoGrad-style | 28.266s | 4.00× | 42.087 GiB | 7.13× | 3.723 GiB |
| + GU, Euclidean | 11.467s | 1.62× | 20.213 GiB | 3.42× | 3.468 GiB |
| + GU, Adam | 178.573s | 25.30× | 23.148 GiB | 3.92× | 8.235 GiB |

At ten updates, none of the three geometric methods improves the overall TOFU
trade-off over SimNPO. Euclidean GU gives the highest MU by 0.000790, but its
forget extraction strength is worse by 0.008750. Adam GU improves the privacy
number but has the weakest ES Re. and ES Un. point estimates.

## MUSE News

MUSE uses its native four metrics rather than TOFU's ES/MU aggregation.

| Method | Forget KnowMem ↓ | Forget VerbMem ↓ | Priv. ↑ | Retain KnowMem ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|
| SimNPO (base) | 0.630270 | 0.582929 | -99.811083 | 0.542849 | 276.505s | 79.249 GiB |
| + OrthoGrad-style (per-sample QR, Euclidean) | 0.638497 | 0.582492 | -99.811083 | 0.550642 | 2223.415s | 79.249 GiB |
| + GU, Euclidean metric (Table 5) | 0.645270 | 0.576867 | -99.811083 | 0.564450 | 641.755s | 79.249 GiB |
| + GU, Adam metric (ours) | 0.645270 | 0.588225 | -99.811083 | 0.550088 | 825.027s | 79.249 GiB |

Training/resource detail:

| Method | train wall | train × base | allocator peak | allocator × base | CPU RSS |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 78.314s | 1.00× | 27.713 GiB | 1.00× | 16.789 GiB |
| + OrthoGrad-style | 2021.233s | 25.81× | 39.137 GiB | 1.41× | 171.292 GiB |
| + GU, Euclidean | 438.498s | 5.60× | 30.005 GiB | 1.08× | 70.206 GiB |
| + GU, Adam | 617.928s | 7.89× | 30.493 GiB | 1.10× | 95.543 GiB |

Euclidean GU has the best Retain KnowMem (+0.021602) and Forget VerbMem
(-0.006062) point estimates, but Forget KnowMem becomes worse by 0.015000.
OrthoGrad and Adam GU also worsen Forget KnowMem while adding substantial
training and host-memory cost. This is a metric trade-off, not a clean win.

## WMDP Cyber

WMDP uses task accuracy for forgetting and full MMLU accuracy for utility.

| Method | WMDP accuracy ↓ | MMLU accuracy ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 0.244590 | 0.256089 | 1095.532s | 79.249 GiB |
| + OrthoGrad-style (per-sample QR, Euclidean) | 0.248616 | 0.235365 | 2462.667s | 79.249 GiB |
| + GU, Euclidean metric (Table 5) | 0.240060 | 0.236220 | 1330.220s | 79.249 GiB |
| + GU, Adam metric (ours) | 0.257172 | 0.254024 | 1720.562s | 79.249 GiB |

Training/resource detail:

| Method | train wall | train × base | allocator peak | allocator × base | CPU RSS |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 63.989s | 1.00× | 27.766 GiB | 1.00× | 10.347 GiB |
| + OrthoGrad-style | 1422.412s | 22.23× | 40.853 GiB | 1.47× | 179.921 GiB |
| + GU, Euclidean | 287.737s | 4.50× | 31.148 GiB | 1.12× | 66.085 GiB |
| + GU, Adam | 678.185s | 10.60× | 31.637 GiB | 1.14× | 92.921 GiB |

The result exposes a clear frontier rather than a dominating method:

- Euclidean GU has the lowest WMDP accuracy (0.240060), but MMLU falls by
  0.019869 versus SimNPO.
- Adam GU nearly recovers SimNPO's MMLU (difference -0.002065, smaller than one
  combined standard error), but WMDP accuracy worsens by 0.012582.
- OrthoGrad is dominated here: worse WMDP and MMLU point estimates than SimNPO,
  with 22.23× training time and 179.921 GiB peak CPU RSS.

WMDP standard errors were approximately 0.0096--0.0098 and MMLU standard errors
approximately 0.0036--0.0037. Therefore the WMDP point-estimate differences are
not decisive at one seed, while the roughly two-point MMLU drops for OrthoGrad
and Euclidean GU are much larger than their evaluation standard errors.

## Geometry and overhead diagnosis

- WMDP OrthoGrad used a global rank-2 basis over 7,241,732,096 selected
  parameters. The basis alone was 53.955 GiB; its maximum relative residual was
  `2.88e-6`, so the high cost is from the intended full-parameter geometry, not
  a failed QR. Its mean preserved candidate-signal ratio was 0.642.
- WMDP Euclidean GU had a 70% conflict rate, mean relative surgery magnitude
  0.589, and maximum residual `2.87e-8`.
- WMDP Adam GU had a 90% conflict rate, mean relative surgery magnitude 0.0719,
  and maximum optimizer-metric residual `3.92e-9`. It is the current
  approximate optimizer-aware implementation, not a strict custom Adam step.
- On MUSE and WMDP, all four methods reached the same 79.249 GiB NVML peak
  because paged AdamW saturated the device. Allocator and CPU measurements are
  therefore more informative about method-specific memory overhead.

## Limitations and blocked data

- These are ten-update, one-seed feasibility runs. They do not establish
  convergence, statistical superiority, or a paper-level quality claim.
- WMDP Bio was not run because its official forget corpus is access restricted.
  No WMDP questions or synthetic substitute were used as training data. It
  remains blocked until an authorized `WMDP_BIO_FORGET_SOURCE` or
  `WMDP_BIO_FORGET_URL` is supplied.
- TOFU Forget05/Forget10 and MUSE Books are not included in this MVP table.

## Artifact provenance

All result directories contain the Hydra config, raw summary, training profile,
process-tree profile, diagnostics where applicable, and no model/optimizer
checkpoint payload:

- TOFU: `saves/exp/ORTHOGRAD_MVP/smoke0712c`, `smoke0712ortho`,
  `smoke0712gue`, `smoke0712gua`
- MUSE: `saves/exp/ORTHOGRAD_MVP/smoke0712mnewsctlc`,
  `smoke0712mnewsortho16`, `smoke0712mnewsgue`, `smoke0712mnewsgua`
- WMDP: `saves/exp/ORTHOGRAD_MVP/smoke0712wcyberctld`,
  `smoke0712wcyberortho`, `smoke0712wcybergue`, `smoke0712wcybergua`

The WMDP corpus is pinned to `cais/wmdp-corpora` revision
`daf89fa9b618b63a624228061a9cebacca88009c` with per-file SHA-256 provenance.
