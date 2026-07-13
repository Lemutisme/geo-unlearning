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

## TOFU Forget05

| Method | ES Re. ↑ | ES Un. ↓ | Priv. ↑ | MU ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|
| SimNPO (base) | 0.692693 | 0.673105 | -99.925502 | 0.594979 | 148.815s | 58.104 GiB |
| + OrthoGrad-style (per-sample QR, Euclidean) | 0.697126 | 0.671235 | -99.917660 | 0.595801 | 170.242s | 78.724 GiB |
| + GU, Euclidean metric (Table 5) | 0.683791 | 0.673752 | -99.921581 | 0.595255 | 153.494s | 72.464 GiB |
| + GU, Adam metric (ours) | 0.689061 | 0.676703 | -99.921581 | 0.596610 | 204.944s | 75.401 GiB |

Training/resource detail:

| Method | train wall | train × base | allocator peak | allocator × base | CPU RSS |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 7.067s | 1.00× | 5.853 GiB | 1.00× | 3.284 GiB |
| + OrthoGrad-style | 28.395s | 4.02× | 42.088 GiB | 7.19× | 3.257 GiB |
| + GU, Euclidean | 11.383s | 1.61× | 20.132 GiB | 3.44× | 3.730 GiB |
| + GU, Adam | 63.271s | 8.95× | 23.067 GiB | 3.94× | 8.100 GiB |

OrthoGrad is the only arm that improves all four point estimates versus the
matched SimNPO run on Forget05, but the changes are small and come with 4.02×
training time and 7.19× allocator memory. Adam GU gives the highest MU, but
worsens both extraction-strength metrics. Euclidean GU improves MU and privacy
while reducing ES Re. and worsening ES Un.

## TOFU Forget10

| Method | ES Re. ↑ | ES Un. ↓ | Priv. ↑ | MU ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|
| SimNPO (base) | 0.699726 | 0.671113 | -99.384503 | 0.598477 | 215.635s | 48.208 GiB |
| + OrthoGrad-style (per-sample QR, Euclidean) | 0.691081 | 0.673885 | -99.380454 | 0.596163 | 238.269s | 78.044 GiB |
| + GU, Euclidean metric (Table 5) | 0.694587 | 0.672414 | -99.401713 | 0.597168 | 223.544s | 63.573 GiB |
| + GU, Adam metric (ours) | 0.688836 | 0.668186 | -99.397664 | 0.595050 | 267.038s | 66.507 GiB |

Training/resource detail:

| Method | train wall | train × base | allocator peak | allocator × base | CPU RSS |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 7.152s | 1.00× | 5.877 GiB | 1.00× | 3.321 GiB |
| + OrthoGrad-style | 28.138s | 3.93× | 42.088 GiB | 7.16× | 3.629 GiB |
| + GU, Euclidean | 12.450s | 1.74× | 20.158 GiB | 3.43× | 3.677 GiB |
| + GU, Adam | 58.445s | 8.17× | 23.093 GiB | 3.93× | 8.135 GiB |

No geometric arm dominates SimNPO on Forget10. Adam GU has the lowest ES Un.
by 0.002927, but loses 0.010890 ES Re. and 0.003427 MU. OrthoGrad and
Euclidean GU both worsen ES Un. and MU; Euclidean GU also has the weakest
privacy point estimate of the four arms.

## MUSE News

MUSE uses its native four metrics rather than TOFU's ES/MU aggregation.

| Method | Forget KnowMem ↓ | Forget VerbMem ↓ | PrivLeak ↑ | Retain KnowMem ↑ | wall-clock | peak mem |
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

## MUSE Books

| Method | Forget KnowMem ↓ | Forget VerbMem ↓ | PrivLeak ↑ | Retain KnowMem ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|
| SimNPO (base) | 0.458212 | 0.997016 | -56.804734 | 0.658574 | 277.511s | 79.249 GiB |
| + OrthoGrad-style (per-sample QR, Euclidean) | 0.455879 | 0.997016 | -56.823225 | 0.658574 | 2228.590s | 79.249 GiB |
| + GU, Euclidean metric (Table 5) | 0.458212 | 0.997016 | -56.804734 | 0.658574 | 620.977s | 79.249 GiB |
| + GU, Adam metric (ours) | 0.458212 | 0.997016 | -56.804734 | 0.658931 | 800.479s | 79.249 GiB |

Training/resource detail:

| Method | train wall | train × base | allocator peak | allocator × base | CPU RSS |
|---|---:|---:|---:|---:|---:|
| SimNPO (base) | 78.832s | 1.00× | 27.713 GiB | 1.00× | 16.032 GiB |
| + OrthoGrad-style | 2023.691s | 25.67× | 39.137 GiB | 1.41× | 171.236 GiB |
| + GU, Euclidean | 416.885s | 5.29× | 30.005 GiB | 1.08× | 70.081 GiB |
| + GU, Adam | 598.380s | 7.59× | 30.493 GiB | 1.10× | 95.315 GiB |

OrthoGrad lowers Forget KnowMem by 0.002333 but worsens PrivLeak by 0.018491;
its other two metrics are unchanged at the reported precision. Euclidean GU
matches all four SimNPO metrics, while Adam GU changes only Retain KnowMem by
+0.000357. These small or absent metric differences do not justify the 5.29×
to 25.67× training overhead in this ten-update run.

The diagnostics explain the near-baseline results. OrthoGrad preserves 99.969%
of the candidate-gradient norm on average despite its rank-2, 50.205 GiB CPU
basis. Euclidean and Adam GU both detect conflicts in 80% of updates, but their
mean relative surgery magnitudes are only 0.0153 and 0.00139, respectively.

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

## Artifact provenance

All result directories contain the Hydra config, raw summary, training profile,
process-tree profile, diagnostics where applicable, and no model/optimizer
checkpoint payload:

- TOFU: `saves/exp/ORTHOGRAD_MVP/smoke0712c`, `smoke0712ortho`,
  `smoke0712gue`, `smoke0712gua`, `smoke0712t05ctl`, `smoke0712t05ortho`,
  `smoke0712t05gue`, `smoke0712t05gua`, `smoke0712t10ctl`,
  `smoke0712t10ortho`, `smoke0712t10gue`, `smoke0712t10gua`
- MUSE: `saves/exp/ORTHOGRAD_MVP/smoke0712mnewsctlc`,
  `smoke0712mnewsortho16`, `smoke0712mnewsgue`, `smoke0712mnewsgua`,
  `smoke0713mbooksctl`, `smoke0713mbooksortho`, `smoke0713mbooksgue`,
  `smoke0713mbooksgua`
- WMDP: `saves/exp/ORTHOGRAD_MVP/smoke0712wcyberctld`,
  `smoke0712wcyberortho`, `smoke0712wcybergue`, `smoke0712wcybergua`

The WMDP corpus is pinned to `cais/wmdp-corpora` revision
`daf89fa9b618b63a624228061a9cebacca88009c` with per-file SHA-256 provenance.
