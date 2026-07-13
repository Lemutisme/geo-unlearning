# WMDP-Cyber Strong UAM Pair Design

**Date:** 2026-07-13
**Branch:** `feat/uam-gu`
**Status:** Approved for implementation and launch

## Goal

Test whether the weak UAM endpoint in the completed W2 matrix is primarily an
under-training effect. Run exactly one stronger, matched pair on GPU 0:

1. UAM;
2. residual UAM+GU.

The comparison changes learning rate and update count together to obtain a
clear stress test, not a factorial attribution study.

## Configuration

Both arms start from `HuggingFaceH4/zephyr-7b-beta` and reuse the completed W2
pipeline, data, seed, evaluator, objectives, and system stack.

| Setting | W2 primary | Strong pair |
|---|---:|---:|
| Learning rate | `5e-5` | `1e-4` |
| Optimizer updates | 80 | 160 |
| UAM rho | `5e-5` | `5e-5` |
| Noise standard deviation | `0.01` | `0.01` |
| Physical batch / GAS | 1 / 4 | 1 / 4 |
| Optimizer | PagedAdamW32 | PagedAdamW32 |
| Adam beta1 / weight decay | 0 / 0 | 0 / 0 |
| Precision / attention | BF16 / FlashAttention 2 | BF16 / FlashAttention 2 |
| Seed | 42 | 42 |

The linear learning-rate schedule makes the approximate integrated learning
rate four times the 80-step primary run. This is deliberately stronger while
leaving the UAM perturbation definition unchanged.

Forget-loss scalar multiplication is not used. Under fixed-loss
normalization,

\[
\delta=\rho\frac{g_f}{\lVert g_f\rVert^2+\epsilon},
\]

scaling the objective also scales the denominator and is not a clean UAM
strength control.

## Execution and outputs

- Run only `uam` and `uam_gu`, sequentially on GPU 0.
- Activate the `unlearning` conda environment.
- Use a new immutable output root under
  `saves/exp/WMDP_UAM_W2_STRONG/<run-id>`.
- Disable and audit all checkpoints, weights, optimizer state, scheduler state,
  RNG state, Trainer state, and symlinks.
- Record the exact Hydra config, logs, geometry, lm-eval summary, wall time,
  NVML peak, and CPU RSS for each arm.
- Reuse the already validated untouched Base from
  `WMDP_UAM_W2/w2-20260713-2`; do not spend GPU time reevaluating it.

## Safety gates

Each arm must:

- finish exactly 160 optimizer updates;
- record exactly 160 UAM finalizations and replays;
- produce finite objectives, gradients, perturbations, and final updates;
- restore temporary perturbations exactly;
- keep every retained residual-GU correction below the existing `1e-6`
  relative orthogonality threshold;
- publish artifacts atomically only after checkpoint audit success.

Failure of UAM stops UAM+GU. Existing result directories are never
overwritten.

## Analysis

Report raw WMDP-Cyber accuracy, MMLU accuracy, their lm-eval standard errors,
training time, end-to-end wall time, and peak NVML memory. Compare:

1. strong UAM against primary UAM;
2. strong UAM+GU against primary UAM+GU;
3. strong UAM+GU against strong UAM.

The result is classified as complementarity only if strong UAM+GU has no
higher WMDP-Cyber accuracy and no lower MMLU point estimate than strong UAM.
Single-seed differences are descriptive and must not be called statistically
significant.

## Claim boundary

This one-pair run can show whether substantially more optimization changes the
observed endpoint. Because learning rate and steps both change, it cannot
identify which knob caused the change. It also does not establish multi-seed
robustness, WMDP-Bio generalization, or exact optimizer-aware GU.
