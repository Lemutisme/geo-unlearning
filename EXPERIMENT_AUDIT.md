# Experiment Audit Report

**Date:** 2026-07-12  
**Auditor:** Independent Codex reviewer (read-only)  
**Project:** UAM / Residual-GU-UAM TOFU smoke  
**Run:** `0712145130`

## Overall Verdict: WARN

## Integrity Status: warn

No fabricated ground truth, phantom result, hidden checkpoint payload, or performance-superiority claim was found. Warnings concern external provenance that is not revision-pinned, aggregate/self-relative diagnostics presented as raw stored values, and limited single-seed scope.

## Checks

### A. Ground Truth Provenance: WARN

- Primary TOFU QA targets and ROUGE references come from dataset labels through the repository's TOFU evaluator.
- MIA uses dataset-defined forget/holdout membership.
- `privleak` uses an external retain-model evaluation log, and gibberish probability uses an external classifier. Both are legitimate proxies, but their exact revisions/checksums are not sealed in the final manifest or analysis artifact.

### B. Score Normalization: WARN

- No normalization by a method's own maximum, minimum, or mean was found.
- Option probability, model utility, and privacy metrics use their documented benchmark aggregations.
- The effective perturbation ratio is self-relative (`effective_norm / requested_norm`). Its numerator and denominator exist in JSONL, but the report table shows only the ratio; therefore “raw” in the report means raw stored aggregate, not an unnormalized primitive.

### C. Result File Existence: WARN

- All four manifest arms exist and completed with exit code zero.
- All 52 metric rows, four diagnostic rows, and eight actual-delta rows match `analysis.json` and the underlying arm artifacts at reported precision.
- The report records code commit `3fd6347`, but the manifest and `analysis.json` do not cryptographically bind the run to that commit.
- A recomputed floating-point mean differs from the stored value only at the final binary rounding digit and renders identically.

### D. Dead Code / Execution Path: WARN

- The run uses `TOFUEvaluator`; unrelated LM-eval code is not part of this experiment.
- Every reported TOFU metric was emitted by the resolved evaluator.
- The analyzer consumes geometry JSONL directly; its summary JSON is not the source for kept-residual orthogonality.

### E. Scope Assessment: PASS

- Exact scope: one seed, ten optimizer updates, one model, one TOFU forget01/retain99/holdout01 setting, and four method arms.
- The report explicitly labels the run as a mechanism smoke check and disclaims superiority, robustness, and statistical significance.

### F. Evaluation Type: PASS

- Dataset QA, ROUGE, memorization, extraction, and utility components: `real_gt`.
- MIA and geometry/actual-delta instrumentation: `self_supervised_proxy`.
- Privacy leakage and gibberish classifier outputs: `synthetic_proxy`.
- No human evaluation or simulation-only evaluation is used.

## Additional Integrity Checks

- Checkpoint payload audit: PASS. No model, optimizer, scheduler, RNG, symlink, or checkpoint directory exists in the successful artifact trees.
- Method pairing: WARN. NLL GU is directly paired against NLL UAM. SimNPO GU should be compared against SimNPO UAM; the current analyzer's universal baseline is `uam_nll`, so its default SimNPO delta is not signal-paired.

## Action Items

- Pin or checksum the base model, gibberish classifier, and retain-evaluation baseline in future manifests.
- Add signal-paired deltas, especially `uam_gu_simnpo - uam_simnpo`.
- Include requested/effective perturbation norms beside the ratio in future reports.
- Run multiple seeds and longer schedules before making empirical retain-safety or performance claims.

## Claim Impact

- Mechanism execution: **partially supported** — counters, replay/finalizer occurrence, gating, and orthogonality are evidenced, but mostly through same-implementation telemetry.
- Optimizer update occurrence: **supported** at audited steps 1 and 10 for all four arms.
- Retain-safety orthogonality: **needs qualification** — the correction is geometrically orthogonal when kept; this does not prove empirical retain preservation.
- Performance superiority: **unsupported and explicitly disclaimed**.
