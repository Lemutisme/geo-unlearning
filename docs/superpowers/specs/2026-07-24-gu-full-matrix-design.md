# GU Full GPU Matrix Design

**Branch:** `feat/GU`
**Execution device:** dev0 only
**Environment:** `conda activate unlearning`

## Goal

Run a checkpoint-free, method-complete evaluation of the common realized-delta
GU implementation across TOFU-8B, MUSE, and WMDP-Cyber, then produce traceable
benchmark tables with endpoint, runtime, and peak-memory metrics.

Only GU-enabled methods are trained. An untouched model is an evaluation
reference, not another unlearning arm.

## Claims Tested

1. The common GU path is objective-agnostic across every shipped unlearning
   trainer with a semantically valid benchmark recipe.
2. GU produces useful forgetting/utility trade-offs under method-native shipped
   hyperparameters without checkpoint persistence.

The matrix is exploratory for seed 0 and confirmatory for every valid,
non-degenerate method after seeds 1 and 2 complete.

## Methods

The matrix contains eleven GU-enabled methods:

1. GradAscent
2. GradDiff
3. NPO
4. DPO
5. SimNPO
6. RMU
7. UNDIAL
8. CEU
9. WGA
10. SatImp
11. PDU

Every method retains its shipped objective, data semantics, learning rate,
training duration, optimizer, precision, and performance configuration. There
is no cross-method LR/step normalization and no endpoint-based hyperparameter
selection.

RMU reuses branch-compatible recipe values and provenance on the common shipped `RMU` trainer
and the common `feat/GU` implementation. Branch-only machinery is
not silently ported across implementations: WMDP uses
`shipped_deterministic_random` sampling and does not claim sequential-sampler parity.
TOFU intentionally translates the branch scope to module 31 and layers 29--31
so the approved common GU scope receives gradients. The experiment must not mix
results from two different GU implementations.

## Fixed GU Protocol

The GU configuration is held constant across methods within each benchmark:

- `retain_history_rank: 8`;
- `projection_eps: 1.0e-6`;
- `retain_filter: first_order`;
- answer-masked retain NLL as the protected functional;
- parameters outside the selected regex are frozen;
- one JSONL diagnostic record per optimizer update.

The selected scopes use the existing benchmark protocols:

- TOFU Llama-3.1-8B: `model[.]layers[.](29|30|31)[.]mlp[.]down_proj[.]weight`;
- MUSE Llama-2-7B: `model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight`;
- WMDP Zephyr-7B: `model[.]layers[.](5|6|7)[.]mlp[.]down_proj[.]weight`.

These scopes are benchmark constants, not per-method hyperparameters. RMU uses
the same selected scope as the other methods on a given benchmark.

## Benchmarks

### TOFU

- Model: Llama-3.1-8B-Instruct.
- Splits: forget01, forget05, forget10.
- Corresponding retain and holdout splits: retain99/holdout01,
  retain95/holdout05, retain90/holdout10.

### MUSE

- Model: shipped Llama-2-7B target model.
- Splits: News and Books.

### WMDP

- Model: pinned Zephyr-7B checkpoint.
- Split: Cyber only.
- Bio is outside this protocol and must not enter commands, files, metrics, or
  claims.

## Compatibility Matrix

The following nine methods run on all six task settings:

- GradAscent
- GradDiff
- NPO
- SimNPO
- UNDIAL
- CEU
- WGA
- SatImp
- PDU

RMU runs on all six settings using its existing benchmark-specific recipes.

DPO is N/A on all six settings. Its shipped TOFU dataset adapter requires
`./data/idk.jsonl`, but that artifact is absent from the repository, all current
worktrees, and upstream HEAD. No substitute artifact or hash is invented. Every
DPO cell records the reason `missing_shipped_idk_artifact` and the missing path.

The seed-0 stage therefore contains 60 training jobs:

```text
9 general methods * 6 settings + RMU * 6 settings = 60
```

No adapter is invented merely to fill an N/A cell.

## Seed Policy

### Stage 1

Run every compatible method with seed 0.
`build_manifest` creates only this seed-0 registry and rejects every other seed.

### Stage 2

Run seeds 1 and 2 for every Stage-1 method satisfying all validity conditions:

- finite optimizer state, gradients, proposal, correction, and endpoint values;
- one GU projection per optimizer update;
- at least one selected parameter changes over the run;
- no erroneous zero-step, wrong-sign, or rejection event;
- complete endpoint and resource schemas;
- no forbidden persistence artifact.

Validity does not depend on whether the endpoint result is competitive. Every
valid method is replicated, not only the seed-0 winner. Failed seeds are never
replaced. Task 4 creates these jobs only by expanding valid seed-0 parents; it
does not call the Stage-1 manifest builder with seeds 1 or 2.

## Job Lifecycle

Each job loads a fresh model and completes training and evaluation in one
process:

```text
load pinned model and tokenizer
  -> train GU on dev0
  -> evaluate the live in-memory model
  -> write endpoints, diagnostics, and resources
  -> audit artifacts
  -> release model and GPU memory
```

The job must not save a model, checkpoint, optimizer, scheduler, RNG, callback,
or Trainer state. Evaluation cannot depend on reloading a trained model from
disk.

## Persistent Local Queue

The run root is:

```text
saves/exp/GU_FULL_MATRIX_20260724/
```

It contains:

```text
manifest.json
queue_state.json
queue.log
jobs/<job_id>/
GU_FULL_RAW.json
GU_FULL_TABLES.md
GU_FULL_REPORT.md
```

The queue is local and binds every training subprocess to
`CUDA_VISIBLE_DEVICES=0`. It runs one job at a time and checks that dev0 uses
less than 500 MiB before launching another job.

Job states are:

```text
pending -> running -> completed
                   -> invalid_scientific
                   -> failed_infrastructure
```

Every transition is written atomically to `queue_state.json`. A restarted queue
reconciles a `running` job against its process and evidence files before
continuing.

Scientific failures include nonfinite state, OOM, GU rejection, wrong-sign
progress, invalid zero-step behavior, incomplete endpoints, and contract
violations. They are preserved and are not retried.

An infrastructure failure may be retried once with the identical command. No
retry changes a seed, hyperparameter, data order, model revision, or method.

## Preflight and Smoke Gates

Before creating full job outputs:

1. resolve every Hydra command;
2. verify method and dataset contracts;
3. verify local corpus and retain-log hashes;
4. verify canonical content manifests for every required Arrow/metadata cache;
5. verify every Hub snapshot revision, symlink containment, and
   content-addressed blob hash;
6. memoize fingerprints by canonical source requirement so each unique source
   is hashed once per dry-run;
7. fail without downloading when any pinned cache is absent;
8. verify no-save flags and the artifact allowlist;
9. verify dev0 is free.

At the time of this spec amendment, the pinned MUSE Books target snapshot is
absent locally. A real dry-run therefore stops with the exact source name
`muse-bench/MUSE-Books_target`; it must not substitute or download a checkpoint.

Run one representative smoke for every compatible method. RMU receives
additional benchmark-specific smokes because its data and model paths differ.
A failed smoke blocks only the corresponding method/benchmark compatibility
class and is reported before Stage 1.

## Failure Semantics

- A job failure never silently launches a modified replacement.
- OOM is a scientific/configuration failure for the shipped recipe, not an
  invitation to shrink the method.
- Cache corruption or host I/O failure is infrastructure evidence and is kept
  separate from method validity.
- A failed benchmark reference gate blocks that benchmark's jobs.
- The queue continues past an individual invalid method so the complete matrix
  remains observable.

## Endpoint Tables

### TOFU

Produce one table for each forget split:

| Method | ES Re. ↑ | ES Un. ↓ | Priv. ↑ | MU ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|

### MUSE

Produce one table for News and one for Books:

| Method | VerbMem ↓ | KnowMem ↓ | Extraction ↓ | Privacy/MIA | Retain utility ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|

### WMDP

Produce the Cyber-only table:

| Method | WMDP-Cyber ↓ | MMLU ↑ | wall-clock | peak mem |
|---|---:|---:|---:|---:|

Three-seed results are reported as mean ± sample standard deviation. Invalid or
missing seeds are listed explicitly and never replaced. DPO N/A cells retain
their compatibility reason.

## Raw-to-Table Integrity

Every displayed scalar must resolve to:

- an endpoint JSON field;
- a method/split/seed manifest identity;
- a finite/resource/diagnostic sidecar;
- the exact model and data provenance;
- the originating job directory.

The analyzer rejects duplicate seeds, mismatched methods, wrong benchmark
schemas, incomplete valid sets, nonfinite values, forbidden WMDP-Bio fields,
and summary/raw discrepancies.

## Resource Metrics

Each job records:

- training seconds;
- evaluation seconds;
- end-to-end wall-clock seconds;
- peak allocated CUDA memory;
- peak reserved CUDA memory;
- optimizer update count;
- GU projection count;
- zero-step count;
- applied-scale distribution;
- correction ratio and maximum final retain violation.

The table's `wall-clock` is end-to-end job time. `peak mem` is peak allocated
CUDA memory unless the table explicitly labels another source.

## Estimated Cost

- Stage 1: approximately 18–30 dev0 GPU-hours.
- Stage 2: up to approximately 36–60 additional dev0 GPU-hours.
- Expected total: 2–4 days of uninterrupted single-GPU execution.

These are scheduling estimates, not result claims.

## Acceptance Criteria

- All 60 Stage-1 compatible jobs reach a terminal recorded state.
- Every valid Stage-1 job receives exactly seeds 1 and 2.
- No two jobs overlap on dev0.
- No forbidden model/checkpoint/state artifact exists.
- Every completed job has exact endpoint, diagnostics, resource, and provenance
  evidence.
- The analyzer can regenerate every table from raw files deterministically.
- Report prose distinguishes complete results, invalid scientific runs,
  infrastructure failures, and N/A combinations.
