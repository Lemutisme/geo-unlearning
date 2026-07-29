# Dual-GPU Baseline Reference Design

## Goal

Produce a fast, checkpoint-free seed-0 reference for every compatible baseline
method on the same six settings used by the GU matrix, then render direct
baseline-versus-GU improvement tables.

The reference contains 60 jobs:

```text
10 methods × (TOFU forget01/05/10 + MUSE News/Books + WMDP Cyber)
```

DPO remains N/A because the shipped `data/idk.jsonl` artifact is absent. No
replacement data is synthesized.

## Why `eval_all.sh` Must Be Adapted

The current script is not executed unchanged. It hard-codes GPUs 6/7, launches
two-process DDP, saves trained models, reloads checkpoints for evaluation, and
mixes older model/benchmark paths. Those choices violate the current protocol.

`scripts/eval_all.sh` remains the user-facing entrypoint, but becomes a thin
launcher for the matched reference workflow described below.

## Baseline Definition

The quick reference is the method-native baseline:

- the same method, model revision, tokenizer revision, dataset revision, split,
  retain log, optimizer, learning rate, duration, effective batch, and evaluator
  as the current GU seed-0 job;
- no `trainer.method_args.gu` mapping;
- no GU parameter freezing or projection;
- method-native trainable parameters;
- seed and data seed fixed to 0.

This is an end-to-end baseline. Its difference from GU includes both the GU
projection and GU's selected-parameter intervention scope. It is not claimed as
a projector-only ablation; a scope-matched no-projection arm can be added later.

## Execution Architecture

`eval_all.sh` launches two independent single-process workers:

```text
GPU0: jobs with sorted index 0,2,4,...
GPU1: jobs with sorted index 1,3,5,...
```

This is intentionally not DDP. Each job preserves single-process optimizer and
batch semantics, while two jobs run concurrently for approximately 2× matrix
throughput.

A single companion, `scripts/run_baseline_reference.py`, imports the existing
GU matrix registry to reuse job identities, pinned provenance, benchmark
arguments, RMU translations, cache roots, and no-save/live-evaluation settings.
It removes only the GU-specific command arguments and binds each worker to its
assigned physical GPU.

Each child performs:

```text
load fresh pinned model
  -> train baseline method
  -> evaluate the live in-memory model
  -> write endpoint/resource evidence
  -> release model and GPU memory
```

No model, checkpoint, optimizer, scheduler, RNG, callback, or Trainer state may
be persisted. The separate `src/eval.py` checkpoint-reload path is removed.

## State and Output

The run root is local disk to avoid workspace FUSE failures:

```text
/root/baseline_reference_20260729/
```

The workspace exposes a symlink under `saves/exp/`. The root contains:

```text
manifest.json
worker0_state.json
worker1_state.json
worker0.log
worker1.log
jobs/<job_id>/
BASELINE_GU_RAW.json
BASELINE_GU_TABLES.md
BASELINE_GU_REPORT.md
```

Each worker exclusively owns its state file, eliminating concurrent JSON writes.
State transitions and result sidecars use atomic replacement. A restart skips
validated completed jobs and reconciles a stale running PID/result before
continuing.

## Job Validity

A completed baseline job requires:

- exact registered identity and command sidecar;
- exit code zero;
- exactly one benchmark summary/raw endpoint pair;
- finite endpoint values;
- at least one optimizer update in the Trainer log;
- positive direct-child NVML memory evidence;
- no checkpoint/model/optimizer/Trainer-state artifact.

Scientific failures are not retried. A host-I/O, cache, launch, or monitor-only
failure may be retried once with the identical command. GPU admission uses a
per-device nonblocking lock and requires the assigned GPU below 500 MiB.

## Comparison

The comparison consumes only validated seed-0 pairs with identical
method/benchmark identities:

- TOFU: ES Re., ES Un., Priv., MU;
- MUSE: VerbMem, KnowMem, Extraction, Privacy/MIA, retain utility;
- WMDP: WMDP-Cyber and MMLU;
- resource: wall-clock seconds and peak NVML MiB.

Tables show baseline, GU, and a direction-normalized improvement:

```text
higher-is-better: improvement = GU - baseline
lower-is-better:  improvement = baseline - GU
```

Positive improvement always favors GU. Raw values are always shown beside the
improvement. The report is explicitly labeled `seed0 reference`; it is not a
mean±standard-deviation claim.

## Testing and Launch Gates

Before launch, tests must prove:

1. exactly 60 baseline jobs and six DPO N/A records;
2. every Hydra command composes and contains no GU arguments;
3. all commands use live evaluation and no-save flags;
4. worker partitions are disjoint, complete, and pinned to GPU0/GPU1;
5. artifact and endpoint validation rejects tampering;
6. restart/retry semantics preserve exact command identity;
7. comparison signs and table headers are exact;
8. `eval_all.sh` launches both workers and propagates failure.

After CPU tests, one real one-step-per-family smoke runs on both GPUs. Only then
does `eval_all.sh` launch the 60-job seed-0 reference.

## Non-Goals

- No three-seed baseline in this quick pass.
- No projector-only scope-matched ablation.
- No DDP or distributed optimizer claim.
- No checkpoint persistence.
- No reuse of historical baseline numbers with different provenance.
