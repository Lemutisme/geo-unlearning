# GPU 1 GU Full-Matrix Rerun Design

**Status:** Approved for implementation planning  
**Date:** 2026-08-04  
**Branch:** `feat/GU`

## Goal

Rerun the complete current common-GU experiment matrix on physical GPU 1 with
fresh outputs and reproducible source evidence. The run includes all 60
registered seed-0 jobs and seeds 1 and 2 for every valid seed-0 parent, for up
to 180 training-and-evaluation jobs. The six registered DPO cells remain N/A
because the matrix contract excludes synthesized replacement data.

## Current Constraints

- GPU 1 is currently idle and has sufficient memory.
- `scripts/run_gu_full_matrix.py` currently binds child processes and admission
  checks to physical GPU 0.
- The pinned `unlearning` environment is absent on this host.
- The runner's offline model, dataset, evaluation-log, and WMDP corpus caches
  are absent. The registered Hugging Face revisions are reachable over the
  network.
- Existing output must not be reused because GU diagnostics intentionally
  require a new file and the user requested a rerun.

## GPU Selection

Add one environment-controlled physical-device selector to the matrix runner:

```text
GU_MATRIX_GPU=<nonnegative integer>, default 0
```

The selected index controls all of the following as one invariant:

- child `CUDA_VISIBLE_DEVICES`;
- NVML admission sampling and error messages;
- the per-device controller lock path;
- command/evidence environment records.

Invalid values fail before manifest or job execution. The default remains GPU
0 so existing commands and tests retain their behavior. The rerun exports
`GU_MATRIX_GPU=1` for manifest validation, smoke, queue, status, seed expansion,
and analysis-related audit commands.

## Runtime and Source Preparation

Create a Python 3.11 `unlearning` conda environment using the repository's
pinned requirements and development test dependencies. Verify the exact Torch,
Transformers, Accelerate, bitsandbytes, datasets, Hydra, and pytest imports
before any experiment launch.

Populate the runner's registered runtime roots with the exact revisions and
content layouts expected by `source_requirements`:

- TOFU target model/tokenizer, evaluation datasets, classifier, and retain
  logs;
- MUSE News/Books target and retrain models, tokenizer, datasets, classifier,
  and retain logs;
- Zephyr/WMDP model, WMDP and MMLU evaluation caches, WMDP corpora, and
  WikiText retain cache used by RMU.

Downloads use pinned revisions. The existing preflight fingerprints remain the
source of truth; cache preparation is incomplete until every registered source
passes `preflight` without weakening or replacing a hash.

## Output and Execution

Use a fresh root:

```text
saves/exp/GU_FULL_MATRIX_20260804_GPU1/
```

Execution order:

1. Run focused GPU-selector tests and the full CPU test suite in the pinned
   environment.
2. Create and validate the seed-0 manifest under the fresh root.
3. Run the representative smoke matrix on GPU 1.
4. Launch the persistent sequential seed-0 queue in a dedicated tmux session.
5. Monitor until all 60 seed-0 jobs reach terminal states, retrying only the
   infrastructure failures permitted by the existing queue contract.
6. Expand seeds 1 and 2 for every completed valid seed-0 parent and drain the
   queue again.
7. Run the complete analyzer and regenerate the final raw JSON, tables, and
   report from validated job evidence.

Only one matrix child may use GPU 1 at a time. Jobs remain checkpoint-free and
are evaluated from their live in-memory models exactly as registered.

## Failure Handling

- A busy GPU 1 blocks launch rather than falling back to another device.
- Source, hash, Hydra composition, or scientific-validation failures remain
  terminal and are never hidden by changing a revision, seed, or recipe.
- Infrastructure failures receive only the retry already permitted by the
  queue state machine.
- The queue state and logs are durable; after host or session interruption, the
  same command resumes from validated state rather than deleting evidence.
- If a required artifact is inaccessible or a pinned fingerprint cannot be
  reproduced, preparation stops and reports the exact artifact instead of
  launching a partial matrix.

## Testing

Add focused tests proving:

1. the default selected GPU is 0;
2. `GU_MATRIX_GPU=1` reaches child environment evidence;
3. admission checks physical GPU 1 and reports GPU 1 failures;
4. invalid selectors fail closed;
5. per-GPU controller lock paths do not collide;
6. existing default-GPU command and manifest contracts remain unchanged.

Before claiming launch success, verify the queue state records one live child,
that the child environment records `CUDA_VISIBLE_DEVICES=1`, and that NVML maps
the child PID to physical GPU 1.

## Acceptance Criteria

- No experiment child runs on physical GPU 0 or any device other than GPU 1.
- Every registered seed-0 job reaches a durable terminal state.
- Every valid completed seed-0 job receives exactly seeds 1 and 2.
- All completed jobs pass existing diagnostic, endpoint, resource, provenance,
  and artifact validation.
- Final tables are generated only with the analyzer's complete-run gate.
- The original code checkout remains the source of all experiment commands;
  outputs are isolated under the fresh rerun root.
