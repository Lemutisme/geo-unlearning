# Approximate Adam/AdamW GU MVP Report

Date: 2026-07-11
Branch: `dev0`
Validation objective: SimNPO + NLL retain loss

## Outcome

The obsolete `GeometricUnlearn.optimizer_step()` path has been replaced by a
reachable Transformers 4.45.1 `training_step()` implementation. It obtains
forget and retain gradients independently, accumulates them across the full
effective batch, performs one global rank-one projection in a frozen Adam
square-root coordinate system, writes the combined raw gradients to
`parameter.grad`, and then leaves clipping and `optimizer.step()` to the
standard Trainer loop.

This implementation is deliberately labelled `approximate_adam_stage_a`.
It certifies orthogonality only for the frozen pre-step diagonal metric. A
standard Adam/AdamW step updates its denominator from the projected gradient,
so the realized parameter delta is not claimed to be strictly GU-safe. A
custom optimizer remains a separate follow-up.

## Implemented behavior

- Removed the unreachable Trainer `optimizer_step()` hook and obsolete
  per-parameter projector.
- Resolved method coefficients once during initialization. SimNPO uses
  `gamma=0.125`, `alpha=1.0`, and `retain_loss_type=NLL`.
- Added `compute_component_losses()` and independent `autograd.grad()` calls
  for the forget and retain objectives.
- Added detached FP32 component buffers and projects only at an optimizer
  update boundary, including Transformers' short final accumulation window.
- Uses one coefficient over all selected tensors rather than blockwise
  projection coefficients.
- Uses the frozen pre-step Adam/AdamW second moment through square-root
  coordinate maps. The first optimizer step uses an explicitly reported
  identity fallback because no second-moment state exists yet.
- Preserves ordinary total-loss gradients for unselected parameters and the
  standard Trainer clipping/step order.
- `gu_enabled=false` delegates to the ordinary training path; a unit test
  verifies one-step equivalence with the native SimNPO trainer.
- Fails closed for DeepSpeed, FSDP, multi-process/multi-GPU, FP16, Apex,
  reentrant checkpointing, non-Torch Adam/AdamW, nonzero `beta1`, nonzero
  weight decay, AMSGrad, and empty parameter selection.
- Added `save_model_after_train=false`, which evaluates the in-memory trained
  model without writing model, optimizer, scheduler, RNG, or Trainer-state
  checkpoints. The default remains `true` for existing workflows.

## Validation configuration

The paired run was launched with:

```bash
conda activate unlearning
bash scripts/gu_adam_mvp_eval.sh
```

Both arms loaded `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` and used:

| Setting | Value |
|---|---:|
| Device/processes | CUDA device 0 / 1 process |
| Precision | FP32 |
| Optimizer | `adamw_torch` |
| Learning rate | `1e-5` |
| Adam beta1 | `0.0` |
| Weight decay | `0.0` |
| Per-device batch | `4` |
| Gradient accumulation | `8` |
| Effective batch | `32` |
| Optimizer updates | `10` |
| Gradient checkpointing | non-reentrant |
| Selected parameters | all trainable parameters |
| Checkpoint saving | disabled |

The control and GU resolved Hydra configurations differ only in
`geometric_config.gu_enabled`, task name, and output directory.

## Projection diagnostics

The GU log records exactly 10 projection calls for 10 optimizer updates.

| Diagnostic | Result |
|---|---:|
| Projection mode | `approximate_adam_stage_a` |
| Projection calls | 10 |
| Maximum transformed-space relative residual | `3.44456161e-08` |
| Step 1 identity-fallback parameters | 146 |
| Steps 2-10 identity-fallback parameters | 0 |
| Projection coefficient range | `[-0.152050897, -0.00170621683]` |

The fallback pattern is expected: Adam has no `exp_avg_sq` before its first
step, while all selected parameter states are initialized afterward.

## Paired TOFU results

These are the raw single-run summaries. `Delta` is GU minus the matched
`gu_enabled=false` control.

| Metric | Control | GU | Delta |
|---|---:|---:|---:|
| `exact_memorization` | 0.863954286 | 0.863954286 | +0.000000000 |
| `extraction_strength` | 0.174492129 | 0.174492129 | +0.000000000 |
| `forget_Q_A_Prob` | 0.549253395 | 0.549581745 | +0.000328350 |
| `forget_Q_A_ROUGE` | 0.460355471 | 0.460355471 | +0.000000000 |
| `forget_Q_A_gibberish` | 0.912147996 | 0.912147996 | +0.000000000 |
| `mia_gradnorm` | 0.791250000 | 0.791875000 | +0.000625000 |
| `mia_loss` | 0.960625000 | 0.960625000 | +0.000000000 |
| `mia_min_k` | 0.968125000 | 0.968125000 | +0.000000000 |
| `mia_min_k_plus_plus` | 0.804375000 | 0.805000000 | +0.000625000 |
| `mia_zlib` | 0.979375000 | 0.979375000 | +0.000000000 |
| `model_utility` | 0.598101079 | 0.598102302 | +0.000001223 |
| `privleak` | -93.978748506 | -93.978748506 | +0.000000000 |
| `retain_extraction_strength` | 0.641050608 | 0.641050608 | +0.000000000 |

The 10-step smoke run demonstrates that the projected path executes and is
numerically stable. The near-identical aggregate metrics are not evidence of
performance superiority; no multi-seed or converged comparison was run.

Raw summaries:

- Control:
  `saves/exp/GU_ADAM_MVP/0711015931/tofu_Llama-3.2-1B-Instruct_forget01_AdamMVP_control_0711015931/checkpoint-10/evals/TOFU_SUMMARY.json`
- GU:
  `saves/exp/GU_ADAM_MVP/0711015931/tofu_Llama-3.2-1B-Instruct_forget01_AdamMVP_gu_0711015931/checkpoint-10/evals/TOFU_SUMMARY.json`

The `checkpoint-10` directory is an evaluator namespace only. It contains
evaluation JSON and is not a saved training checkpoint.

## Historical summaries

The user-provided historical summaries are retained as context, not as a
matched comparison:

| Metric | Historical SimNPO | Historical ineffective GU | New control | New GU |
|---|---:|---:|---:|---:|
| `model_utility` | 0.588452101 | 0.594625657 | 0.598101079 | 0.598102302 |
| `forget_Q_A_Prob` | 0.680213287 | 0.738535015 | 0.549253395 | 0.549581745 |
| `forget_Q_A_ROUGE` | 0.570465184 | 0.605802164 | 0.460355471 | 0.460355471 |
| `extraction_strength` | 0.266061673 | 0.302658830 | 0.174492129 | 0.174492129 |
| `exact_memorization` | 0.908532383 | 0.921323550 | 0.863954286 | 0.863954286 |
| `retain_extraction_strength` | 0.629852946 | 0.642575898 | 0.641050608 | 0.641050608 |

The historical runs used 10 epochs, batch 4 with GAS 4, BF16,
`paged_adamw_32bit`, and weight decay 0.01. The historical ineffective GU also
used SimNPO `gamma=0.2`; the historical baseline used `gamma=0.125`. The new
validation instead uses 10 optimizer steps, batch 4 with GAS 8, FP32, Torch
AdamW, `beta1=0`, and zero weight decay. The historical metric differences
therefore cannot be attributed to the repaired projection.

## Verification

The focused suite covers global geometry, the blockwise counterexample,
frozen-metric safety, zero retain gradients, component-loss separation, FP32
buffers, projection/update count, short GAS windows, GAS equivalence with a
nonuniform initialized Adam metric, malformed-state rejection, supported and
unsupported runtime contracts, native SimNPO baseline recovery, YAML
contracts, and the checkpoint-saving gate. The completed real-run artifacts
are audited separately for checkpoint payload files.

Final local verification uses:

```bash
conda activate unlearning
pytest tests/test_geometric_adam.py -q
ruff check src/trainer/unlearn/geometric.py src/train.py tests
ruff format --check src/trainer/unlearn/geometric.py src/train.py tests
bash -n scripts/gu_adam_mvp_eval.sh
git diff --check
```

Result: 35 tests passed; Ruff check and format check, shell syntax, and
`git diff --check` all passed. The five pytest warnings are external
DeepSpeed/Pydantic deprecation warnings.

The completed validation output contains no `*.safetensors`,
`pytorch_model*.bin`, `training_args.bin`, `trainer_state.json`, `optimizer.pt`,
`scheduler.pt`, `rng_state.pth`, or `*.ckpt` files.
