#!/usr/bin/env bash

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unlearning

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}

timestamp=$(date +%m%d%H%M%S)
experiment_dir="saves/exp/GU_ADAM_MVP/${timestamp}"
base_model="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
model_name="Llama-3.2-1B-Instruct"
forget_split="forget01"
retain_split="retain99"
holdout_split="holdout01"
retain_logs_path="saves/eval/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"

mkdir -p "${experiment_dir}"

run_arm() {
    local arm=$1
    local gu_enabled=$2
    local task_name="tofu_${model_name}_${forget_split}_AdamMVP_${arm}_${timestamp}"
    local gu_override="+trainer.method_args.gu={\
enabled:${gu_enabled},\
parameter_regex:[\"lm_head[.]weight\"],\
retain_history_rank:8,\
projection_eps:1e-6,\
retain_filter:first_order,\
retain_budget:1e-4,\
backtracking_scales:[1.0,0.5,0.25,0.125],\
diagnostics_path:gu_diagnostics.jsonl}"

    accelerate launch \
        --config_file configs/accelerate/gu_single_gpu.yaml \
        src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=SimNPO \
        task_name="${task_name}" \
        model="${model_name}" \
        model.model_args.pretrained_model_name_or_path="${base_model}" \
        model.model_args.attn_implementation=eager \
        model.model_args.torch_dtype=float32 \
        forget_split="${forget_split}" \
        retain_split="${retain_split}" \
        holdout_split="${holdout_split}" \
        retain_logs_path="${retain_logs_path}" \
        paths.output_dir="${experiment_dir}/${task_name}" \
        save_model_after_train=false \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.max_steps=10 \
        trainer.args.learning_rate=1e-5 \
        trainer.args.optim=adamw_torch \
        trainer.args.adam_beta1=0.0 \
        trainer.args.weight_decay=0.0 \
        trainer.args.fp16=false \
        trainer.args.bf16=false \
        trainer.args.bf16_full_eval=false \
        trainer.args.gradient_checkpointing=true \
        trainer.args.gradient_checkpointing_kwargs.use_reentrant=false \
        trainer.args.logging_steps=1 \
        trainer.args.report_to=none \
        trainer.args.save_strategy=no \
        trainer.args.do_eval=true \
        trainer.args.eval_on_start=false \
        trainer.args.eval_strategy=no \
        "${gu_override}"

    local checkpoint_file
    checkpoint_file=$(find "${experiment_dir}/${task_name}" -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \
        -o -name 'training_args.bin' -o -name 'trainer_state.json' \) \
        -print -quit)
    if [[ -n "${checkpoint_file}" ]]; then
        echo "Unexpected checkpoint file: ${checkpoint_file}" >&2
        return 1
    fi

    echo "${arm} summary: ${experiment_dir}/${task_name}/checkpoint-10/evals/TOFU_SUMMARY.json"
}

run_arm control false
run_arm gu true

echo "Approximate Adam GU validation outputs: ${experiment_dir}"
