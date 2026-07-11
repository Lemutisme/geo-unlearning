#!/usr/bin/env bash

set -euo pipefail

audit_checkpoint_payloads() {
    local audit_root=$1
    local checkpoint_file
    checkpoint_file=$(find "${audit_root}" -type f \( \
        -name '*.safetensors' -o \
        -name 'pytorch_model*.bin' -o \
        -name 'training_args.bin' -o \
        -name 'trainer_state.json' -o \
        -name 'optimizer.pt' -o \
        -name 'scheduler.pt' -o \
        -name 'rng_state.pth' -o \
        -name '*.ckpt' \
    \) -print -quit)
    if [[ -n "${checkpoint_file}" ]]; then
        echo "Unexpected checkpoint payload: ${checkpoint_file}" >&2
        return 1
    fi
}

if [[ $# -eq 2 && $1 == --audit-only ]]; then
    audit_checkpoint_payloads "$2"
    exit $?
fi

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <dataset> <method> <gpu> <timestamp> <system_mode>" >&2
    exit 2
fi

dataset=$1
method=$2
gpu=$3
timestamp=$4
system_mode=$5

conda_exe=${CONDA_EXE:-}
if [[ -z "${conda_exe}" ]]; then
    conda_exe=$(command -v conda || true)
fi
if [[ -z "${conda_exe}" && -x /root/miniconda3/bin/conda ]]; then
    conda_exe=/root/miniconda3/bin/conda
fi
if [[ -z "${conda_exe}" ]]; then
    echo "Unable to locate the conda executable." >&2
    exit 127
fi
conda_base=$("${conda_exe}" info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate unlearning

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME=${PCGRAD_HF_HOME:-/root/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

case "${method}" in
    control)
        gu_enabled=false
        gradient_surgery=gu
        ;;
    gu|pcgrad)
        gu_enabled=true
        gradient_surgery=${method}
        ;;
    *)
        echo "Unsupported method: ${method}" >&2
        exit 2
        ;;
esac

case "${system_mode}" in
    production)
        optimizer=paged_adamw_32bit
        torch_dtype=bfloat16
        attention_implementation=flash_attention_2
        bf16=true
        bf16_full_eval=true
        ;;
    torch_flash)
        optimizer=adamw_torch
        torch_dtype=bfloat16
        attention_implementation=flash_attention_2
        bf16=true
        bf16_full_eval=true
        ;;
    torch_eager_fp32)
        optimizer=adamw_torch
        torch_dtype=float32
        attention_implementation=eager
        bf16=false
        bf16_full_eval=false
        ;;
    *)
        echo "Unsupported system mode: ${system_mode}" >&2
        exit 2
        ;;
esac

if [[ "${system_mode}" != production ]] && {
    [[ "${dataset}" != tofu01 ]] || [[ "${method}" != pcgrad ]];
}; then
    echo "System-isolation modes are restricted to tofu01 PCGrad." >&2
    exit 2
fi

model_name=
base_model=
experiment=
component_buffer_device=parameter
per_device_batch_size=4
gradient_accumulation_steps=8
retain_logs_path=
dataset_overrides=()
summary_name=

case "${dataset}" in
    tofu01|tofu05|tofu10)
        suffix=${dataset#tofu}
        forget_split="forget${suffix}"
        retain_split="retain$((100 - 10#${suffix}))"
        holdout_split="holdout${suffix}"
        model_name=Llama-3.2-1B-Instruct
        base_model=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
        experiment=unlearn/tofu/default
        retain_logs_path="saves/eval/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"
        summary_name=TOFU_SUMMARY.json
        dataset_overrides=(
            "forget_split=${forget_split}"
            "retain_split=${retain_split}"
            "holdout_split=${holdout_split}"
        )
        ;;
    muse_news|muse_books)
        if [[ "${dataset}" == muse_news ]]; then
            data_split=News
        else
            data_split=Books
        fi
        model_name=Llama-2-7b-hf
        base_model="muse-bench/MUSE-${data_split}_target"
        experiment=unlearn/muse/default
        component_buffer_device=cpu
        per_device_batch_size=1
        gradient_accumulation_steps=8
        retain_logs_path="saves/eval/muse_${model_name}_${data_split}_retrain/MUSE_EVAL.json"
        summary_name=MUSE_SUMMARY.json
        dataset_overrides=("data_split=${data_split}")

        selected_numel=6738415616
        required_host_bytes=$((selected_numel * 8 * 12 / 10))
        available_host_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
        available_host_bytes=$((available_host_kib * 1024))
        if (( available_host_bytes < required_host_bytes )); then
            echo "Insufficient host memory for GU component buffers: requires ${required_host_bytes}, found ${available_host_bytes}." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported dataset: ${dataset}" >&2
        exit 2
        ;;
esac

matrix_root="saves/exp/PCGRAD_SMOKE/${timestamp}"
arm_name="${dataset}_${method}_${system_mode}"
arm_dir="${matrix_root}/${arm_name}"
local_root=${PCGRAD_LOCAL_ROOT:-/tmp/pcgrad_smoke}
local_arm_dir="${local_root}/${timestamp}/${arm_name}"
task_name="pcgrad_smoke_${arm_name}_${timestamp}"
mkdir -p "${arm_dir}" "${local_arm_dir}"

diagnostics_path=null
actual_delta_mode=off
if [[ "${gu_enabled}" == true ]]; then
    diagnostics_path="${local_arm_dir}/gu_diagnostics.jsonl"
fi
if [[ "${dataset}" == muse_news || "${dataset}" == muse_books ]]; then
    if [[ "${gu_enabled}" == true ]]; then
        actual_delta_mode=sampled
    fi
elif [[ "${dataset}" == tofu01 && "${method}" == pcgrad ]]; then
    actual_delta_mode=full
fi

command=(
    accelerate launch
    --config_file configs/accelerate/gu_single_gpu.yaml
    --gpu_ids "${gpu}"
    src/train.py
    --config-name=unlearn.yaml
    "experiment=${experiment}"
    trainer=GeometricUnlearn
    "task_name=${task_name}"
    "model=${model_name}"
    "model.model_args.pretrained_model_name_or_path=${base_model}"
    "model.model_args.attn_implementation=${attention_implementation}"
    "model.model_args.torch_dtype=${torch_dtype}"
    "retain_logs_path=${retain_logs_path}"
    "paths.output_dir=${local_arm_dir}"
    save_model_after_train=false
    "trainer.args.per_device_train_batch_size=${per_device_batch_size}"
    "trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}"
    trainer.args.max_steps=10
    trainer.args.learning_rate=1e-5
    "trainer.args.optim=${optimizer}"
    trainer.args.adam_beta1=0.0
    trainer.args.weight_decay=0.0
    trainer.args.fp16=false
    "trainer.args.bf16=${bf16}"
    "trainer.args.bf16_full_eval=${bf16_full_eval}"
    trainer.args.gradient_checkpointing=true
    trainer.args.gradient_checkpointing_kwargs.use_reentrant=false
    trainer.args.logging_steps=1
    trainer.args.report_to=none
    trainer.args.save_strategy=no
    trainer.args.save_only_model=false
    trainer.args.do_train=true
    trainer.args.do_eval=true
    trainer.args.eval_on_start=false
    trainer.args.eval_strategy=no
    trainer.args.seed=0
    trainer.method_args.geometric_config.loss=simnpo
    "trainer.method_args.geometric_config.gu_enabled=${gu_enabled}"
    "trainer.method_args.geometric_config.gradient_surgery=${gradient_surgery}"
    "trainer.method_args.geometric_config.component_buffer_device=${component_buffer_device}"
    "trainer.method_args.geometric_config.diagnostics_path=${diagnostics_path}"
    "trainer.method_args.geometric_config.actual_delta_mode=${actual_delta_mode}"
    'trainer.method_args.geometric_config.actual_delta_steps=[1,10]'
    trainer.method_args.geometric_config.actual_delta_sample_elements=1000000
    "${dataset_overrides[@]}"
)

printf 'Launching %q ' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee "${local_arm_dir}/run.log"

audit_checkpoint_payloads "${local_arm_dir}"

summary_path=$(find "${local_arm_dir}" -type f -name "${summary_name}" -print -quit)
if [[ -z "${summary_path}" ]]; then
    echo "Missing ${summary_name} under ${local_arm_dir}." >&2
    exit 1
fi

persist_artifacts() {
    local persistent_summary="${arm_dir}/checkpoint-10/evals/${summary_name}"
    mkdir -p "${arm_dir}/checkpoint-10/evals" "${arm_dir}/.hydra"
    cp "${summary_path}" "${persistent_summary}"
    cp "${local_arm_dir}/.hydra/config.yaml" "${arm_dir}/.hydra/config.yaml"
    for artifact in \
        run.log \
        GeometricUnlearn.log \
        gu_diagnostics.jsonl \
        gu_diagnostics.summary.json; do
        if [[ -f "${local_arm_dir}/${artifact}" ]]; then
            cp "${local_arm_dir}/${artifact}" "${arm_dir}/${artifact}"
        fi
    done
    audit_checkpoint_payloads "${arm_dir}"
    echo "${persistent_summary}"
}

persistent_summary=$(persist_artifacts)
echo "Completed ${arm_name}: ${persistent_summary}"
