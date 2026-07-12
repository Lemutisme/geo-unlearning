#!/usr/bin/env bash

set -euo pipefail

audit_artifacts() {
    local root=$1
    local link payload
    link=$(find "${root}" -type l -print -quit 2>/dev/null || true)
    if [[ -n "${link}" ]]; then
        echo "Unexpected symbolic link: ${link}" >&2
        return 1
    fi
    payload=$(find "${root}" -type f \( \
        -name '*.safetensors' -o \
        -name '*.bin' -o \
        -name '*.pt' -o \
        -name '*.pth' -o \
        -name '*.ckpt' -o \
        -name 'trainer_state.json' -o \
        -name 'optimizer*' -o \
        -name 'scheduler*' -o \
        -name 'rng_state*' \
    \) -print -quit 2>/dev/null || true)
    if [[ -n "${payload}" ]]; then
        echo "Unexpected checkpoint payload: ${payload}" >&2
        return 1
    fi
}

if [[ $# -eq 2 && $1 == --audit-only ]]; then
    audit_artifacts "$2"
    exit $?
fi

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <benchmark> <method> <gpu> <timestamp>" >&2
    exit 2
fi

benchmark=$1
method=$2
physical_gpu=$3
timestamp=$4

conda_exe=${CONDA_EXE:-}
if [[ -z "${conda_exe}" ]]; then
    conda_exe=$(command -v conda || true)
fi
if [[ -z "${conda_exe}" && -x /root/miniconda3/bin/conda ]]; then
    conda_exe=/root/miniconda3/bin/conda
fi
if [[ -z "${conda_exe}" ]]; then
    echo "Unable to locate conda." >&2
    exit 127
fi
conda_base=$("${conda_exe}" info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate unlearning

repo_root=$(git rev-parse --show-toplevel)
shared_root=${ORTHOGRAD_SHARED_ROOT:-$(dirname "$(git rev-parse --git-common-dir)")}
artifact_root=${ORTHOGRAD_ARTIFACT_ROOT:-${shared_root}/saves/exp/ORTHOGRAD_MVP}
local_root=${ORTHOGRAD_LOCAL_ROOT:-/tmp/orthograd_mvp}
matrix_root="${artifact_root}/${timestamp}"
arm_dir="${matrix_root}/${benchmark}/${method}"
local_arm_dir="${local_root}/${timestamp}/${benchmark}/${method}"

if [[ -e "${arm_dir}" || -e "${local_arm_dir}" ]]; then
    echo "Refusing to overwrite an existing arm: ${benchmark}/${method}" >&2
    exit 2
fi
mkdir -p "${arm_dir}/.hydra" "${local_arm_dir}"

export CUDA_VISIBLE_DEVICES=${physical_gpu}
export HF_HOME=${ORTHOGRAD_HF_HOME:-${HF_HOME:-/root/.cache/huggingface}}
export TOKENIZERS_PARALLELISM=false

method_overrides=()
case "${method}" in
    control)
        method_overrides=(
            trainer=GeometricUnlearn
            trainer.method_args.geometric_config.gu_enabled=false
            trainer.method_args.geometric_config.optimizer_geometry=adam
        )
        ;;
    orthograd)
        method_overrides=(
            trainer=OrthogradUnlearn
            trainer.method_args.geometric_config.gu_enabled=true
            trainer.method_args.geometric_config.optimizer_geometry=euclidean
        )
        ;;
    gu_euclidean)
        method_overrides=(
            trainer=GeometricUnlearn
            trainer.method_args.geometric_config.gu_enabled=true
            trainer.method_args.geometric_config.optimizer_geometry=euclidean
        )
        ;;
    gu_adam)
        method_overrides=(
            trainer=GeometricUnlearn
            trainer.method_args.geometric_config.gu_enabled=true
            trainer.method_args.geometric_config.optimizer_geometry=adam
        )
        ;;
    *)
        echo "Unsupported method: ${method}" >&2
        exit 2
        ;;
esac

per_device_batch_size=4
component_buffer_device=parameter
basis_device=parameter
model_name=
base_model=
experiment=
retain_logs_path=null
summary_name=
dataset_overrides=()

case "${benchmark}" in
    tofu01|tofu05|tofu10)
        suffix=${benchmark#tofu}
        forget_split="forget${suffix}"
        retain_split="retain$((100 - 10#${suffix}))"
        holdout_split="holdout${suffix}"
        model_name=Llama-3.2-1B-Instruct
        base_model=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
        experiment=unlearn/tofu/default
        retain_logs_path="${shared_root}/saves/eval/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"
        summary_name=TOFU_SUMMARY.json
        dataset_overrides=(
            "forget_split=${forget_split}"
            "retain_split=${retain_split}"
            "holdout_split=${holdout_split}"
        )
        ;;
    muse_news|muse_books)
        data_split=News
        [[ "${benchmark}" == muse_books ]] && data_split=Books
        model_name=Llama-2-7b-hf
        base_model="muse-bench/MUSE-${data_split}_target"
        experiment=unlearn/muse/default
        retain_logs_path="${shared_root}/saves/eval/muse_${model_name}_${data_split}_retrain/MUSE_EVAL.json"
        summary_name=MUSE_SUMMARY.json
        per_device_batch_size=2
        component_buffer_device=cpu
        basis_device=cpu
        dataset_overrides=("data_split=${data_split}")
        ;;
    wmdp_cyber|wmdp_bio)
        data_split=${benchmark#wmdp_}
        model_name=zephyr-7b-beta
        base_model=HuggingFaceH4/zephyr-7b-beta
        experiment=unlearn/wmdp/default
        summary_name=LMEval_SUMMARY.json
        per_device_batch_size=2
        component_buffer_device=cpu
        basis_device=cpu
        corpus_root="${shared_root}/data/wmdp/wmdp-corpora"
        bash "${repo_root}/scripts/prepare_wmdp_data.sh" "${data_split}"
        dataset_overrides=(
            "data_split=${data_split}"
            "data.forget.WMDP_forget.args.hf_args.data_files=${corpus_root}/${data_split}-forget-corpus.jsonl"
            "data.retain.WMDP_retain.args.hf_args.data_files=${corpus_root}/${data_split}-retain-corpus.jsonl"
        )
        ;;
    *)
        echo "Unsupported benchmark: ${benchmark}" >&2
        exit 2
        ;;
esac

if [[ "${component_buffer_device}" == cpu ]]; then
    selected_numel=6738415616
    live_vectors=2
    [[ "${method}" == orthograd ]] && live_vectors=$((per_device_batch_size + 2))
    required_host_bytes=$((selected_numel * 4 * live_vectors * 12 / 10))
    available_host_kib=$(awk '/MemAvailable:/{print $2}' /proc/meminfo)
    available_host_bytes=$((available_host_kib * 1024))
    if (( available_host_bytes < required_host_bytes )); then
        echo "Insufficient host memory: requires ${required_host_bytes}, found ${available_host_bytes}." >&2
        exit 1
    fi
fi

diagnostics_path=null
if [[ "${method}" != control ]]; then
    diagnostics_path="${local_arm_dir}/diagnostics.jsonl"
fi
resource_profile_path="${local_arm_dir}/training_resource_profile.json"
task_name="orthograd_mvp_${benchmark}_${method}_${timestamp}"

command=(
    accelerate launch
    --config_file configs/accelerate/gu_single_gpu.yaml
    --gpu_ids 0
    src/train.py
    --config-name=unlearn.yaml
    "experiment=${experiment}"
    "${method_overrides[@]}"
    "task_name=${task_name}"
    "model=${model_name}"
    "model.model_args.pretrained_model_name_or_path=${base_model}"
    "model.tokenizer_args.pretrained_model_name_or_path=${base_model}"
    model.model_args.attn_implementation=flash_attention_2
    model.model_args.torch_dtype=bfloat16
    "retain_logs_path=${retain_logs_path}"
    "paths.output_dir=${local_arm_dir}"
    save_model_after_train=false
    "trainer.args.per_device_train_batch_size=${per_device_batch_size}"
    trainer.args.gradient_accumulation_steps=1
    trainer.args.max_steps=10
    trainer.args.learning_rate=1e-5
    trainer.args.optim=paged_adamw_32bit
    trainer.args.adam_beta1=0.0
    trainer.args.weight_decay=0.0
    trainer.args.warmup_epochs=0
    trainer.args.fp16=false
    trainer.args.bf16=true
    trainer.args.bf16_full_eval=true
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
    trainer.method_args.geometric_config.gradient_surgery=gu
    "trainer.method_args.geometric_config.diagnostics_path=${diagnostics_path}"
    "trainer.method_args.geometric_config.resource_profile_path=${resource_profile_path}"
    trainer.method_args.geometric_config.actual_delta_mode=off
    "${dataset_overrides[@]}"
)

if [[ "${component_buffer_device}" == cpu ]]; then
    command+=(trainer.method_args.geometric_config.component_buffer_device=cpu)
else
    command+=(trainer.method_args.geometric_config.component_buffer_device=parameter)
fi

if [[ "${method}" == orthograd ]]; then
    if [[ "${basis_device}" == cpu ]]; then
        command+=(trainer.method_args.orthograd_config.basis_device=cpu)
    else
        command+=(trainer.method_args.orthograd_config.basis_device=parameter)
    fi
    command+=(
        "trainer.method_args.orthograd_config.maximum_retain_batch_size=${per_device_batch_size}"
        "trainer.method_args.orthograd_config.diagnostics_path=${diagnostics_path}"
    )
fi

printf 'Launching'
printf ' %q' "${command[@]}"
printf '\n'
set +e
python scripts/profile_process_tree.py \
    --output "${local_arm_dir}/process_resource_profile.json" \
    --gpu "${physical_gpu}" \
    --poll-ms 100 \
    -- "${command[@]}" 2>&1 | tee "${local_arm_dir}/run.log"
exit_code=${PIPESTATUS[0]}
set -e

copy_if_present() {
    local source=$1
    local destination=$2
    if [[ -f "${source}" ]]; then
        local temporary="${destination}.tmp.$$"
        cp "${source}" "${temporary}"
        mv "${temporary}" "${destination}"
    fi
}

copy_if_present "${local_arm_dir}/run.log" "${arm_dir}/run.log"
copy_if_present "${local_arm_dir}/.hydra/config.yaml" "${arm_dir}/.hydra/config.yaml"
copy_if_present "${local_arm_dir}/diagnostics.jsonl" "${arm_dir}/diagnostics.jsonl"
copy_if_present "${local_arm_dir}/diagnostics.summary.json" "${arm_dir}/diagnostics.summary.json"
copy_if_present "${local_arm_dir}/training_resource_profile.json" "${arm_dir}/training_resource_profile.json"
copy_if_present "${local_arm_dir}/process_resource_profile.json" "${arm_dir}/process_resource_profile.json"

summary_path=$(find "${local_arm_dir}" -type f -name "${summary_name}" -print -quit 2>/dev/null || true)
if [[ -n "${summary_path}" ]]; then
    copy_if_present "${summary_path}" "${arm_dir}/${summary_name}"
fi
if [[ "${benchmark}" == wmdp_* ]]; then
    copy_if_present "${shared_root}/data/wmdp/wmdp-corpora/provenance.json" "${arm_dir}/wmdp_provenance.json"
fi

audit_artifacts "${local_arm_dir}"
audit_artifacts "${arm_dir}"
if [[ ${exit_code} -ne 0 ]]; then
    echo "Arm failed with exit code ${exit_code}: ${benchmark}/${method}" >&2
    exit "${exit_code}"
fi
if [[ -z "${summary_path}" ]]; then
    echo "Missing ${summary_name} for ${benchmark}/${method}." >&2
    exit 1
fi
if [[ ! -f "${arm_dir}/.hydra/config.yaml" || ! -f "${arm_dir}/training_resource_profile.json" || ! -f "${arm_dir}/process_resource_profile.json" ]]; then
    echo "Missing required profiling/config artifacts for ${benchmark}/${method}." >&2
    exit 1
fi

echo "Completed ${benchmark}/${method}: ${arm_dir}"
