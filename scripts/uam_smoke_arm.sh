#!/usr/bin/env bash

set -euo pipefail

checkpoint_file_payload() {
    local audit_root=$1
    find "${audit_root}" -type f \( \
        -name '*.safetensors' -o \
        -name '*.bin' -o \
        -name '*.pt' -o \
        -name '*.pth' -o \
        -name '*.ckpt' -o \
        -name 'trainer_state.json' -o \
        -name 'model_state*' -o \
        -name 'training_args*' -o \
        -name 'optimizer*' -o \
        -name 'scheduler*' -o \
        -name 'rng_state*' \
    \) -print -quit
}

audit_checkpoint_files() {
    local audit_root=$1
    local unexpected
    unexpected=$(checkpoint_file_payload "${audit_root}")
    if [[ -n "${unexpected}" ]]; then
        echo "Unexpected checkpoint payload: ${unexpected}" >&2
        return 1
    fi
}

audit_symlinks() {
    local audit_root=$1
    local unexpected
    unexpected=$(find "${audit_root}" -type l -print -quit)
    if [[ -n "${unexpected}" ]]; then
        echo "Unexpected symlink: ${unexpected}" >&2
        return 1
    fi
}

audit_checkpoint_payloads() {
    local audit_root=$1
    local unexpected

    if [[ ! -d "${audit_root}" ]]; then
        echo "Audit root is not a directory: ${audit_root}" >&2
        return 1
    fi
    audit_symlinks "${audit_root}"
    audit_checkpoint_files "${audit_root}"
    unexpected=$(find "${audit_root}" -type d -name 'checkpoint-*' -print -quit)
    if [[ -n "${unexpected}" ]]; then
        echo "Unexpected checkpoint payload: ${unexpected}" >&2
        return 1
    fi
}

audit_persistent_allowlist() {
    local persistent_root=$1
    local artifact relative
    while IFS= read -r -d '' artifact; do
        relative=${artifact#"${persistent_root}/"}
        case "${relative}" in
            .hydra|evals)
                if [[ ! -d "${artifact}" || -L "${artifact}" ]]; then
                    echo "Unexpected persistent artifact: ${artifact}" >&2
                    return 1
                fi
                ;;
            .hydra/config.yaml|evals/TOFU_SUMMARY.json|run.log|UAMUnlearn.log|uam_diagnostics.jsonl|uam_diagnostics.summary.json)
                if [[ ! -f "${artifact}" || -L "${artifact}" ]]; then
                    echo "Unexpected persistent artifact: ${artifact}" >&2
                    return 1
                fi
                ;;
            *)
                echo "Unexpected persistent artifact: ${artifact}" >&2
                return 1
                ;;
        esac
    done < <(find "${persistent_root}" -mindepth 1 -print0)
}

validate_persistent_matrix_root() {
    local expected_matrix_root=$1
    local current=""
    local component
    for component in saves exp UAM_SMOKE "${timestamp}"; do
        current=${current:+${current}/}${component}
        if [[ -L "${current}" ]]; then
            echo "Persistent path contains symlink: ${current}" >&2
            return 1
        fi
        if [[ ! -d "${current}" ]]; then
            echo "Persistent matrix path is not a directory: ${current}" >&2
            return 1
        fi
    done

    local expected actual
    expected="$(pwd -P)/${expected_matrix_root}"
    actual=$(cd "${expected_matrix_root}" && pwd -P)
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Persistent path escapes repository containment: ${actual}" >&2
        return 1
    fi
}

persist_tmp=""
training_pid=""
created_local_staging=0
cleanup_persist_tmp() {
    if [[ -n "${persist_tmp}" && -d "${persist_tmp}" ]]; then
        rm -rf -- "${persist_tmp}"
    fi
}
trap cleanup_persist_tmp EXIT

training_group_is_running() {
    [[ -n "${training_pid}" ]] || return 1
    kill -0 -- "-${training_pid}" 2>/dev/null
}

terminate_training_group() {
    [[ -n "${training_pid}" ]] || return 0
    if training_group_is_running; then
        kill -TERM -- "-${training_pid}" 2>/dev/null || true
    fi
    local deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )) && training_group_is_running; do
        sleep 0.05
    done
    if training_group_is_running; then
        kill -KILL -- "-${training_pid}" 2>/dev/null || true
    fi
    set +e
    wait "${training_pid}" 2>/dev/null
    set -e
    training_pid=""
}

handle_arm_signal() {
    local exit_code=$1
    trap - INT TERM HUP
    terminate_training_group
    if [[ ${created_local_staging} -ne 0 && -n "${local_arm_dir:-}" ]]; then
        rm -rf -- "${local_arm_dir}"
    fi
    exit "${exit_code}"
}

trap 'handle_arm_signal 130' INT
trap 'handle_arm_signal 143' TERM
trap 'handle_arm_signal 129' HUP

if [[ $# -eq 2 && $1 == --audit-only ]]; then
    audit_checkpoint_payloads "$2"
    exit $?
fi

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <method> <gpu> <timestamp>" >&2
    echo "       $0 --audit-only <root>" >&2
    exit 2
fi

method=$1
gpu=$2
timestamp=$3

case "${method}" in
    uam_nll)
        mode=uam; signal=nll; normalization=fixed_loss ;;
    uam_simnpo)
        mode=uam; signal=simnpo; normalization=fixed_loss ;;
    uam_gu_nll)
        mode=uam_gu; signal=nll; normalization=metric_trust ;;
    uam_gu_simnpo)
        mode=uam_gu; signal=simnpo; normalization=metric_trust ;;
    *)
        echo "Unsupported method: ${method}" >&2
        exit 2
        ;;
esac

if [[ ! "${timestamp}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Unsafe timestamp: ${timestamp}" >&2
    exit 2
fi
if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU: ${gpu}" >&2
    exit 2
fi
smoke_rho=${UAM_SMOKE_RHO:-0.05}

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

if ! python -c \
    'import math, sys; value = float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0.0))' \
    "${smoke_rho}" 2>/dev/null; then
    echo "Invalid UAM_SMOKE_RHO: ${smoke_rho}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME=/root/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

matrix_root="saves/exp/UAM_SMOKE/${timestamp}"
arm_dir="${matrix_root}/${method}"
local_root=${UAM_LOCAL_ROOT:-/tmp/uam_smoke}
local_arm_dir="${local_root}/${timestamp}/${method}"
task_name="uam_smoke_${method}_${timestamp}"
diagnostics_path="${local_arm_dir}/uam_diagnostics.jsonl"

expected_arm_dir="saves/exp/UAM_SMOKE/${timestamp}/${method}"
if [[ "${arm_dir}" != "${expected_arm_dir}" ]]; then
    echo "Refusing unexpected persistent arm path: ${arm_dir}" >&2
    exit 2
fi
validate_persistent_matrix_root "${matrix_root}"
if [[ -e "${arm_dir}" || -L "${arm_dir}" ]]; then
    echo "Persistent arm already exists: ${arm_dir}" >&2
    exit 1
fi
if [[ -e "${local_arm_dir}" || -L "${local_arm_dir}" ]]; then
    echo "Local staging already exists: ${local_arm_dir}" >&2
    exit 1
fi
mkdir -p "$(dirname "${local_arm_dir}")"
mkdir "${local_arm_dir}"
created_local_staging=1

command=(
    accelerate launch
    --config_file configs/accelerate/gu_single_gpu.yaml
    --gpu_ids "${gpu}"
    src/train.py
    --config-name=unlearn.yaml
    experiment=unlearn/tofu/default
    trainer=UAMUnlearn
    "task_name=${task_name}"
    model=Llama-3.2-1B-Instruct
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
    model.model_args.torch_dtype=bfloat16
    model.model_args.attn_implementation=flash_attention_2
    retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json
    forget_split=forget01
    retain_split=retain99
    holdout_split=holdout01
    "paths.output_dir=${local_arm_dir}"
    save_model_after_train=false
    trainer.args.per_device_train_batch_size=4
    trainer.args.gradient_accumulation_steps=8
    trainer.args.max_steps=10
    trainer.args.learning_rate=1e-5
    trainer.args.optim=paged_adamw_32bit
    trainer.args.adam_beta1=0.0
    trainer.args.weight_decay=0.0
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
    "trainer.method_args.geometric_config.diagnostics_path=${diagnostics_path}"
    trainer.method_args.geometric_config.actual_delta_mode=full
    'trainer.method_args.geometric_config.actual_delta_steps=[1,10]'
    "trainer.method_args.uam_config.mode=${mode}"
    "trainer.method_args.uam_config.forget_signal=${signal}"
    "trainer.method_args.uam_config.perturbation_normalization=${normalization}"
    "trainer.method_args.uam_config.rho=${smoke_rho}"
)

printf 'Launching %q ' "${command[@]}"
printf '\n'
setsid "${command[@]}" > "${local_arm_dir}/run.log" 2>&1 &
training_pid=$!
set +e
wait "${training_pid}"
training_exit_code=$?
set -e
training_pid=""
if [[ ${training_exit_code} -ne 0 ]]; then
    echo "Training command failed with exit code ${training_exit_code}." >&2
    exit "${training_exit_code}"
fi

audit_symlinks "${local_arm_dir}"
audit_checkpoint_files "${local_arm_dir}"

mapfile -t summary_paths < <(
    find "${local_arm_dir}" -type f -name TOFU_SUMMARY.json -print
)
if (( ${#summary_paths[@]} != 1 )); then
    echo "Expected exactly one TOFU_SUMMARY.json under ${local_arm_dir}; found ${#summary_paths[@]}." >&2
    exit 1
fi
summary_path=${summary_paths[0]}

mkdir -p "${local_arm_dir}/evals"
normalized_summary="${local_arm_dir}/evals/TOFU_SUMMARY.json"
if [[ "${summary_path}" != "${normalized_summary}" ]]; then
    cp "${summary_path}" "${normalized_summary}"
fi
mapfile -t evaluation_checkpoint_dirs < <(
    find "${local_arm_dir}" -depth -type d -name 'checkpoint-*' -print
)
for checkpoint_dir in "${evaluation_checkpoint_dirs[@]}"; do
    rm -rf -- "${checkpoint_dir}"
done

audit_checkpoint_payloads "${local_arm_dir}"
mapfile -t summary_paths < <(
    find "${local_arm_dir}" -type f -name TOFU_SUMMARY.json -print
)
if (( ${#summary_paths[@]} != 1 )); then
    echo "Expected exactly one normalized TOFU_SUMMARY.json under ${local_arm_dir}; found ${#summary_paths[@]}." >&2
    exit 1
fi
summary_path=${summary_paths[0]}

for required_artifact in \
    "${local_arm_dir}/.hydra/config.yaml" \
    "${local_arm_dir}/uam_diagnostics.jsonl" \
    "${local_arm_dir}/uam_diagnostics.summary.json"; do
    if [[ ! -f "${required_artifact}" ]]; then
        echo "Missing required smoke artifact: ${required_artifact}" >&2
        exit 1
    fi
done

persist_tmp=$(mktemp -d "${matrix_root}/.${method}.tmp.XXXXXX")
mkdir "${persist_tmp}/evals" "${persist_tmp}/.hydra"
cp "${summary_path}" "${persist_tmp}/evals/TOFU_SUMMARY.json"
cp "${local_arm_dir}/run.log" "${persist_tmp}/run.log"
cp "${local_arm_dir}/.hydra/config.yaml" "${persist_tmp}/.hydra/config.yaml"
cp "${local_arm_dir}/uam_diagnostics.jsonl" \
    "${persist_tmp}/uam_diagnostics.jsonl"
cp "${local_arm_dir}/uam_diagnostics.summary.json" \
    "${persist_tmp}/uam_diagnostics.summary.json"
if [[ -f "${local_arm_dir}/UAMUnlearn.log" ]]; then
    cp "${local_arm_dir}/UAMUnlearn.log" "${persist_tmp}/UAMUnlearn.log"
fi

audit_checkpoint_payloads "${persist_tmp}"
audit_persistent_allowlist "${persist_tmp}"
while IFS= read -r -d '' artifact; do
    sync -f "${artifact}"
done < <(find "${persist_tmp}" -type f -print0)
sync -f "${persist_tmp}"

if [[ -e "${arm_dir}" || -L "${arm_dir}" ]]; then
    echo "Persistent arm already exists: ${arm_dir}" >&2
    exit 1
fi
mv -T -n "${persist_tmp}" "${arm_dir}"
if [[ -d "${persist_tmp}" ]]; then
    echo "Atomic persistent arm publication was not completed: ${arm_dir}" >&2
    exit 1
fi
persist_tmp=""
sync -f "${matrix_root}"
audit_checkpoint_payloads "${arm_dir}"
audit_persistent_allowlist "${arm_dir}"
persistent_summary="${arm_dir}/evals/TOFU_SUMMARY.json"
echo "Completed ${method}: ${persistent_summary}"
