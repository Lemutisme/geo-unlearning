#!/usr/bin/env bash

set -euo pipefail

audit_checkpoint_payloads() {
    local root=$1
    if [[ ! -d "${root}" ]]; then
        echo "Audit root is not a directory: ${root}" >&2
        return 1
    fi
    local unexpected
    unexpected=$(find "${root}" -type l -print -quit)
    if [[ -n "${unexpected}" ]]; then
        echo "Unexpected checkpoint payload (symlink): ${unexpected}" >&2
        return 1
    fi
    unexpected=$(find "${root}" -type d -name 'checkpoint-*' -print -quit)
    if [[ -n "${unexpected}" ]]; then
        echo "Unexpected checkpoint payload: ${unexpected}" >&2
        return 1
    fi
    unexpected=$(find "${root}" -type f \( \
        -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o \
        -name '*.pth' -o -name '*.ckpt' -o -name 'trainer_state.json' -o \
        -name 'optimizer*' -o -name 'scheduler*' -o -name 'rng_state*' \
        \) -print -quit)
    if [[ -n "${unexpected}" ]]; then
        echo "Unexpected checkpoint payload: ${unexpected}" >&2
        return 1
    fi
}

if [[ $# -eq 2 && $1 == --audit-only ]]; then
    audit_checkpoint_payloads "$2"
    exit $?
fi

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <arm> <gpu> <run-id>" >&2
    echo "       $0 --audit-only <root>" >&2
    exit 2
fi

arm=$1
gpu=$2
run_id=$3
case "${arm}" in
    base) ;;
    rmu) trainer_override=trainer=WMDPRMUUnlearn; gu_enabled=false ;;
    rmu_gu) trainer_override=trainer=WMDPRMUUnlearn; gu_enabled=true ;;
    uam) trainer_override=trainer=WMDPUAMUnlearn; uam_mode=uam ;;
    uam_gu) trainer_override=trainer=WMDPUAMUnlearn; uam_mode=uam_gu ;;
    *) echo "Unsupported W2 arm: ${arm}" >&2; exit 2 ;;
esac
if [[ "${gpu}" != "0" ]]; then
    echo "W2 is restricted to GPU 0." >&2
    exit 2
fi
if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Unsafe run id: ${run_id}" >&2
    exit 2
fi

learning_rate=${WMDP_W2_LEARNING_RATE:-5e-5}
max_steps=${WMDP_W2_MAX_STEPS:-80}
lr_scheduler_type=${WMDP_W2_LR_SCHEDULER_TYPE:-linear}
if ! [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WMDP W2 max steps must be a positive integer: ${max_steps}" >&2
    exit 2
fi
case "${lr_scheduler_type}" in
    linear|constant) ;;
    *)
        echo "WMDP W2 scheduler must be linear or constant: ${lr_scheduler_type}" >&2
        exit 2
        ;;
esac
if ! python - "${learning_rate}" <<'PY'
import math
import sys

raw = sys.argv[1]
try:
    value = float(raw)
except ValueError:
    value = float("nan")
if raw.lower() in {"true", "false"} or not math.isfinite(value) or value <= 0.0:
    print(f"WMDP W2 learning rate must be positive and finite: {raw}", file=sys.stderr)
    raise SystemExit(2)
PY
then
    exit 2
fi

conda_exe=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "${conda_exe}" ]]; then
    conda_exe=/root/miniconda3/bin/conda
fi
conda_base=$("${conda_exe}" info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate unlearning

export CUDA_VISIBLE_DEVICES="0"
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

repo_data_root=${WMDP_DATA_ROOT:-/workspace/re/GU/geo-unlearning/data}
cyber_corpus="${repo_data_root}/wmdp/wmdp-corpora/cyber-forget-corpus.jsonl"
if [[ ! -f "${cyber_corpus}" ]]; then
    echo "Missing WMDP-Cyber corpus: ${cyber_corpus}" >&2
    exit 1
fi
persist_root=${WMDP_W2_PERSIST_ROOT:-/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2}
run_root="${persist_root}/${run_id}"
arm_dir="${run_root}/${arm}"
local_root=${WMDP_W2_LOCAL_ROOT:-/tmp/wmdp_uam_w2}
local_arm="${local_root}/${run_id}/${arm}"
if [[ -e "${arm_dir}" || -L "${arm_dir}" ]]; then
    echo "Persistent arm already exists: ${arm_dir}" >&2
    exit 1
fi
if [[ -e "${local_arm}" || -L "${local_arm}" ]]; then
    echo "Local staging already exists: ${local_arm}" >&2
    exit 1
fi
mkdir -p "${run_root}" "$(dirname "${local_arm}")"
mkdir "${local_arm}"

cleanup_local=1
cleanup() {
    if [[ ${cleanup_local} -eq 1 && -d "${local_arm}" ]]; then
        rm -rf -- "${local_arm}"
    fi
}
trap cleanup EXIT

task_name="wmdp_uam_w2_${arm}_${run_id}"
diagnostics_path="${local_arm}/geometry.jsonl"
if [[ "${arm}" == "base" ]]; then
    command=(
        accelerate launch
        --config_file configs/accelerate/gu_single_gpu.yaml
        --gpu_ids 0
        src/eval.py
        --config-name=eval.yaml
        experiment=eval/wmdp/default
        "task_name=${task_name}"
        model=zephyr-7b-beta
        model.model_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta
        model.tokenizer_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta
        model.model_args.torch_dtype=bfloat16
        model.model_args.attn_implementation=flash_attention_2
        "paths.output_dir=${local_arm}"
        "eval.lm_eval.output_dir=${local_arm}/evals"
        eval.lm_eval.overwrite=true
        eval.lm_eval.simple_evaluate_args.batch_size=8
        seed=42
    )
else
    command=(
        accelerate launch
        --config_file configs/accelerate/gu_single_gpu.yaml
        --gpu_ids 0
        src/train.py
        --config-name=unlearn.yaml
        experiment=unlearn/wmdp/default
        "${trainer_override}"
        "task_name=${task_name}"
        model=zephyr-7b-beta
        model.model_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta
        model.tokenizer_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta
        model.model_args.torch_dtype=bfloat16
        model.model_args.attn_implementation=flash_attention_2
        "data.forget.WMDP_forget.args.hf_args.data_files=${cyber_corpus}"
        data/datasets@data.retain=WMDP_wikitext_retain
        "~data.retain.WMDP_retain"
        "paths.output_dir=${local_arm}"
        save_model_after_train=false
        trainer.args.per_device_train_batch_size=1
        trainer.args.gradient_accumulation_steps=4
        "trainer.args.max_steps=${max_steps}"
        "trainer.args.learning_rate=${learning_rate}"
        "trainer.args.lr_scheduler_type=${lr_scheduler_type}"
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
        trainer.args.seed=42
        trainer.args.data_seed=42
        "trainer.method_args.geometric_config.diagnostics_path=${diagnostics_path}"
        trainer.method_args.geometric_config.actual_delta_mode=full
        "trainer.method_args.geometric_config.actual_delta_steps=[1,${max_steps}]"
        eval.lm_eval.overwrite=true
        eval.lm_eval.simple_evaluate_args.batch_size=8
    )
    if [[ "${arm}" == rmu* ]]; then
        command+=("trainer.method_args.geometric_config.gu_enabled=${gu_enabled}")
    else
        command+=("trainer.method_args.uam_config.mode=${uam_mode}")
    fi
fi

start_seconds=$(date +%s.%N)
printf 'Launching %q ' "${command[@]}"
printf '\n'
setsid "${command[@]}" \
    > "${local_arm}/run.log" 2>&1 &
process_pid=$!
peak_nvml_mib=0
max_rss_kib=0
while kill -0 "${process_pid}" 2>/dev/null; do
    current=$(nvidia-smi --id=0 --query-compute-apps=used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk '{sum += $1} END {print sum + 0}')
    if [[ "${current}" =~ ^[0-9]+$ ]] && (( current > peak_nvml_mib )); then
        peak_nvml_mib=${current}
    fi
    current_rss=$(ps -o rss= -g "${process_pid}" 2>/dev/null | \
        awk '{sum += $1} END {print sum + 0}' || true)
    if [[ "${current_rss}" =~ ^[0-9]+$ ]] && (( current_rss > max_rss_kib )); then
        max_rss_kib=${current_rss}
    fi
    sleep 0.2
done
set +e
wait "${process_pid}"
exit_code=$?
set -e
if [[ ${exit_code} -ne 0 ]]; then
    if [[ -f "${local_arm}/run.log" ]]; then
        echo "Last 200 lines from failed W2 arm ${arm}:" >&2
        tail -200 "${local_arm}/run.log" >&2
    fi
    echo "W2 arm failed with exit code ${exit_code}: ${arm}" >&2
    exit "${exit_code}"
fi
end_seconds=$(date +%s.%N)

mapfile -t summaries < <(find "${local_arm}" -type f -name LMEval_SUMMARY.json -print)
if (( ${#summaries[@]} != 1 )); then
    echo "Expected one LMEval_SUMMARY.json, found ${#summaries[@]}." >&2
    exit 1
fi
mkdir -p "${local_arm}/evals"
if [[ "${summaries[0]}" != "${local_arm}/evals/LMEval_SUMMARY.json" ]]; then
    cp "${summaries[0]}" "${local_arm}/evals/LMEval_SUMMARY.json"
fi
find "${local_arm}" -depth -type d -name 'checkpoint-*' -exec rm -rf {} +

training_wall=null
if [[ "${arm}" != "base" ]]; then
    training_wall=$(python - "${local_arm}/run.log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
matches = re.findall(r"['\"]train_runtime['\"]:\s*([0-9.eE+-]+)", text)
print(matches[-1] if matches else "null")
PY
)
fi
python - "${local_arm}/resource.json" "${start_seconds}" "${end_seconds}" \
    "${training_wall}" "${peak_nvml_mib}" "${max_rss_kib}" <<'PY'
import json
import sys

training = None if sys.argv[4] == "null" else float(sys.argv[4])
value = {
    "wall_seconds": float(sys.argv[3]) - float(sys.argv[2]),
    "training_wall_seconds": training,
    "peak_nvml_mib": float(sys.argv[5]),
    "peak_cuda_allocated_mib": None,
    "max_rss_kib": float(sys.argv[6]),
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, indent=2) + "\n")
PY

cyber_sha=$(sha256sum "${cyber_corpus}" | awk '{print $1}')
python - "${local_arm}/provenance.json" "${cyber_sha}" <<'PY'
import importlib.metadata
import json
import sys

value = {
    "model": "HuggingFaceH4/zephyr-7b-beta",
    "tokenizer": "HuggingFaceH4/zephyr-7b-beta",
    "lm_eval_version": importlib.metadata.version("lm_eval"),
    "cyber_sha256": sys.argv[2],
    "retain_dataset": "wikitext/wikitext-2-raw-v1/train",
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(value, indent=2) + "\n")
PY

if [[ "${arm}" != "base" && ! -s "${diagnostics_path}" ]]; then
    echo "Missing W2 geometry diagnostics: ${diagnostics_path}" >&2
    exit 1
fi
audit_checkpoint_payloads "${local_arm}"

publish_tmp=$(mktemp -d "${run_root}/.${arm}.tmp.XXXXXX")
mkdir "${publish_tmp}/evals" "${publish_tmp}/.hydra"
cp "${local_arm}/evals/LMEval_SUMMARY.json" "${publish_tmp}/evals/"
cp "${local_arm}/resource.json" "${publish_tmp}/resource.json"
cp "${local_arm}/provenance.json" "${publish_tmp}/provenance.json"
cp "${local_arm}/run.log" "${publish_tmp}/run.log"
cp "${local_arm}/.hydra/config.yaml" "${publish_tmp}/.hydra/config.yaml"
if [[ "${arm}" != "base" ]]; then
    cp "${diagnostics_path}" "${publish_tmp}/geometry.jsonl"
fi
audit_checkpoint_payloads "${publish_tmp}"
mv -T -n "${publish_tmp}" "${arm_dir}"
audit_checkpoint_payloads "${arm_dir}"
rm -rf -- "${local_arm}"
cleanup_local=0
echo "Completed W2 arm ${arm}: ${arm_dir}/evals/LMEval_SUMMARY.json"
