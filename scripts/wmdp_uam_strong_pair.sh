#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <run-id>" >&2
    exit 2
fi
run_id=$1
if [[ ! "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Unsafe run id: ${run_id}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export WMDP_W2_LEARNING_RATE=1.25e-4
export WMDP_W2_MAX_STEPS=160
export WMDP_W2_LR_SCHEDULER_TYPE=constant
export WMDP_W2_PERSIST_ROOT=${WMDP_UAM_STRONG_ROOT:-/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2_STRONG}

primary_summary=${WMDP_UAM_PRIMARY_SUMMARY:-/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2/w2-20260713-2/W2_SUMMARY.json}
if [[ ! -f "${primary_summary}" ]]; then
    echo "Missing primary W2 summary: ${primary_summary}" >&2
    exit 1
fi

run_root="${WMDP_W2_PERSIST_ROOT}/${run_id}"
if [[ -e "${run_root}" || -L "${run_root}" ]]; then
    echo "Strong-pair root already exists: ${run_root}" >&2
    exit 1
fi
mkdir -p "${run_root}"
manifest="${run_root}/RUN_MANIFEST.tsv"
printf 'arm\tstart_utc\tend_utc\texit_code\tcommand\n' > "${manifest}"

run_arm() {
    local arm=$1
    local start end command exit_code
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    command="bash scripts/wmdp_uam_w2_arm.sh ${arm} 0 ${run_id}"
    set +e
    bash scripts/wmdp_uam_w2_arm.sh "${arm}" 0 "${run_id}"
    exit_code=$?
    set -e
    end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${start}" "${end}" "${exit_code}" "${command}" >> "${manifest}"
    if [[ ${exit_code} -ne 0 ]]; then
        echo "Stopping strong pair after failed arm: ${arm}" >&2
        exit "${exit_code}"
    fi
}

run_arm "uam"
run_arm "uam_gu"
python scripts/analyze_wmdp_uam_strong_pair.py \
    --matrix-root "${run_root}" \
    --primary-summary "${primary_summary}"
echo "Completed strong UAM pair: ${run_root}"
