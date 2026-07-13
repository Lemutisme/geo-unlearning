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
persist_root=${WMDP_W2_PERSIST_ROOT:-/workspace/re/GU/geo-unlearning/saves/exp/WMDP_UAM_W2}
run_root="${persist_root}/${run_id}"
if [[ -e "${run_root}" || -L "${run_root}" ]]; then
    echo "Matrix root already exists: ${run_root}" >&2
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
        echo "Stopping W2 matrix after failed arm: ${arm}" >&2
        exit "${exit_code}"
    fi
}

run_arm "base"
python scripts/analyze_wmdp_uam_w2.py \
    --matrix-root "${run_root}" \
    --check-base-gate
run_arm "rmu"
run_arm "rmu_gu"
run_arm "uam"
run_arm "uam_gu"
python scripts/analyze_wmdp_uam_w2.py \
    --matrix-root "${run_root}" \
    --output-json "${run_root}/W2_SUMMARY.json" \
    --output-markdown "${run_root}/W2_TABLE.md"
echo "Completed W2 matrix: ${run_root}"
