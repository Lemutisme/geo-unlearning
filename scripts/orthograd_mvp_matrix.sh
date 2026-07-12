#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <timestamp> [gpu]" >&2
    exit 2
fi

timestamp=$1
gpu=${2:-${ORTHOGRAD_GPU:-0}}
shared_root=${ORTHOGRAD_SHARED_ROOT:-$(dirname "$(git rev-parse --git-common-dir)")}
artifact_root=${ORTHOGRAD_ARTIFACT_ROOT:-${shared_root}/saves/exp/ORTHOGRAD_MVP}
matrix_root="${artifact_root}/${timestamp}"
manifest="${matrix_root}/RUN_MANIFEST.tsv"

if [[ -e "${manifest}" ]]; then
    echo "Refusing to overwrite manifest: ${manifest}" >&2
    exit 2
fi
mkdir -p "${matrix_root}"
printf 'benchmark\tmethod\tgpu\tstart_utc\tend_utc\texit_code\tcommand\tarm_dir\n' > "${manifest}"

append_manifest() {
    local row=$1
    local temporary
    temporary=$(mktemp "${matrix_root}/.manifest.XXXXXX")
    cp "${manifest}" "${temporary}"
    printf '%s\n' "${row}" >> "${temporary}"
    mv "${temporary}" "${manifest}"
}

benchmarks=(tofu01 tofu05 tofu10 muse_news muse_books wmdp_cyber wmdp_bio)
methods=(control orthograd gu_euclidean gu_adam)
failure=0

for benchmark in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        command="bash scripts/orthograd_mvp_arm.sh ${benchmark} ${method} ${gpu} ${timestamp}"
        start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        set +e
        bash scripts/orthograd_mvp_arm.sh \
            "${benchmark}" "${method}" "${gpu}" "${timestamp}"
        exit_code=$?
        set -e
        end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        arm_dir="${matrix_root}/${benchmark}/${method}"
        row=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
            "${benchmark}" "${method}" "${gpu}" "${start_utc}" \
            "${end_utc}" "${exit_code}" "${command}" "${arm_dir}")
        append_manifest "${row}"
        if [[ ${exit_code} -ne 0 ]]; then
            failure=1
            echo "Recorded failed arm ${benchmark}/${method} (${exit_code}); continuing the matrix." >&2
        fi
    done
done

if [[ ${failure} -ne 0 ]]; then
    echo "Matrix completed with failed arms: ${matrix_root}" >&2
    exit 1
fi
echo "Completed Orthograd MVP matrix: ${matrix_root}"
