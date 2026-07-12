#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <timestamp>" >&2
    exit 2
fi

timestamp=$1
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

matrix_root="saves/exp/UAM_SMOKE/${timestamp}"
manifest="${matrix_root}/RUN_MANIFEST.tsv"
mkdir -p "${matrix_root}"
printf 'pid\tgpu\tmethod\tstart_utc\tend_utc\texit_code\tcommand\n' > "${manifest}"

arms=(uam_nll uam_simnpo uam_gu_nll uam_gu_simnpo)

declare -a active_pid
active_pid[0]=""
active_pid[1]=""
declare -a active_method=("" "")
declare -a active_start=("" "")
declare -a active_command=("" "")
failure=0

launch_arm() {
    local slot=$1
    local method=$2
    local command="bash scripts/uam_smoke_arm.sh ${method} ${slot} ${timestamp}"
    local start_utc
    start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    bash scripts/uam_smoke_arm.sh "${method}" "${slot}" "${timestamp}" &
    active_pid[slot]=$!
    active_method[slot]=${method}
    active_start[slot]=${start_utc}
    active_command[slot]=${command}
    echo "Started pid=${active_pid[slot]} gpu=${slot} method=${method}"
}

wait_for_slot() {
    local slot=$1
    local pid=${active_pid[slot]}
    [[ -n "${pid}" ]] || return 0

    local exit_code
    set +e
    wait "${pid}"
    exit_code=$?
    set -e

    local end_utc
    end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pid}" "${slot}" "${active_method[slot]}" \
        "${active_start[slot]}" "${end_utc}" "${exit_code}" \
        "${active_command[slot]}" >> "${manifest}"
    if [[ ${exit_code} -ne 0 ]]; then
        failure=1
        echo "Arm failed pid=${pid} exit_code=${exit_code}; no new arms will be scheduled." >&2
    fi
    active_pid[slot]=""
}

next_slot=0
for method in "${arms[@]}"; do
    if [[ ${failure} -ne 0 ]]; then
        break
    fi
    if [[ -n "${active_pid[next_slot]}" ]]; then
        wait_for_slot "${next_slot}"
        if [[ ${failure} -ne 0 ]]; then
            break
        fi
    fi
    launch_arm "${next_slot}" "${method}"
    next_slot=$((1 - next_slot))
done

for slot in 0 1; do
    wait_for_slot "${slot}"
done

if [[ ${failure} -ne 0 ]]; then
    exit 1
fi
echo "Completed UAM smoke matrix: ${matrix_root}"
