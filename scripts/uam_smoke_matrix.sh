#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <timestamp>" >&2
    exit 2
fi

timestamp=$1
if [[ ! "${timestamp}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid timestamp: ${timestamp}" >&2
    exit 2
fi

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
active_count=0

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
    active_count=$((active_count + 1))
    echo "Started pid=${active_pid[slot]} gpu=${slot} method=${method}"
}

record_completion() {
    local slot=$1
    local pid=$2
    local exit_code=$3

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
    active_count=$((active_count - 1))
}

reap_next_completion() {
    local -a pids=()
    local slot
    for slot in 0 1; do
        if [[ -n "${active_pid[slot]}" ]]; then
            pids+=("${active_pid[slot]}")
        fi
    done
    if (( ${#pids[@]} == 0 )); then
        echo "Internal scheduler error: no active PID to reap." >&2
        return 1
    fi

    local completed_pid=""
    local exit_code
    set +e
    wait -n -p completed_pid "${pids[@]}"
    exit_code=$?
    set -e
    if [[ -z "${completed_pid}" ]]; then
        echo "Internal scheduler error: wait -n returned no PID." >&2
        return 1
    fi

    local completed_slot=""
    for slot in 0 1; do
        if [[ "${active_pid[slot]}" == "${completed_pid}" ]]; then
            completed_slot=${slot}
            break
        fi
    done
    if [[ -z "${completed_slot}" ]]; then
        echo "Internal scheduler error: unknown completed PID ${completed_pid}." >&2
        return 1
    fi
    record_completion "${completed_slot}" "${completed_pid}" "${exit_code}"
}

next_arm=0
for slot in 0 1; do
    launch_arm "${slot}" "${arms[next_arm]}"
    next_arm=$((next_arm + 1))
done

while (( active_count > 0 )); do
    reap_next_completion
    if [[ ${failure} -eq 0 && ${next_arm} -lt ${#arms[@]} ]]; then
        for slot in 0 1; do
            if [[ -z "${active_pid[slot]}" ]]; then
                launch_arm "${slot}" "${arms[next_arm]}"
                next_arm=$((next_arm + 1))
                break
            fi
        done
    fi
done

if [[ ${failure} -ne 0 ]]; then
    exit 1
fi
echo "Completed UAM smoke matrix: ${matrix_root}"
