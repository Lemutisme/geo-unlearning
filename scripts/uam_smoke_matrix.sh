#!/usr/bin/env bash

set -euo pipefail

prepare_persistent_base() {
    local current=""
    local component
    for component in saves exp UAM_SMOKE; do
        current=${current:+${current}/}${component}
        if [[ -L "${current}" ]]; then
            echo "Persistent path contains symlink: ${current}" >&2
            return 1
        fi
        if [[ -e "${current}" && ! -d "${current}" ]]; then
            echo "Persistent path component is not a directory: ${current}" >&2
            return 1
        fi
        if [[ ! -e "${current}" ]]; then
            mkdir "${current}"
        fi
    done

    local expected actual
    expected="$(pwd -P)/saves/exp/UAM_SMOKE"
    actual=$(cd saves/exp/UAM_SMOKE && pwd -P)
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Persistent path escapes repository containment: ${actual}" >&2
        return 1
    fi
}

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

prepare_persistent_base
matrix_root="saves/exp/UAM_SMOKE/${timestamp}"
manifest="${matrix_root}/RUN_MANIFEST.tsv"
if [[ -e "${matrix_root}" || -L "${matrix_root}" ]]; then
    echo "Matrix root already exists: ${matrix_root}" >&2
    exit 1
fi
mkdir "${matrix_root}"
printf 'pid\tgpu\tmethod\tstart_utc\tend_utc\texit_code\tcommand\n' > "${manifest}"

arms=(uam_nll uam_simnpo uam_gu_nll uam_gu_simnpo)

declare -a active_pid
active_pid[0]=""
active_pid[1]=""
declare -a active_method=("" "")
declare -a active_start=("" "")
declare -a active_command=("" "")
failure=0
cleanup_in_progress=0

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
    record_completion "${slot}" "${pid}" "${exit_code}"
}

pid_is_running() {
    local pid=$1
    [[ -r "/proc/${pid}/stat" ]] || return 1
    [[ "$(awk '{print $3}' "/proc/${pid}/stat")" != Z ]]
}

terminate_active_arms() {
    if [[ ${cleanup_in_progress} -ne 0 ]]; then
        return 0
    fi
    cleanup_in_progress=1

    local slot pid
    for slot in 0 1; do
        pid=${active_pid[slot]}
        if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    local deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )); do
        local running=0
        for slot in 0 1; do
            pid=${active_pid[slot]}
            if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
                running=1
            fi
        done
        [[ ${running} -ne 0 ]] || break
        sleep 0.05
    done

    for slot in 0 1; do
        pid=${active_pid[slot]}
        if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    for slot in 0 1; do
        wait_for_slot "${slot}"
    done
    cleanup_in_progress=0
}

handle_signal() {
    local exit_code=$1
    trap - INT TERM HUP
    failure=1
    terminate_active_arms
    exit "${exit_code}"
}

handle_exit() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        terminate_active_arms
    fi
    return "${exit_code}"
}

trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap 'handle_signal 129' HUP
trap handle_exit EXIT

for wave_start in 0 2; do
    launch_arm 0 "${arms[wave_start]}"
    launch_arm 1 "${arms[wave_start + 1]}"
    wait_for_slot 0
    wait_for_slot 1
    if [[ ${failure} -ne 0 ]]; then
        break
    fi
done

if [[ ${failure} -ne 0 ]]; then
    exit 1
fi
echo "Completed UAM smoke matrix: ${matrix_root}"
