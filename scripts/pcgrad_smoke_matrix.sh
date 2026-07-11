#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <timestamp>" >&2
    exit 2
fi

timestamp=$1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unlearning

matrix_root="saves/exp/PCGRAD_SMOKE/${timestamp}"
manifest="${matrix_root}/RUN_MANIFEST.tsv"
mkdir -p "${matrix_root}"
printf 'pid\tgpu\tdataset\tmethod\tsystem_mode\tstart_utc\tend_utc\texit_code\tcommand\n' > "${manifest}"

datasets=(tofu01 tofu05 tofu10 muse_news muse_books)
methods=(control gu pcgrad)
arms=()
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        arms+=("${dataset}:${method}:production")
    done
done
arms+=(
    "tofu01:pcgrad:torch_flash"
    "tofu01:pcgrad:torch_eager_fp32"
)

declare -a active_pid
active_pid[0]=""
active_pid[1]=""
declare -a active_dataset=("" "")
declare -a active_method=("" "")
declare -a active_system=("" "")
declare -a active_start=("" "")
declare -a active_command=("" "")
failure=0

launch_arm() {
    local slot=$1
    local specification=$2
    local dataset method system_mode
    IFS=: read -r dataset method system_mode <<< "${specification}"
    local command="bash scripts/pcgrad_smoke_arm.sh ${dataset} ${method} ${slot} ${timestamp} ${system_mode}"
    local start_utc
    start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    bash scripts/pcgrad_smoke_arm.sh \
        "${dataset}" "${method}" "${slot}" "${timestamp}" "${system_mode}" &
    active_pid[slot]=$!
    active_dataset[slot]=${dataset}
    active_method[slot]=${method}
    active_system[slot]=${system_mode}
    active_start[slot]=${start_utc}
    active_command[slot]=${command}
    echo "Started pid=${active_pid[slot]} gpu=${slot} arm=${specification}"
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${pid}" "${slot}" "${active_dataset[slot]}" \
        "${active_method[slot]}" "${active_system[slot]}" \
        "${active_start[slot]}" "${end_utc}" "${exit_code}" \
        "${active_command[slot]}" >> "${manifest}"
    if [[ ${exit_code} -ne 0 ]]; then
        failure=1
        echo "Arm failed pid=${pid} exit_code=${exit_code}; no new arms will be scheduled." >&2
    fi
    active_pid[slot]=""
}

next_slot=0
for arm in "${arms[@]}"; do
    if [[ ${failure} -ne 0 ]]; then
        break
    fi
    if [[ -n "${active_pid[next_slot]}" ]]; then
        wait_for_slot "${next_slot}"
        if [[ ${failure} -ne 0 ]]; then
            break
        fi
    fi
    launch_arm "${next_slot}" "${arm}"
    next_slot=$((1 - next_slot))
done

for slot in 0 1; do
    wait_for_slot "${slot}"
done

if [[ ${failure} -ne 0 ]]; then
    exit 1
fi
echo "Completed PCGrad smoke matrix: ${matrix_root}"
