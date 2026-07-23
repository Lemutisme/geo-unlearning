#!/bin/bash

# ==============================================================================
# Comprehensive Evaluation Script for Common GU with Various Losses
# ==============================================================================

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export CUDA_VISIBLE_DEVICES=6


per_device_train_batch_size=4
gradient_accumulation_steps=4
NUM_GPUS=1
EVAL_GPU=4

gu_override='+trainer.method_args.gu={\
enabled:true,\
parameter_regex:["lm_head[.]weight"],\
retain_history_rank:8,\
projection_eps:1e-6,\
retain_filter:first_order,\
retain_budget:1e-4,\
backtracking_scales:[1.0,0.5,0.25,0.125],\
diagnostics_path:gu_diagnostics.jsonl}'

resolve_trainer_config() {
    case "$1" in
        simnpo) trainer_config=SimNPO ;;
        npo) trainer_config=NPO ;;
        dpo) trainer_config=DPO ;;
        undial) trainer_config=UNDIAL ;;
        ceu) trainer_config=CEU ;;
        wga) trainer_config=WGA ;;
        satimp) trainer_config=SatImp ;;
        *)
            echo "Unknown loss function: $1" >&2
            return 1
            ;;
    esac
}

EVAL_DIR="saves/exp/GU/$(date +%m%d%H%M)"

LOSS_FUNCTIONS=(
    "ceu"
    "dpo"
    "simnpo"
    "npo"
    "undial"
    "wga"
    "satimp"
)

mkdir -p ${EVAL_DIR}
echo "EVAL SAVED IN ${EVAL_DIR}"
###################################################################################################
# TOFU Benchmark Evaluation
###################################################################################################
echo "================================================="
echo "Starting common GU on TOFU Benchmark"
echo "================================================="

tofu_models=(
    "Llama-3.1-8B-Instruct"
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
)
tofu_splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

for loss_func in "${LOSS_FUNCTIONS[@]}"; do

    resolve_trainer_config "${loss_func}"
    METHOD_NAME=${trainer_config}
    for model in "${tofu_models[@]}"; do
        for split in "${tofu_splits[@]}"; do

            forget_split=$(echo $split | cut -d' ' -f1)
            holdout_split=$(echo $split | cut -d' ' -f2)
            retain_split=$(echo $split | cut -d' ' -f3)

            task_name=tofu_${model}_${forget_split}_GU_${METHOD_NAME}
            model_path=open-unlearning/tofu_${model}_full

            echo "--- Running TOFU Task: ${task_name} ---"
            echo "Model: ${model_path}, Forget Split: ${forget_split}, Loss: ${loss_func}"

            if [ "$loss_func" = "dpo" ] || [ "$loss_func" = "altpo" ]; then
                experiment_config="unlearn/tofu/idk"
            else
                experiment_config="unlearn/tofu/default"
            fi

            accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment_config} \
            "trainer=${trainer_config}" \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            +trainer.args.gradient_checkpointing_kwargs.use_reentrant=false \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            trainer.args.eval_strategy=no \
            "${gu_override}"

            CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
            experiment=eval/tofu/default.yaml \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            paths.output_dir=${EVAL_DIR}/${task_name} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
        done
    done
done

###################################################################################################
# MUSE Benchmark Evaluation
###################################################################################################

for loss_func in "${LOSS_FUNCTIONS[@]}"; do
    echo "================================================="
    echo "Starting ${loss_func} on MUSE Benchmark"
    echo "================================================="

    muse_models=(
        "Llama-2-7b-hf"
    )
    muse_data_splits=(
        "News"
        "Books"
    )

    resolve_trainer_config "${loss_func}"
    METHOD_NAME=${trainer_config}

    for model in "${muse_models[@]}"; do
        for data_split in "${muse_data_splits[@]}"; do

            task_name=muse_${model}_${data_split}_GU_${METHOD_NAME}
            model_path=muse-bench/MUSE-${data_split}_target

            echo "--- Running MUSE Task: ${task_name} ---"
            echo "Model: ${model_path}, Data Split: ${data_split}"

            accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/default \
            "trainer=${trainer_config}" \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            data_split=${data_split} \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            trainer.args.per_device_train_batch_size=2 \
            trainer.args.gradient_accumulation_steps=8 \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            trainer.args.eval_strategy=no \
            "${gu_override}"

            CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
            experiment=eval/muse/default.yaml \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            data_split=${data_split} \
            paths.output_dir=${EVAL_DIR}/${task_name} \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
        done
    done
done

###################################################################################################
# WMDP Benchmark Evaluation
###################################################################################################
for loss_func in "${LOSS_FUNCTIONS[@]}"; do
    echo "================================================="
    echo "Starting ${loss_func} on WMDP Benchmark"
    echo "================================================="
    resolve_trainer_config "${loss_func}"
    METHOD_NAME=${trainer_config}

    wmdp_data_splits=(
        "cyber"
    )
    wmdp_model="zephyr-7b-beta"

    for data_split in "${wmdp_data_splits[@]}"; do
        task_name=wmdp_${wmdp_model}_${data_split}_GU_${METHOD_NAME}
        model_path=wmdp-bench/WMDP-${data_split}_target

        echo "--- Running WMDP Task: ${task_name} ---"
        echo "Model: ${wmdp_model}, Data Split: ${data_split}"


        accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/wmdp/default \
        "trainer=${trainer_config}" \
        task_name=${task_name} \
        model=${wmdp_model} \
        data_split=${data_split} \
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.eval_strategy=no \
        "${gu_override}"

        CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/wmdp/default.yaml \
        task_name=${task_name} \
        model=${wmdp_model} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        data_split=${data_split} \
        paths.output_dir=${EVAL_DIR}/${task_name}
    done
done

echo "================================================="
echo "All benchmarks completed!"
echo "================================================="
