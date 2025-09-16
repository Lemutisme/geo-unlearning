#!/bin/bash

# ==============================================================================
# Comprehensive Evaluation Script for Unlearning Methods
# ==============================================================================

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export CUDA_VISIBLE_DEVICES=4,6

# --- Configuration ---
EVAL_DIR="saves/exp/Eval_base" # $(date +%m%d%H%M)" 
UNLEARN_METHODS=(
    "GradAscent"
    "GradDiff"
    # "GeometricUnlearn"
    "NPO"
    "SimNPO"
    "DPO"
    "RMU"
    "UNDIAL"
    "CEU"
    "SatImp"
    "WGA"
    "PDU"
)

per_device_train_batch_size=4
gradient_accumulation_steps=4
NUM_GPUS=2
EVAL_GPU=4

# 创建评估目录，如果它不存在的话
mkdir -p ${EVAL_DIR}

###################################################################################################
# TOFU Benchmark Evaluation
###################################################################################################
for method in "${UNLEARN_METHODS[@]}"; do
    echo "================================================="
    echo "Starting ${method} on TOFU Benchmark"
    echo "================================================="

    tofu_models=(
        "Llama-3.2-1B-Instruct"
        "Llama-3.2-3B-Instruct"
        "Llama-3.1-8B-Instruct"
    )

    tofu_splits=(
        "forget10 holdout10 retain90"
        "forget05 holdout05 retain95"
        "forget01 holdout01 retain99"
    )

    for model in "${tofu_models[@]}"; do
        for split in "${tofu_splits[@]}"; do
            forget_split=$(echo $split | cut -d' ' -f1)
            holdout_split=$(echo $split | cut -d' ' -f2)
            retain_split=$(echo $split | cut -d' ' -f3)

            task_name=tofu_${model}_${forget_split}_${method}
            model_path=open-unlearning/tofu_${model}_full

            echo "--- Running TOFU Task: ${task_name} ---"
            echo "Model: ${model_path}, Forget Split: ${forget_split}"

            # Unlearn
            accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=${method} \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true

            # 步骤 2: 评估
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
for method in "${UNLEARN_METHODS[@]}"; do
    echo "================================================="
    echo "Starting ${method} on MUSE Benchmark"
    echo "================================================="

    muse_models=(
        "Llama-2-7b-hf"
    )
    muse_data_splits=(
        "News"
        "Books"
    )

    for model in "${muse_models[@]}"; do
        for data_split in "${muse_data_splits[@]}"; do

            task_name=muse_${model}_${data_split}_${method}
            model_path=muse-bench/MUSE-${data_split}_target

            per_device_train_batch_size=2
            gradient_accumulation_steps=8

            echo "--- Running MUSE Task: ${task_name} ---"
            echo "Model: ${model_path}, Data Split: ${data_split}"

            # 步骤 1: 遗忘训练
            accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
            src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/muse/default \
            trainer=${method} \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            data_split=${data_split} \
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true

            # 步骤 2: 评估
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
for method in "${UNLEARN_METHODS[@]}"; do
    echo "================================================="
    echo "Starting ${method} on WMDP Benchmark"
    echo "================================================="

    wmdp_data_splits=(
        "cyber"
        # "bio" 
    )
    wmdp_model="zephyr-7b-beta"

    for data_split in "${wmdp_data_splits[@]}"; do
        task_name=wmdp_${wmdp_model}_${data_split}_${method}
        model_path=wmdp-bench/WMDP-${data_split}_target

        echo "--- Running WMDP Task: ${task_name} ---"
        echo "Model: ${wmdp_model}, Data Split: ${data_split}"

        # 步骤 1: 遗忘训练
        accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/wmdp/default \
        trainer=${method} \
        task_name=${task_name} \
        model=${wmdp_model} \
        data_split=${data_split} \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.gradient_accumulation_steps=16 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        ~trainer.method_args.steering_coeff \
        ~trainer.method_args.module_regex \
        ~trainer.method_args.trainable_params_regex \
        trainer.method_args.retain_loss_type=NLL

        # 步骤 2: 评估
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