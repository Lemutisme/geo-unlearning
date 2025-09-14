#!/bin/bash

# ==============================================================================
# Comprehensive Evaluation Script for GeometricUnlearn Method
# ==============================================================================

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export CUDA_VISIBLE_DEVICES=3,4

per_device_train_batch_size=4
gradient_accumulation_steps=4
NUM_GPUS=2
EVAL_GPU=3

###################################################################################################
# TOFU Benchmark Evaluation
###################################################################################################
echo "================================================="
echo "Starting GeometricUnlearn on TOFU Benchmark"
echo "================================================="

tofu_models=(
    "Llama-3.2-1B-Instruct"
    "Llama-2-7b-hf"
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

        task_name=tofu_${model}_${forget_split}_GeometricUnlearn
        # TOFU 基准的目标模型是在完整数据集上微调过的
        model_path=open-unlearning/tofu_${model}_full

        echo "--- Running TOFU Task: ${task_name} ---"
        echo "Model: ${model_path}, Forget Split: ${forget_split}"

        # 步骤 1: 使用您的 GeometricUnlearn 方法进行遗忘训练
        accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=GeometricUnlearn \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        retain_logs_path=saves/eval/GU/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

        # 步骤 2: 评估刚刚经过遗忘训练的模型
        # 注意: 评估通常在单个 GPU 上运行以简化流程
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/tofu/default.yaml \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        retain_logs_path=saves/eval/GU/tofu_${model}_${retain_split}/TOFU_EVAL.json
    done
done


###################################################################################################
# MUSE Benchmark Evaluation
###################################################################################################
echo "================================================="
echo "Starting GeometricUnlearn on MUSE Benchmark"
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
        
        task_name=muse_${model}_${data_split}_GeometricUnlearn
        # MUSE 基准的目标模型直接从 muse-bench 加载
        model_path=muse-bench/MUSE-${data_split}_target

        echo "--- Running MUSE Task: ${task_name} ---"
        echo "Model: ${model_path}, Data Split: ${data_split}"

        # 步骤 1: 遗忘训练
        accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default \
        trainer=GeometricUnlearn \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        data_split=${data_split} \
        retain_logs_path=saves/eval/GU/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
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
        retain_logs_path=saves/eval/GU/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
    done
done


###################################################################################################
# WMDP Benchmark Evaluation
###################################################################################################
echo "================================================="
echo "Starting GeometricUnlearn on WMDP Benchmark"
echo "================================================="

wmdp_data_splits=(
    "cyber"
    # "bio" # 如果需要，可以取消注释以运行生物领域的数据
)
wmdp_model="zephyr-7b-beta"

for data_split in "${wmdp_data_splits[@]}"; do

    task_name=wmdp_${wmdp_model}_${data_split}_GeometricUnlearn
    model_path=wmdp-bench/WMDP-${data_split}_target
    echo "--- Running WMDP Task: ${task_name} ---"
    echo "Model: ${wmdp_model}, Data Split: ${data_split}"

    # 步骤 1: 遗忘训练
    accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/wmdp/default \
    trainer=GeometricUnlearn \
    task_name=${task_name} \
    model=${wmdp_model} \
    data_split=${data_split} \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=16 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true

    # 步骤 2: 评估
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
    experiment=eval/wmdp/default.yaml \
    task_name=${task_name} \
    model=${wmdp_model} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
    data_split=${data_split}
done

echo "================================================="
echo "All benchmarks completed!"
echo "================================================="