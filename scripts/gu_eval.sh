#!/bin/bash

# ==============================================================================
# Comprehensive Evaluation Script for GeometricUnlearn Method with Various Losses
# ==============================================================================

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export CUDA_VISIBLE_DEVICES=6,7

# --- 可配置参数 ---
per_device_train_batch_size=4
gradient_accumulation_steps=4
NUM_GPUS=2
EVAL_GPU=4

EVAL_DIR="saves/exp/GU_NLL_gamma15" #test/$(date +%m%d%H%M)"
# --- 新增：定义要测试的 Geometric Unlearn 内部损失函数 ---
# 您可以在这里添加或删除损失函数，例如 "ce", "simnpo", "dpo" 等
LOSS_FUNCTIONS=(
    "graddiff"
    "ceu"
    "dpo"
    "simnpo"
    "gradascent"
    "npo"
    "undial"
    "wga"
    "satimp"
)
rt="NLL"  # 可选: "NLL" 或 "KL"

mkdir -p ${EVAL_DIR}
echo "EVAL SAVED IN ${EVAL_DIR}"
###################################################################################################
# TOFU Benchmark Evaluation
###################################################################################################
echo "================================================="
echo "Starting GeometricUnlearn on TOFU Benchmark"
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

# --- 修改：在原有循环内增加对 LOSS_FUNCTIONS 的循环 ---
for loss_func in "${LOSS_FUNCTIONS[@]}"; do

    METHOD_NAME=""
        case "$loss_func" in
            "graddiff")
                METHOD_NAME="GradDiff"
                ;;
            "ceu")
                METHOD_NAME="CEU"
                ;;
            "npo")
                METHOD_NAME="NPO"
                ;;
            "simnpo")
                METHOD_NAME="SimNPO"
                ;;
            "dpo")
                METHOD_NAME="DPO"
                ;;
            "undial")
                METHOD_NAME="UNDIAL"
                ;;
            "wga")
                METHOD_NAME="WGA"
                ;;
            "satimp")
                METHOD_NAME="SatImp"
                ;;
            "gradascent")
                METHOD_NAME="GradAscent"
                ;;
            *)
                # 如果有未知的 loss_func，可以设置一个默认名称或报错
                echo "未知的 loss function: $loss_func"
                continue
                ;;
        esac
    for model in "${tofu_models[@]}"; do
        for split in "${tofu_splits[@]}"; do
        # --- 新增：根据 loss_func 设置对应的任务名称 ---
            

            forget_split=$(echo $split | cut -d' ' -f1)
            holdout_split=$(echo $split | cut -d' ' -f2)
            retain_split=$(echo $split | cut -d' ' -f3)

            # --- 修改：动态构建 task_name，加入损失函数信息 ---
            task_name=tofu_${model}_${forget_split}_GU_${METHOD_NAME}
            model_path=open-unlearning/tofu_${model}_full

            echo "--- Running TOFU Task: ${task_name} ---"
            echo "Model: ${model_path}, Forget Split: ${forget_split}, Loss: ${loss_func}"
            
            # --- 新增：根据损失类型选择 experiment 配置文件 (处理DPO等特殊情况) ---
            if [ "$loss_func" = "dpo" ] || [ "$loss_func" = "altpo" ]; then
                experiment_config="unlearn/tofu/idk"
            else
                experiment_config="unlearn/tofu/default"
            fi

            # 步骤 1: 使用您的 GeometricUnlearn 方法进行遗忘训练
            accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUS \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment_config} \
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
            trainer.args.gradient_checkpointing=true \
            +trainer.args.gradient_checkpointing_kwargs.use_reentrant=false \
            trainer.method_args.geometric_config.loss=${loss_func} \
            trainer.method_args.retain_loss_type=${rt}

            # 步骤 2: 评估刚刚经过遗忘训练的模型
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

    # --- 新增：根据 loss_func 设置对应的任务名称 ---
    METHOD_NAME=""
    case "$loss_func" in
        "graddiff")
            METHOD_NAME="GradDiff"
            ;;
        "ceu")
            METHOD_NAME="CEU"
            ;;
        "npo")
            METHOD_NAME="NPO"
            ;;
        "simnpo")
            METHOD_NAME="SimNPO"
            ;;
        "dpo")
            METHOD_NAME="DPO"
            ;;
        "undial")
            METHOD_NAME="UNDIAL"
            ;;
        "wga")
            METHOD_NAME="WGA"
            ;;
        "satimp")
            METHOD_NAME="SatImp"
            ;;
        *)
            # 如果有未知的 loss_func，可以设置一个默认名称或报错
            echo "未知的 loss function: $loss_func"
            continue
            ;;
    esac

    for model in "${muse_models[@]}"; do
        for data_split in "${muse_data_splits[@]}"; do

            task_name=muse_${model}_${data_split}_GU_${METHOD_NAME}
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
            retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
            trainer.args.per_device_train_batch_size=2 \
            trainer.args.gradient_accumulation_steps=8 \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.method_args.geometric_config.loss=${loss_func} \
            trainer.method_args.retain_loss_type=${rt}

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
for loss_func in "${LOSS_FUNCTIONS[@]}"; do
    echo "================================================="
    echo "Starting ${method} on WMDP Benchmark"
    echo "================================================="
    # --- 新增：根据 loss_func 设置对应的任务名称 ---
    METHOD_NAME=""
    case "$loss_func" in
        "graddiff")
            METHOD_NAME="GradDiff"
            ;;
        "ceu")
            METHOD_NAME="CEU"
            ;;
        "npo")
            METHOD_NAME="NPO"
            ;;
        "simnpo")
            METHOD_NAME="SimNPO"
            ;;
        "dpo")
            METHOD_NAME="DPO"
            ;;
        "undial")
            METHOD_NAME="UNDIAL"
            ;;
        "wga")
            METHOD_NAME="WGA"
            ;;
        "satimp")
            METHOD_NAME="SatImp"
            ;;
        *)
            # 如果有未知的 loss_func，可以设置一个默认名称或报错
            echo "未知的 loss function: $loss_func"
            continue
            ;;
    esac

    wmdp_data_splits=(
        "cyber"
    )
    wmdp_model="zephyr-7b-beta"

    for data_split in "${wmdp_data_splits[@]}"; do
        task_name=wmdp_${wmdp_model}_${data_split}_GU_${METHOD_NAME}
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
        trainer.args.per_device_train_batch_size=2 \
        trainer.args.gradient_accumulation_steps=8 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.method_args.geometric_config.loss=${loss_func} \
        trainer.method_args.retain_loss_type=${rt}

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