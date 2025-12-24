#!/bin/bash
# 简单的 FSDP 检查点转换脚本
# bash convert_checkpoint.sh <源检查点> <目标路径>

# 源检查点路径
SOURCE_CHECKPOINT="$1"

# 目标路径
TARGET_MODEL_DIR="$2"

# 如果没有提供参数，使用默认值
if [ -z "$SOURCE_CHECKPOINT" ]; then
    SOURCE_CHECKPOINT="/caobing/biomedical/TTRL/verl/checkpoints/TTRL-verl/AIME-TTT-Qwen2.5-Math-1.5B/1208/TTRL-Len@3k-grpo-none-b1-165958/global_step_240/actor"
fi

if [ -z "$TARGET_MODEL_DIR" ]; then
    TARGET_MODEL_DIR="/caobing/biomedical/TTRL/verl/checkpoints/merged_models/AIME-TTT-Qwen2.5-Math-1.5B-step240"
fi

echo "=========================================="
echo "转换 FSDP 检查点为 HuggingFace 格式"
echo "=========================================="
echo "源检查点: $SOURCE_CHECKPOINT"
echo "目标路径: $TARGET_MODEL_DIR"
echo ""

cd /caobing/biomedical/TTRL/verl

conda run -n verl python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_CHECKPOINT" \
    --target_dir "$TARGET_MODEL_DIR"

echo ""
echo "=========================================="
echo "转换完成！"
echo "模型保存在: $TARGET_MODEL_DIR"
echo "=========================================="

