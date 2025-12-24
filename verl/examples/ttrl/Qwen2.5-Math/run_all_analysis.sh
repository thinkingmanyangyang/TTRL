#!/bin/bash
# 完整流程：先推理，然后批量分析所有 LCS 配置组合


# 临时设置 PATH，优先使用指定环境的 Python（脚本内有效，不影响系统）
export PATH="/caobing/conda_envs/verl/bin:$PATH"

# ============================================================================
# 配置区域
# ============================================================================

# 推理配置
START_INDEX=0
END_INDEX=30
N_SAMPLES=32
TEMPERATURE=1.0
TOP_P=0.95
CACHE_DIR="inference_cache_1"
CACHE_NAME="responses"

# 分析配置
MAX_TOKENS_LIST=(3000 2048 1024)
NORMALIZE_MODES=("avg" "l1" "l2" "max" "min")
OUTPUT_DIR="inference_results_1_4"
NUM_WORKERS=32  # 并行进程数，建议设置为 CPU 核心数

# ============================================================================
# 步骤 1: 推理
# ============================================================================

CACHE_FILE="${CACHE_DIR}/${CACHE_NAME}.pkl"

echo "============================================================================"
echo "完整实验流程：推理 + 批量分析"
echo "============================================================================"
echo "步骤 1: 推理"
echo "  数据范围: [$START_INDEX, $END_INDEX)"
echo "  每题样本数: $N_SAMPLES"
echo "  缓存文件: $CACHE_FILE"
echo ""
echo "步骤 2: 批量分析"
echo "  Max Tokens: ${MAX_TOKENS_LIST[@]}"
echo "  Normalize Modes: ${NORMALIZE_MODES[@]}"
echo "  总分析数: $((${#MAX_TOKENS_LIST[@]} * ${#NORMALIZE_MODES[@]}))"
echo "  并行进程数: $NUM_WORKERS"
echo "============================================================================"
echo ""

# 询问是否跳过推理
if [ -f "$CACHE_FILE" ]; then
    echo "⚠️  缓存文件已存在: $CACHE_FILE"
    read -p "是否跳过推理，直接进行分析？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "============================================================================"
        echo "步骤 1: 运行推理"
        echo "============================================================================"
        
        python inference_only.py \
            --start-index $START_INDEX \
            --end-index $END_INDEX \
            --n-samples $N_SAMPLES \
            --temperature $TEMPERATURE \
            --top-p $TOP_P \
            --cache-dir $CACHE_DIR \
            --cache-name $CACHE_NAME
        
        if [ $? -ne 0 ]; then
            echo "❌ 推理失败！"
            exit 1
        fi
    else
        echo "✓ 跳过推理，使用现有缓存"
    fi
else
    echo "============================================================================"
    echo "步骤 1: 运行推理"
    echo "============================================================================"
    
    python inference_only.py \
        --start-index $START_INDEX \
        --end-index $END_INDEX \
        --n-samples $N_SAMPLES \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --cache-dir $CACHE_DIR \
        --cache-name $CACHE_NAME
    
    if [ $? -ne 0 ]; then
        echo "❌ 推理失败！"
        exit 1
    fi
fi

# 检查缓存文件
if [ ! -f "$CACHE_FILE" ]; then
    echo "❌ 缓存文件不存在: $CACHE_FILE"
    exit 1
fi

# ============================================================================
# 步骤 2: 批量分析
# ============================================================================

echo ""
echo "============================================================================"
echo "步骤 2: 批量分析（所有 LCS 配置组合）"
echo "============================================================================"
echo ""

EXP_NUM=0
TOTAL=$((${#MAX_TOKENS_LIST[@]} * ${#NORMALIZE_MODES[@]}))
START_TIME=$(date +%s)

# 运行所有分析
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
    for NORM_MODE in "${NORMALIZE_MODES[@]}"; do
        EXP_NUM=$((EXP_NUM + 1))
        
        echo "========================================================================"
        echo "分析 ${EXP_NUM}/${TOTAL}: max_tokens=${MAX_TOKENS}, norm_mode=${NORM_MODE}"
        echo "========================================================================"
        
        python analyze_cached.py \
            --cache-file "$CACHE_FILE" \
            --lcs-max-tokens ${MAX_TOKENS} \
            --lcs-normalize-mode ${NORM_MODE} \
            --output-dir ${OUTPUT_DIR} \
            --num-workers ${NUM_WORKERS}
        
        if [ $? -eq 0 ]; then
            echo "✓ 完成"
        else
            echo "✗ 失败"
        fi
        echo ""
    done
done

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================================"
echo "所有分析完成！"
echo "============================================================================"
echo "总分析数: ${TOTAL}"
echo "耗时: ${ELAPSED} 秒"
echo "结果保存在: ${OUTPUT_DIR}/"
echo ""
echo "生成的文件:"
ls -lh ${OUTPUT_DIR}/lcs_viz_*.png 2>/dev/null | tail -5
echo "..."
echo "============================================================================"

