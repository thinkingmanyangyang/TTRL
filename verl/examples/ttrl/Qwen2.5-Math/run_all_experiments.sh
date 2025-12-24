#!/bin/bash
# 批量运行 LCS 相似度实验

# 实验配置
MAX_TOKENS_LIST=(2048 1024 800 600)
NORMALIZE_MODES=("avg" "l1" "l2" "max" "min")

# 固定参数
START_INDEX=0
END_INDEX=30
N_SAMPLES=32

echo "============================================================================"
echo "LCS 相似度消融实验 - 批量运行"
echo "============================================================================"
echo "Max Tokens: ${MAX_TOKENS_LIST[@]}"
echo "Normalize Modes: ${NORMALIZE_MODES[@]}"
echo "总实验数: $((${#MAX_TOKENS_LIST[@]} * ${#NORMALIZE_MODES[@]}))"
echo "============================================================================"
echo ""

EXP_NUM=0
TOTAL=$((${#MAX_TOKENS_LIST[@]} * ${#NORMALIZE_MODES[@]}))

# 运行所有实验
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
    for NORM_MODE in "${NORMALIZE_MODES[@]}"; do
        EXP_NUM=$((EXP_NUM + 1))
        
        echo ""
        echo "========================================================================"
        echo "实验 ${EXP_NUM}/${TOTAL}: max_tokens=${MAX_TOKENS}, norm_mode=${NORM_MODE}"
        echo "========================================================================"
        
        python simple_inference.py \
            --start-index ${START_INDEX} \
            --end-index ${END_INDEX} \
            --n-samples ${N_SAMPLES} \
            --lcs-max-tokens ${MAX_TOKENS} \
            --lcs-normalize-mode ${NORM_MODE}
        
        echo "✓ 完成"
        sleep 2
    done
done

echo ""
echo "============================================================================"
echo "所有实验完成！"
echo "结果保存在: inference_results/"
echo "============================================================================"

