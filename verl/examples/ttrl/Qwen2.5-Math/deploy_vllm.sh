#!/bin/bash
# 使用 vLLM 部署模型为 OpenAI 兼容 API 服务

# 临时设置 PATH，优先使用指定环境的 Python（脚本内有效，不影响系统）
export PATH="/caobing/conda_envs/evo_new/bin:$PATH"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 模型路径
MODEL_PATH="/caobing/biomedical/TTRL/verl/checkpoints/merged_models/AIME-TTT-Qwen2.5-Math-1.5B-step240"

# 服务配置
HOST="0.0.0.0"
PORT="2333"
MODEL_NAME="math-model"  # API 中使用的模型名称
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.7
MAX_MODEL_LEN=4096

echo "=========================================="
echo "启动 vLLM OpenAI 兼容 API 服务"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "服务地址: http://$HOST:$PORT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code

