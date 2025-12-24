#!/usr/bin/env python3
"""
简单的 FSDP 检查点转换脚本

使用方法:
    python convert_checkpoint.py <源检查点路径> <目标模型路径>
    python convert_checkpoint.py  # 使用默认路径
"""

import sys
import subprocess
from pathlib import Path

# 默认路径
DEFAULT_SOURCE = "/caobing/biomedical/TTRL/verl/checkpoints/TTRL-verl/AIME-TTT-Qwen2.5-Math-1.5B/1208/TTRL-Len@3k-grpo-none-b1-165958/global_step_240/actor"
DEFAULT_TARGET = "/caobing/biomedical/TTRL/verl/checkpoints/merged_models/AIME-TTT-Qwen2.5-Math-1.5B-step240"

# 获取参数
source_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
target_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TARGET

print("=" * 60)
print("转换 FSDP 检查点为 HuggingFace 格式")
print("=" * 60)
print(f"源检查点: {source_path}")
print(f"目标路径: {target_path}")
print()

# 创建目标目录
Path(target_path).mkdir(parents=True, exist_ok=True)

# 执行转换
cmd = [
    "python", "-m", "verl.model_merger", "merge",
    "--backend", "fsdp",
    "--local_dir", source_path,
    "--target_dir", target_path
]

subprocess.run(cmd, check=True)

print()
print("=" * 60)
print("转换完成！")
print(f"模型保存在: {target_path}")
print("=" * 60)

