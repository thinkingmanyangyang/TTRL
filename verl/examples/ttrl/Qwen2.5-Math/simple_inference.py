#!/usr/bin/env python3
"""
通过并发请求 vLLM API 进行批量推理（TTRL 风格）
- 答案抽取与简化
- 众数投票（Majority Voting）
- Token 级别 LCS 相似度计算

需要先启动 vLLM API 服务: bash deploy_vllm.sh
"""

import asyncio
import aiohttp
import time
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from transformers import AutoTokenizer

# 设置 matplotlib 后端（支持无显示器环境）
matplotlib.use('Agg')

# 添加 verl 路径
sys.path.insert(0, '/caobing/biomedical/TTRL/verl')
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade


# ============================================================================
# LCS 相似度计算函数（本地实现，避免包依赖）
# ============================================================================

def _lcs_python(seq1: List[int], seq2: List[int]) -> int:
    """
    纯 Python 实现的 LCS 长度计算（动态规划）
    时间复杂度: O(m*n), 空间复杂度: O(min(m,n))
    """
    if not seq1 or not seq2:
        return 0
    
    # 优化：让 seq1 是较短的序列
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    
    m, n = len(seq1), len(seq2)
    
    # 使用滚动数组优化空间
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[i] = prev[i-1] + 1
            else:
                curr[i] = max(curr[i-1], prev[i])
        prev, curr = curr, prev
    
    return prev[m]


def token_lcs_similarity(
    token_ids_1: List[int],
    token_ids_2: List[int],
    max_tokens: int = 1500,
    normalize: bool = True,
    normalize_mode: str = "l2"
) -> float:
    """
    基于 token IDs 的 LCS 相似度。
    
    Args:
        token_ids_1: 第一个序列的 token IDs
        token_ids_2: 第二个序列的 token IDs
        max_tokens: 最大 token 长度（防止过长序列导致性能问题）
        normalize: 是否归一化到 [0, 1]
        normalize_mode: 归一化模式
            - "avg": (len1 + len2) / 2  平均长度
            - "l1":  len1               第一个序列长度
            - "l2":  len2               第二个序列长度（参考序列）
            - "max": max(len1, len2)    最大长度
            - "min": min(len1, len2)    最小长度
    
    Returns:
        LCS 相似度分数 [0, 1] (normalize=True) 或原始 LCS 长度 (normalize=False)
    
    Examples:
        >>> ids1 = [100, 200, 300, 400]
        >>> ids2 = [100, 250, 300, 450]
        >>> token_lcs_similarity(ids1, ids2, normalize_mode="avg")
        0.5  # LCS=[100, 300], len=2, avg_len=4
        >>> token_lcs_similarity(ids1, ids2, normalize_mode="l2")
        0.5  # LCS=2, l2=4
    """
    if not token_ids_1 or not token_ids_2:
        return 0.0
    
    # 截断过长序列（性能优化）
    if len(token_ids_1) > max_tokens:
        token_ids_1 = token_ids_1[:max_tokens]
    if len(token_ids_2) > max_tokens:
        token_ids_2 = token_ids_2[:max_tokens]
    
    # 尝试使用 Numba 加速版本
    try:
        from verl.utils.reward_score.ttrl_math.utils_numba import lcs_length_numba, NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            # Numba 需要 NumPy 数组
            arr1 = np.array(token_ids_1, dtype=np.int32)
            arr2 = np.array(token_ids_2, dtype=np.int32)
            lcs_len = lcs_length_numba(arr1, arr2)
        else:
            lcs_len = _lcs_python(token_ids_1, token_ids_2)
    except (ImportError, Exception):
        # Numba 未安装或出错，使用纯 Python 版本
        lcs_len = _lcs_python(token_ids_1, token_ids_2)
    
    if not normalize:
        return float(lcs_len)
    
    # 根据归一化模式计算分母
    len1 = len(token_ids_1)
    len2 = len(token_ids_2)
    
    if normalize_mode == "avg":
        norm_len = (len1 + len2) / 2.0
    elif normalize_mode == "l1":
        norm_len = len1
    elif normalize_mode == "l2":
        norm_len = len2
    elif normalize_mode == "max":
        norm_len = max(len1, len2)
    elif normalize_mode == "min":
        norm_len = min(len1, len2)
    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}. "
                        f"Valid options: 'avg', 'l1', 'l2', 'max', 'min'")
    
    return lcs_len / norm_len if norm_len > 0 else 0.0




# API 配置
API_BASE_URL = "http://localhost:2333/v1"
MODEL_NAME = "math-model"

# 数据配置
DATA_PATH = "/caobing/biomedical/TTRL/verl/data/AIME-TTT/test.parquet"
START_INDEX = 0  # 起始索引
END_INDEX = 30    # 结束索引（不包含），None 表示到末尾
# 或者使用 NUM_SAMPLES 指定数量：从 START_INDEX 开始读取 NUM_SAMPLES 条
NUM_SAMPLES = None  # 如果设置了，END_INDEX 会被忽略

# 模型配置（用于 tokenizer）
CHECKPOINT_PATH = "/caobing/biomedical/TTRL/verl/checkpoints/merged_models/AIME-TTT-Qwen2.5-Math-1.5B-step240"

# 推理参数
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 3072
N_SAMPLES = 32  # 每个问题生成多少个答案

# LCS 相似度计算配置
LCS_MAX_TOKENS = 2048  # LCS 计算时的最大 token 长度（性能优化）
LCS_NORMALIZE_MODE = "avg"  # 归一化模式: "avg", "l1", "l2", "max", "min"
# - "avg": (len1 + len2) / 2  平均长度
# - "l1":  len1               第一个序列长度
# - "l2":  len2               第二个序列长度（参考序列）
# - "max": max(len1, len2)    最大长度
# - "min": min(len1, len2)    最小长度


async def call_api_single(session: aiohttp.ClientSession, question: str, index: int) -> Dict:
    """异步调用 API 单个请求"""
    
    url = f"{API_BASE_URL}/completions"
    
    # 构建请求数据
    payload = {
        "model": MODEL_NAME,
        "prompt": f"User: {question}\n\nAssistant:",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n": N_SAMPLES,  # 生成多个答案
    }
    
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return {
                "index": index,
                "question": question,
                "success": True,
                "result": result
            }
    except Exception as e:
        return {
            "index": index,
            "question": question,
            "success": False,
            "error": str(e)
        }


async def batch_inference(questions: List[str], max_concurrent: int = 10) -> List[Dict]:
    """批量并发推理"""
    
    print(f"开始批量推理 {len(questions)} 个问题...")
    print(f"并发数: {max_concurrent}, 每题生成 {N_SAMPLES} 个答案\n")
    
    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_call(session, question, index):
        async with semaphore:
            return await call_api_single(session, question, index)
    
    # 创建 HTTP 会话
    timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 创建所有任务
        tasks = [
            limited_call(session, question, i) 
            for i, question in enumerate(questions)
        ]
        
        # 并发执行
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        print(f"✓ 批量推理完成！耗时: {elapsed:.2f}s")
        print(f"  平均速度: {elapsed/len(questions):.2f}s/问题\n")
        
        return results


def majority_vote_with_lcs(
    responses: List[str],
    tokenizer: AutoTokenizer,
    lcs_max_tokens: int = LCS_MAX_TOKENS,
    lcs_normalize_mode: str = LCS_NORMALIZE_MODE
) -> Dict:
    """
    对多个回答进行众数投票，并计算 LCS 相似度
    
    Returns:
        Dict containing:
        - majority_answer: 众数答案
        - majority_count: 众数答案出现次数
        - majority_ratio: 众数答案占比
        - all_answers: 所有提取的答案列表
        - answer_distribution: 答案分布
        - lcs_similarities: 每个答案与众数的 LCS 相似度
        - avg_lcs_similarity: 平均 LCS 相似度
    """
    # 1. 抽取和简化答案
    extracted_answers = []
    valid_responses = []
    
    for response in responses:
        extracted = extract_answer(response)
        if extracted is not None:
            simplified = simplify_expression_string(extracted)
            extracted_answers.append(simplified)
            valid_responses.append(response)
        else:
            extracted_answers.append(None)
    
    # 统计有效答案
    valid_answers = [a for a in extracted_answers if a is not None]
    
    if len(valid_answers) == 0:
        return {
            "majority_answer": None,
            "majority_count": 0,
            "majority_ratio": 0.0,
            "all_answers": [],
            "answer_distribution": {},
            "lcs_similarities": [],
            "avg_lcs_similarity": 0.0,
            "valid_count": 0,
            "total_count": len(responses)
        }
    
    # 2. 众数投票
    answer_counter = Counter(valid_answers)
    majority_answer, majority_count = answer_counter.most_common(1)[0]
    majority_ratio = majority_count / len(responses)
    
    # 3. 区分正样本和负样本
    positive_indices = []  # 答案等于众数的样本索引
    negative_indices = []  # 答案不等于众数的样本索引
    
    for i, answer in enumerate(extracted_answers):
        if answer == majority_answer:
            positive_indices.append(i)
        elif answer is not None:  # 只考虑有效答案
            negative_indices.append(i)
    
    # 4. Tokenize 所有 responses（只 tokenize 一次）
    all_token_ids = []
    for response in responses:
        tokens = tokenizer.encode(response, add_special_tokens=False)
        all_token_ids.append(tokens)
    
    # 5. 计算正样本的 LCS（每个正样本与其他正样本的 max LCS，不包括自己）
    positive_lcs_list = []
    for pos_idx in positive_indices:
        max_similarity = 0.0
        for other_pos_idx in positive_indices:
            if pos_idx != other_pos_idx:  # 不计算自己
                similarity = token_lcs_similarity(
                    all_token_ids[pos_idx],
                    all_token_ids[other_pos_idx],
                    max_tokens=lcs_max_tokens,
                    normalize=True,
                    normalize_mode=lcs_normalize_mode
                )
                max_similarity = max(max_similarity, similarity)
        positive_lcs_list.append(max_similarity)
    
    # 6. 计算负样本的 LCS（每个负样本与所有正样本的 max LCS）
    negative_lcs_list = []
    for neg_idx in negative_indices:
        max_similarity = 0.0
        for pos_idx in positive_indices:
            similarity = token_lcs_similarity(
                all_token_ids[neg_idx],
                all_token_ids[pos_idx],
                max_tokens=lcs_max_tokens,
                normalize=True,
                normalize_mode=lcs_normalize_mode
            )
            max_similarity = max(max_similarity, similarity)
        negative_lcs_list.append(max_similarity)
    
    # 7. 计算所有样本的 LCS（用于显示）
    all_lcs_similarities = [0.0] * len(responses)
    for i, pos_idx in enumerate(positive_indices):
        all_lcs_similarities[pos_idx] = positive_lcs_list[i] if i < len(positive_lcs_list) else 0.0
    for i, neg_idx in enumerate(negative_indices):
        all_lcs_similarities[neg_idx] = negative_lcs_list[i] if i < len(negative_lcs_list) else 0.0
    
    # 8. 统计
    avg_positive_lcs = sum(positive_lcs_list) / len(positive_lcs_list) if positive_lcs_list else 0.0
    avg_negative_lcs = sum(negative_lcs_list) / len(negative_lcs_list) if negative_lcs_list else 0.0
    avg_all_lcs = sum(all_lcs_similarities) / len(responses) if responses else 0.0
    
    return {
        "majority_answer": majority_answer,
        "majority_count": majority_count,
        "majority_ratio": majority_ratio,
        "all_answers": extracted_answers,
        "answer_distribution": dict(answer_counter),
        # LCS 相似度
        "lcs_similarities": all_lcs_similarities,
        "positive_lcs_list": positive_lcs_list,
        "negative_lcs_list": negative_lcs_list,
        "avg_positive_lcs": avg_positive_lcs,
        "avg_negative_lcs": avg_negative_lcs,
        "avg_all_lcs": avg_all_lcs,
        # 样本索引
        "positive_indices": positive_indices,
        "negative_indices": negative_indices,
        # 统计
        "valid_count": len(valid_answers),
        "total_count": len(responses),
        "positive_count": len(positive_indices),
        "negative_count": len(negative_indices)
    }


def visualize_lcs_distribution(
    results: List[Dict], 
    output_dir: str = "inference_results",
    lcs_max_tokens: int = LCS_MAX_TOKENS,
    lcs_normalize_mode: str = LCS_NORMALIZE_MODE
):
    """
    可视化所有问题的 LCS 相似度分布
    
    Args:
        results: 推理结果列表
        output_dir: 输出目录
        lcs_max_tokens: LCS 最大 token 长度
        lcs_normalize_mode: LCS 归一化模式
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 收集所有问题的 LCS 数据
    all_positive_lcs = []
    all_negative_lcs = []
    
    for result in results:
        if not result.get("success") or "ttrl_analysis" not in result:
            continue
        
        ttrl = result["ttrl_analysis"]
        all_positive_lcs.extend(ttrl.get("positive_lcs_list", []))
        all_negative_lcs.extend(ttrl.get("negative_lcs_list", []))
    
    if not all_positive_lcs and not all_negative_lcs:
        print("⚠️  没有足够的数据进行可视化")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Token-level LCS Similarity Distribution', fontsize=16, fontweight='bold')
    
    # 1. 散点图
    ax1 = axes[0, 0]
    if all_positive_lcs:
        ax1.scatter(range(len(all_positive_lcs)), all_positive_lcs, 
                   alpha=0.6, s=50, c='green', label='Positive Samples', marker='o')
    if all_negative_lcs:
        ax1.scatter(range(len(all_positive_lcs), len(all_positive_lcs) + len(all_negative_lcs)), 
                   all_negative_lcs, alpha=0.6, s=50, c='red', label='Negative Samples', marker='x')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('LCS Similarity', fontsize=12)
    ax1.set_title('LCS Similarity by Sample', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 2. 箱线图
    ax2 = axes[0, 1]
    data_to_plot = []
    labels = []
    if all_positive_lcs:
        data_to_plot.append(all_positive_lcs)
        labels.append(f'Positive\n(n={len(all_positive_lcs)})')
    if all_negative_lcs:
        data_to_plot.append(all_negative_lcs)
        labels.append(f'Negative\n(n={len(all_negative_lcs)})')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    if len(bp['boxes']) >= 1 and all_positive_lcs:
        bp['boxes'][0].set_facecolor('lightgreen')
    if len(bp['boxes']) >= 2 and all_negative_lcs:
        bp['boxes'][-1].set_facecolor('lightcoral')
    
    ax2.set_ylabel('LCS Similarity', fontsize=12)
    ax2.set_title('LCS Distribution Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])
    
    # 3. 直方图对比
    ax3 = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    if all_positive_lcs:
        ax3.hist(all_positive_lcs, bins=bins, alpha=0.6, color='green', 
                label=f'Positive (μ={np.mean(all_positive_lcs):.3f})', edgecolor='black')
    if all_negative_lcs:
        ax3.hist(all_negative_lcs, bins=bins, alpha=0.6, color='red', 
                label=f'Negative (μ={np.mean(all_negative_lcs):.3f})', edgecolor='black')
    ax3.set_xlabel('LCS Similarity', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('LCS Similarity Histogram', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 统计摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "Statistical Summary\n" + "="*40 + "\n\n"
    
    if all_positive_lcs:
        stats_text += f"Positive Samples (n={len(all_positive_lcs)}):\n"
        stats_text += f"  Mean:   {np.mean(all_positive_lcs):.4f}\n"
        stats_text += f"  Median: {np.median(all_positive_lcs):.4f}\n"
        stats_text += f"  Std:    {np.std(all_positive_lcs):.4f}\n"
        stats_text += f"  Min:    {np.min(all_positive_lcs):.4f}\n"
        stats_text += f"  Max:    {np.max(all_positive_lcs):.4f}\n\n"
    
    if all_negative_lcs:
        stats_text += f"Negative Samples (n={len(all_negative_lcs)}):\n"
        stats_text += f"  Mean:   {np.mean(all_negative_lcs):.4f}\n"
        stats_text += f"  Median: {np.median(all_negative_lcs):.4f}\n"
        stats_text += f"  Std:    {np.std(all_negative_lcs):.4f}\n"
        stats_text += f"  Min:    {np.min(all_negative_lcs):.4f}\n"
        stats_text += f"  Max:    {np.max(all_negative_lcs):.4f}\n\n"
    
    if all_positive_lcs and all_negative_lcs:
        diff = np.mean(all_positive_lcs) - np.mean(all_negative_lcs)
        stats_text += f"Difference (Pos - Neg):\n"
        stats_text += f"  Mean Diff: {diff:.4f}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # 构建包含配置参数的文件名
    filename_base = f"lcs_viz_maxlen{lcs_max_tokens}_norm{lcs_normalize_mode}"
    
    # 保存图表
    output_path = Path(output_dir) / f"{filename_base}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 可视化图表已保存: {output_path}")
    
    # 也保存为 PDF
    pdf_path = Path(output_dir) / f"{filename_base}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"📊 PDF 版本已保存: {pdf_path}")
    
    plt.close()


def print_results(results: List[Dict], tokenizer: AutoTokenizer):
    """打印推理结果（包含 TTRL 分析）"""
    
    for result in results:
        index = result["index"]
        question = result["question"]
        
        print(f"\n{'='*80}")
        print(f"【问题 {index + 1}】")
        print(f"Q: {question}")
        print(f"{'='*80}")
        
        if not result["success"]:
            print(f"❌ 错误: {result['error']}")
            continue
        
        # 解析 API 结果
        api_result = result["result"]
        choices = api_result.get("choices", [])
        
        # 提取所有回答
        responses = [choice.get("text", "").strip() for choice in choices]
        
        # TTRL 分析：众数投票 + LCS 相似度（使用命令行参数）
        # 注意：这里需要从全局获取 args，或者通过函数传递
        import __main__
        if hasattr(__main__, 'args'):
            lcs_max_tokens = __main__.args.lcs_max_tokens
            lcs_normalize_mode = __main__.args.lcs_normalize_mode
        else:
            lcs_max_tokens = LCS_MAX_TOKENS
            lcs_normalize_mode = LCS_NORMALIZE_MODE
            
        ttrl_analysis = majority_vote_with_lcs(
            responses, 
            tokenizer,
            lcs_max_tokens=lcs_max_tokens,
            lcs_normalize_mode=lcs_normalize_mode
        )
        
        # 保存分析结果到 result 中（用于后续可视化）
        result["ttrl_analysis"] = ttrl_analysis
        
        # 显示 TTRL 分析结果
        print(f"\n📊 TTRL 分析:")
        print(f"  有效答案: {ttrl_analysis['valid_count']}/{ttrl_analysis['total_count']}")
        print(f"  众数答案: {ttrl_analysis['majority_answer']}")
        print(f"  众数占比: {ttrl_analysis['majority_ratio']:.2%} ({ttrl_analysis['majority_count']}/{ttrl_analysis['total_count']})")
        print(f"\n  样本分布:")
        print(f"    正样本（众数）: {ttrl_analysis['positive_count']} 个")
        print(f"    负样本（非众数）: {ttrl_analysis['negative_count']} 个")
        print(f"\n  Token-level LCS 相似度:")
        print(f"    正样本平均 LCS: {ttrl_analysis['avg_positive_lcs']:.4f} (正样本间相似度)")
        print(f"    负样本平均 LCS: {ttrl_analysis['avg_negative_lcs']:.4f} (负样本与正样本)")
        print(f"    整体平均 LCS: {ttrl_analysis['avg_all_lcs']:.4f}")
        
        # 显示答案分布
        if ttrl_analysis['answer_distribution']:
            print(f"\n  答案分布:")
            # 按数量排序显示
            sorted_answers = sorted(ttrl_analysis['answer_distribution'].items(), key=lambda x: x[1], reverse=True)
            for answer, count in sorted_answers:
                print(f"    {answer}: {count}次 ({count/ttrl_analysis['total_count']:.1%})")
        
        # 显示示例回答（正负样本各显示一些）
        print(f"\n📝 示例回答:")
        
        # 显示正样本示例（最多3个）
        pos_indices = ttrl_analysis['positive_indices'][:3]
        if pos_indices:
            print(f"\n  ✅ 正样本示例（众数答案）:")
            for idx in pos_indices:
                extracted = ttrl_analysis['all_answers'][idx]
                lcs_sim = ttrl_analysis['lcs_similarities'][idx]
                display_text = responses[idx][:150] + "..." if len(responses[idx]) > 150 else responses[idx]
                print(f"\n    [{idx+1}] LCS={lcs_sim:.4f} | 答案: {extracted}")
                print(f"    {display_text}")
        
        # 显示负样本示例（最多2个）
        neg_indices = ttrl_analysis['negative_indices'][:2]
        if neg_indices:
            print(f"\n  ❌ 负样本示例（非众数答案）:")
            for idx in neg_indices:
                extracted = ttrl_analysis['all_answers'][idx]
                lcs_sim = ttrl_analysis['lcs_similarities'][idx]
                display_text = responses[idx][:150] + "..." if len(responses[idx]) > 150 else responses[idx]
                print(f"\n    [{idx+1}] LCS={lcs_sim:.4f} | 答案: {extracted}")
                print(f"    {display_text}")
        
        print()


def load_questions_from_dataset(
    data_path: str, 
    start_index: int = 0, 
    end_index: int = None,
    num_samples: int = None
) -> List[str]:
    """
    从数据集中加载测试问题
    
    Args:
        data_path: 数据集路径
        start_index: 起始索引（包含）
        end_index: 结束索引（不包含），None 表示到末尾
        num_samples: 读取数量，如果设置则忽略 end_index
    
    Returns:
        问题列表
    """
    
    print(f"从数据集加载问题: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ 数据集文件不存在: {data_path}")
        # 返回默认测试问题
        return [
            "What is the sum of all positive integers $n$ such that $\\frac{n+18}{n}$ is an integer?",
            "Solve for x: 2x + 5 = 13",
            "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        ]
    
    # 读取 parquet 文件
    df = pd.read_parquet(data_path)
    total_size = len(df)
    print(f"✓ 数据集加载成功，共 {total_size} 条数据")
    
    # 计算实际的索引范围
    if num_samples is not None:
        actual_end = min(start_index + num_samples, total_size)
    elif end_index is not None:
        actual_end = min(end_index, total_size)
    else:
        actual_end = total_size
    
    actual_start = min(start_index, total_size)
    
    print(f"  读取范围: [{actual_start}, {actual_end}) (共 {actual_end - actual_start} 条)")
    
    # 获取问题列（可能是 'prompt' 或 'question' 等）
    if 'prompt' in df.columns:
        question_col = 'prompt'
    elif 'question' in df.columns:
        question_col = 'question'
    else:
        print(f"可用列: {df.columns.tolist()}")
        question_col = df.columns[0]
        print(f"使用第一列作为问题列: {question_col}")
    
    # 提取指定范围的问题
    questions = []
    for i, row in df.iloc[actual_start:actual_end].iterrows():
        question = row[question_col]
        # 如果是列表格式（chat template），提取最后一条用户消息
        if isinstance(question, list):
            for msg in reversed(question):
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    question = msg.get('content', '')
                    break
        questions.append(str(question))
    
    print(f"✓ 已加载 {len(questions)} 条问题 (索引 {actual_start}-{actual_end-1})\n")
    return questions


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="vLLM API 批量推理（TTRL 风格）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据配置
    parser.add_argument("--start-index", type=int, default=START_INDEX,
                       help="起始索引")
    parser.add_argument("--end-index", type=int, default=END_INDEX,
                       help="结束索引（不包含），None 表示到末尾")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                       help="读取数量，如果设置则忽略 end-index")
    
    # LCS 配置
    parser.add_argument("--lcs-max-tokens", type=int, default=LCS_MAX_TOKENS,
                       help="LCS 计算时的最大 token 长度")
    parser.add_argument("--lcs-normalize-mode", type=str, default=LCS_NORMALIZE_MODE,
                       choices=["avg", "l1", "l2", "max", "min"],
                       help="LCS 归一化模式")
    
    # 推理配置
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                       help="每个问题生成多少个答案")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                       help="采样温度")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                       help="Top-p 采样")
    parser.add_argument("--max-concurrent", type=int, default=20,
                       help="最大并发请求数")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str, default="inference_results",
                       help="输出目录")
    
    return parser.parse_args()


def main():
    """主函数"""
    
    # 解析命令行参数
    args = parse_args()
    
    # 加载 tokenizer（用于 token 级别 LCS 计算）
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    print("✓ Tokenizer 加载完成\n")
    
    # 从数据集加载问题
    test_questions = load_questions_from_dataset(
        DATA_PATH, 
        start_index=args.start_index,
        end_index=args.end_index,
        num_samples=args.num_samples
    )
    
    print("="*80)
    print("vLLM API 批量并发推理测试（TTRL 风格）")
    print("="*80)
    print(f"API 地址: {API_BASE_URL}")
    print(f"问题数量: {len(test_questions)}")
    print(f"每题样本数: {args.n_samples}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"\nLCS 配置:")
    print(f"  最大 token 长度: {args.lcs_max_tokens}")
    print(f"  归一化模式: {args.lcs_normalize_mode}")
    print("="*80 + "\n")
    
    # 检查 API 是否可用
    import requests
    try:
        response = requests.get(f"{API_BASE_URL.replace('/v1', '')}/health", timeout=5)
        if response.status_code != 200:
            print("❌ vLLM API 服务未启动或不可用！")
            print("   请先运行: bash deploy_vllm.sh")
            return
    except Exception as e:
        print("❌ 无法连接到 vLLM API 服务！")
        print(f"   错误: {e}")
        print("   请先运行: bash deploy_vllm.sh")
        return
    
    print("✓ API 服务连接成功\n")
    
    # 运行批量推理
    results = asyncio.run(batch_inference(test_questions, max_concurrent=args.max_concurrent))
    
    # 打印结果（带 TTRL 分析）
    print_results(results, tokenizer)
    
    # 可视化 LCS 相似度分布
    print("\n" + "="*80)
    print("生成可视化图表...")
    print("="*80)
    visualize_lcs_distribution(
        results, 
        output_dir=args.output_dir,
        lcs_max_tokens=args.lcs_max_tokens,
        lcs_normalize_mode=args.lcs_normalize_mode
    )
    
    # 统计
    success_count = sum(1 for r in results if r["success"])
    print("\n" + "="*80)
    print(f"推理完成！成功: {success_count}/{len(results)}")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    # 保存 args 到 __main__ 模块，供其他函数访问
    import __main__
    __main__.args = args
    main()

