#!/usr/bin/env python3
"""
步骤2: 分析缓存的推理结果
使用不同的 LCS 配置对已保存的推理结果进行分析和可视化
"""

import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
matplotlib.use('Agg')

sys.path.insert(0, '/caobing/biomedical/TTRL/verl')
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string

# 检查 numba 是否可用
try:
    from verl.utils.reward_score.ttrl_math.utils_numba import NUMBA_AVAILABLE
    if NUMBA_AVAILABLE:
        print("✓ Numba LCS 加速可用")
    else:
        print("⚠️  Numba 不可用，将使用纯 Python LCS 实现")
except ImportError:
    print("⚠️  无法导入 Numba 模块，将使用纯 Python LCS 实现")
    NUMBA_AVAILABLE = False


# 模型配置
CHECKPOINT_PATH = "/caobing/biomedical/TTRL/verl/checkpoints/merged_models/AIME-TTT-Qwen2.5-Math-1.5B-step240"


def _lcs_python(seq1: List[int], seq2: List[int]) -> int:
    """纯 Python LCS 实现"""
    if not seq1 or not seq2:
        return 0
    
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    
    m, n = len(seq1), len(seq2)
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


def token_lcs_similarity(token_ids_1: List[int], token_ids_2: List[int],
                        max_tokens: int, normalize_mode: str) -> float:
    """计算 token LCS 相似度"""
    if not token_ids_1 or not token_ids_2:
        return 0.0
    
    if len(token_ids_1) > max_tokens:
        token_ids_1 = token_ids_1[:max_tokens]
    if len(token_ids_2) > max_tokens:
        token_ids_2 = token_ids_2[:max_tokens]
    
    try:
        from verl.utils.reward_score.ttrl_math.utils_numba import lcs_length_numba, NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            arr1 = np.array(token_ids_1, dtype=np.int32)
            arr2 = np.array(token_ids_2, dtype=np.int32)
            lcs_len = lcs_length_numba(arr1, arr2)
        else:
            lcs_len = _lcs_python(token_ids_1, token_ids_2)
    except:
        lcs_len = _lcs_python(token_ids_1, token_ids_2)
    
    len1, len2 = len(token_ids_1), len(token_ids_2)
    
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
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")
    
    return lcs_len / norm_len if norm_len > 0 else 0.0


def analyze_with_lcs(responses: List[str], tokenizer_path: str, lcs_max_tokens: int, lcs_normalize_mode: str, temperature: float = 0.1) -> Dict:
    """对responses进行TTRL分析（可并行调用）"""
    
    # 在子进程中加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 1. 抽取答案
    extracted_answers = []
    for response in responses:
        extracted = extract_answer(response)
        if extracted is not None:
            simplified = simplify_expression_string(extracted)
            extracted_answers.append(simplified)
        else:
            extracted_answers.append(None)
    
    valid_answers = [a for a in extracted_answers if a is not None]
    
    if len(valid_answers) == 0:
        return {
            "majority_answer": None,
            "positive_lcs_list": [],
            "negative_lcs_list": [],
            "avg_positive_lcs": 0.0,
            "avg_negative_lcs": 0.0,
            "positive_count": 0,
            "negative_count": 0
        }
    
    # 2. 众数投票
    answer_counter = Counter(valid_answers)
    majority_answer, majority_count = answer_counter.most_common(1)[0]
    
    # 3. 区分正负样本
    positive_indices = []
    negative_indices = []
    
    for i, answer in enumerate(extracted_answers):
        if answer == majority_answer:
            positive_indices.append(i)
        elif answer is not None:
            negative_indices.append(i)
    
    # 4. Tokenize
    all_token_ids = []
    for response in responses:
        tokens = tokenizer.encode(response, add_special_tokens=False)
        all_token_ids.append(tokens)
    
    # 5. 预计算所有需要的相似度（避免重复计算）
    # 使用字典缓存相似度，利用对称性 sim(i,j) = sim(j,i)
    similarity_cache = {}
    
    def get_similarity(idx1, idx2):
        """获取缓存的相似度，避免重复计算"""
        if idx1 == idx2:
            return 1.0
        key = tuple(sorted([idx1, idx2]))  # 利用对称性
        if key not in similarity_cache:
            similarity_cache[key] = token_lcs_similarity(
                all_token_ids[idx1],
                all_token_ids[idx2],
                lcs_max_tokens,
                lcs_normalize_mode
            )
        return similarity_cache[key]
    
    # 6. 计算正样本的统计量
    positive_max_pos_lcs = []  # 正样本与其他正样本的max LCS
    positive_avg_neg_lcs = []  # 正样本与所有负样本的平均 LCS
    
    for pos_idx in positive_indices:
        # 与其他正样本的最大相似度
        max_pos_sim = 0.0
        for other_pos_idx in positive_indices:
            if pos_idx != other_pos_idx:
                max_pos_sim = max(max_pos_sim, get_similarity(pos_idx, other_pos_idx))
        positive_max_pos_lcs.append(max_pos_sim)
        
        # 与所有负样本的平均相似度
        if negative_indices:
            negative_indices = negative_indices[-5:]
            neg_sims = [get_similarity(pos_idx, neg_idx) for neg_idx in negative_indices]
            positive_avg_neg_lcs.append(sum(neg_sims) / len(neg_sims))
        else:
            positive_avg_neg_lcs.append(0.0)
    
    # 7. 计算负样本的统计量（复用已计算的相似度）
    negative_max_pos_lcs = []  # 负样本与正样本的max LCS
    negative_avg_neg_lcs = []  # 负样本与所有负样本的平均 LCS
    
    for neg_idx in negative_indices:
        # 与正样本的最大相似度（复用缓存）
        max_pos_sim = 0.0
        for pos_idx in positive_indices:
            max_pos_sim = max(max_pos_sim, get_similarity(neg_idx, pos_idx))
        negative_max_pos_lcs.append(max_pos_sim)
        
        # 与其他负样本的平均相似度
        if len(negative_indices) > 1:
            other_negative_indices = [n for n in negative_indices if n != neg_idx]
            other_negative_indices = other_negative_indices[-5:]
            neg_sims = [get_similarity(neg_idx, other_neg_idx) for other_neg_idx in other_negative_indices]
            negative_avg_neg_lcs.append(sum(neg_sims) / len(neg_sims) if neg_sims else 0.0)
        else:
            negative_avg_neg_lcs.append(0.0)   
    
    # 保持原有的统计变量（向后兼容）
    avg_positive_lcs = sum(positive_max_pos_lcs) / len(positive_max_pos_lcs) if positive_max_pos_lcs else 0.0
    avg_negative_lcs = sum(negative_max_pos_lcs) / len(negative_max_pos_lcs) if negative_max_pos_lcs else 0.0
    
    # 计算对比学习风格的相似度分数 (SimCLR/MoCo 风格)
    # 公式: contrastive_score = exp(max_sim_pos/τ) / [exp(max_sim_pos/τ) + exp(avg_sim_neg/τ)]
    # 其中 τ 是温度参数，控制区分度
    # max_sim_pos: 当前样本与正样本的最大相似度
    # avg_sim_neg: 当前样本与所有负样本的平均相似度
    
    # 正样本的对比分数
    positive_contrastive_scores = []
    for i, (max_pos_sim, avg_neg_sim) in enumerate(zip(positive_max_pos_lcs, positive_avg_neg_lcs)):
        if avg_neg_sim > 0:
            numerator = np.exp(max_pos_sim / temperature)
            denominator = numerator + np.exp(avg_neg_sim / temperature)
            contrastive_score = numerator / denominator
        else:
            # 如果没有负样本，分数为 1.0（完美匹配）
            contrastive_score = 1.0
        positive_contrastive_scores.append(contrastive_score)
    
    # 负样本的对比分数
    negative_contrastive_scores = []
    for i, (max_pos_sim, avg_neg_sim) in enumerate(zip(negative_max_pos_lcs, negative_avg_neg_lcs)):
        numerator = np.exp(max_pos_sim / temperature)
        if avg_neg_sim > 0:
            denominator = numerator + np.exp(avg_neg_sim / temperature)
            contrastive_score = numerator / denominator
        else:
            # 如果只有一个负样本，无法计算与其他负样本的平均
            contrastive_score = numerator / (numerator + 1.0)
        negative_contrastive_scores.append(contrastive_score)
    
    avg_positive_contrastive = sum(positive_contrastive_scores) / len(positive_contrastive_scores) if positive_contrastive_scores else 0.0
    avg_negative_contrastive = sum(negative_contrastive_scores) / len(negative_contrastive_scores) if negative_contrastive_scores else 0.0
    
    return {
        "majority_answer": majority_answer,
        "positive_lcs_list": positive_max_pos_lcs,  # 正样本与正样本的max LCS
        "negative_lcs_list": negative_max_pos_lcs,  # 负样本与正样本的max LCS
        "positive_contrastive_scores": positive_contrastive_scores,
        "negative_contrastive_scores": negative_contrastive_scores,
        "avg_positive_lcs": avg_positive_lcs,
        "avg_negative_lcs": avg_negative_lcs,
        "avg_positive_contrastive": avg_positive_contrastive,
        "avg_negative_contrastive": avg_negative_contrastive,
        "positive_count": len(positive_indices),
        "negative_count": len(negative_indices)
    }


def visualize_lcs(all_positive_lcs, all_negative_lcs, all_positive_contrastive, all_negative_contrastive, output_path):
    """生成可视化图表"""
    
    if not all_positive_lcs and not all_negative_lcs:
        print("⚠️  没有数据可视化")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Token-level LCS Similarity & Contrastive Score Distribution (SimCLR-style)', 
                 fontsize=16, fontweight='bold')
    
    # 1. LCS散点图
    ax1 = axes[0, 0]
    if all_positive_lcs:
        ax1.scatter(range(len(all_positive_lcs)), all_positive_lcs, 
                   alpha=0.6, s=50, c='green', label='Positive', marker='o')
    if all_negative_lcs:
        ax1.scatter(range(len(all_positive_lcs), len(all_positive_lcs) + len(all_negative_lcs)), 
                   all_negative_lcs, alpha=0.6, s=50, c='red', label='Negative', marker='x')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('LCS Similarity')
    ax1.set_title('LCS Similarity by Sample')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 2. LCS箱线图
    ax2 = axes[0, 1]
    data_to_plot = []
    labels = []
    if all_positive_lcs:
        data_to_plot.append(all_positive_lcs)
        labels.append(f'Positive\n(n={len(all_positive_lcs)})')
    if all_negative_lcs:
        data_to_plot.append(all_negative_lcs)
        labels.append(f'Negative\n(n={len(all_negative_lcs)})')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    if len(bp['boxes']) >= 1:
        bp['boxes'][0].set_facecolor('lightgreen')
    if len(bp['boxes']) >= 2:
        bp['boxes'][-1].set_facecolor('lightcoral')
    
    ax2.set_ylabel('LCS Similarity')
    ax2.set_title('LCS Distribution Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])
    
    # 3. 对比分数散点图
    ax3 = axes[0, 2]
    if all_positive_contrastive:
        ax3.scatter(range(len(all_positive_contrastive)), all_positive_contrastive,
                   alpha=0.6, s=50, c='green', label='Positive', marker='o')
    if all_negative_contrastive:
        ax3.scatter(range(len(all_positive_contrastive), len(all_positive_contrastive) + len(all_negative_contrastive)), 
                   all_negative_contrastive, alpha=0.6, s=50, c='red', label='Negative', marker='x')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Contrastive Score')
    ax3.set_title('Contrastive Score by Sample')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. LCS直方图
    ax4 = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    if all_positive_lcs:
        ax4.hist(all_positive_lcs, bins=bins, alpha=0.6, color='green', 
                label=f'Positive (μ={np.mean(all_positive_lcs):.3f})', edgecolor='black')
    if all_negative_lcs:
        ax4.hist(all_negative_lcs, bins=bins, alpha=0.6, color='red', 
                label=f'Negative (μ={np.mean(all_negative_lcs):.3f})', edgecolor='black')
    ax4.set_xlabel('LCS Similarity')
    ax4.set_ylabel('Frequency')
    ax4.set_title('LCS Similarity Histogram')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 对比分数直方图
    ax5 = axes[1, 1]
    if all_positive_contrastive:
        ax5.hist(all_positive_contrastive, bins=20, alpha=0.6, color='green', 
                label=f'Positive (μ={np.mean(all_positive_contrastive):.3f})', edgecolor='black')
    if all_negative_contrastive:
        ax5.hist(all_negative_contrastive, bins=20, alpha=0.6, color='red', 
                label=f'Negative (μ={np.mean(all_negative_contrastive):.3f})', edgecolor='black')
    ax5.set_xlabel('Contrastive Score')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Contrastive Score Histogram')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 统计信息
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats_text = "Statistical Summary\n" + "="*32 + "\n\n"
    if all_positive_lcs:
        stats_text += f"Positive LCS (n={len(all_positive_lcs)}):\n"
        stats_text += f"  Mean:   {np.mean(all_positive_lcs):.4f}\n"
        stats_text += f"  Median: {np.median(all_positive_lcs):.4f}\n"
        stats_text += f"  Std:    {np.std(all_positive_lcs):.4f}\n\n"
    
    if all_negative_lcs:
        stats_text += f"Negative LCS (n={len(all_negative_lcs)}):\n"
        stats_text += f"  Mean:   {np.mean(all_negative_lcs):.4f}\n"
        stats_text += f"  Median: {np.median(all_negative_lcs):.4f}\n"
        stats_text += f"  Std:    {np.std(all_negative_lcs):.4f}\n\n"
    
    if all_positive_lcs and all_negative_lcs:
        diff = np.mean(all_positive_lcs) - np.mean(all_negative_lcs)
        stats_text += f"LCS Diff: {diff:+.4f}\n\n"
    
    if all_positive_contrastive:
        stats_text += f"Positive Contrastive:\n"
        stats_text += f"  Mean:   {np.mean(all_positive_contrastive):.4f}\n"
        stats_text += f"  Std:    {np.std(all_positive_contrastive):.4f}\n\n"
    
    if all_negative_contrastive:
        stats_text += f"Negative Contrastive:\n"
        stats_text += f"  Mean:   {np.mean(all_negative_contrastive):.4f}\n"
        stats_text += f"  Std:    {np.std(all_negative_contrastive):.4f}\n\n"
    
    if all_positive_contrastive and all_negative_contrastive:
        diff_c = np.mean(all_positive_contrastive) - np.mean(all_negative_contrastive)
        stats_text += f"Contrastive Diff: {diff_c:+.4f}\n"
    
    ax6.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 保存 PDF 版本
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()


def analyze_single_question(args_tuple):
    """单个问题的分析（用于并行处理）"""
    result, tokenizer_path, lcs_max_tokens, lcs_normalize_mode, temperature = args_tuple
    
    if not result['success']:
        return None
    
    try:
        analysis = analyze_with_lcs(
            result['responses'],
            tokenizer_path,
            lcs_max_tokens,
            lcs_normalize_mode,
            temperature
        )
        return analysis
    except Exception as e:
        print(f"⚠️  问题 {result.get('index', '?')} 分析失败: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="分析缓存的推理结果")
    
    parser.add_argument("--cache-file", type=str, required=True, help="缓存文件路径")
    parser.add_argument("--lcs-max-tokens", type=int, default=2048, help="LCS 最大 tokens")
    parser.add_argument("--lcs-normalize-mode", type=str, default="avg",
                       choices=["avg", "l1", "l2", "max", "min"], help="归一化模式")
    parser.add_argument("--output-dir", type=str, default="inference_results", help="输出目录")
    parser.add_argument("--num-workers", type=int, default=None, help="并行进程数（默认: CPU核心数）")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="对比学习温度参数 τ (默认: 0.1, 范围: 0.05-0.5)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载缓存
    print(f"加载缓存: {args.cache_file}")
    with open(args.cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    results = cached_data['results']
    metadata = cached_data['metadata']
    
    print(f"✓ 缓存加载成功")
    print(f"  推理配置: n_samples={metadata['n_samples']}, "
          f"temperature={metadata['temperature']}, top_p={metadata['top_p']}")
    print(f"  问题数: {len(results)}\n")
    
    # 确定并行进程数
    num_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    print(f"并行进程数: {num_workers}\n")
    
    print("="*80)
    print(f"LCS 分析配置: max_tokens={args.lcs_max_tokens}, mode={args.lcs_normalize_mode}")
    print(f"对比学习温度: τ={args.temperature}")
    print("="*80 + "\n")
    
    # 准备并行处理的参数
    analysis_tasks = [
        (result, CHECKPOINT_PATH, args.lcs_max_tokens, args.lcs_normalize_mode, args.temperature)
        for result in results
    ]
    
    # 并行分析所有结果
    all_positive_lcs = []
    all_negative_lcs = []
    all_positive_contrastive = []
    all_negative_contrastive = []
    
    print(f"开始并行分析 {len(results)} 个问题...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(analyze_single_question, task): i 
                  for i, task in enumerate(analysis_tasks)}
        
        # 收集结果，带进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="分析进度"):
            analysis = future.result()
            if analysis is not None:
                all_positive_lcs.extend(analysis['positive_lcs_list'])
                all_negative_lcs.extend(analysis['negative_lcs_list'])
                all_positive_contrastive.extend(analysis['positive_contrastive_scores'])
                all_negative_contrastive.extend(analysis['negative_contrastive_scores'])
    
    print("✓ 分析完成\n")
    
    # 输出统计
    print(f"正样本: {len(all_positive_lcs)} 个, 平均 LCS: {np.mean(all_positive_lcs):.4f}, 平均对比分数: {np.mean(all_positive_contrastive):.4f}" if all_positive_lcs else "正样本: 0 个")
    print(f"负样本: {len(all_negative_lcs)} 个, 平均 LCS: {np.mean(all_negative_lcs):.4f}, 平均对比分数: {np.mean(all_negative_contrastive):.4f}" if all_negative_lcs else "负样本: 0 个")
    
    # 可视化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"lcs_viz_maxlen{args.lcs_max_tokens}_norm{args.lcs_normalize_mode}.png"
    output_path = output_dir / filename
    
    print(f"\n生成可视化...")
    visualize_lcs(all_positive_lcs, all_negative_lcs, all_positive_contrastive, all_negative_contrastive, str(output_path))
    print(f"✓ 图表已保存: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    print(f"✓ PDF 已保存: {pdf_path}")


if __name__ == "__main__":
    main()

