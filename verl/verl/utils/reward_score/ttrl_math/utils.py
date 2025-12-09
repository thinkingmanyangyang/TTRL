# Copyright 2024 TTRL Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for TTRL math reward computation.
"""

import numpy as np
from typing import List, Optional


def longest_common_subsequence_length(str1: str, str2: str) -> int:
    """
    计算两个字符串的最长公共子序列（LCS）长度。
    
    使用动态规划算法，时间复杂度 O(m*n)，空间复杂度 O(m*n)。
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
    
    Returns:
        最长公共子序列的长度
    
    Examples:
        >>> longest_common_subsequence_length("ABCD", "ACBAD")
        3  # "ABD"
        >>> longest_common_subsequence_length("hello", "hola")
        2  # "hl"
    """
    if not str1 or not str2:
        return 0
    
    m, n = len(str1), len(str2)
    
    # 创建 DP 表格
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充 DP 表格
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def lcs_similarity_score(str1: str, str2: str, normalize: bool = True) -> float:
    """
    计算两个字符串基于 LCS 的相似度分数。
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
        normalize: 是否归一化到 [0, 1] 区间
    
    Returns:
        相似度分数。如果 normalize=True，返回 [0, 1] 范围内的分数；
        否则返回原始 LCS 长度。
    
    Examples:
        >>> lcs_similarity_score("ABCD", "ACBAD", normalize=True)
        0.6  # LCS=3, max_len=5, score=3/5
        >>> lcs_similarity_score("hello", "hello", normalize=True)
        1.0  # 完全相同
    """
    if not str1 or not str2:
        return 0.0
    
    lcs_len = longest_common_subsequence_length(str1, str2)
    
    if not normalize:
        return float(lcs_len)
    
    # 归一化：LCS长度 / 两个字符串长度的平均值
    avg_len = (len(str1) + len(str2)) / 2.0
    if avg_len == 0:
        return 0.0
    
    return lcs_len / avg_len


def max_lcs_similarity_with_texts(
    target_str: str,
    reference_texts: List[str],
    normalize: bool = True
) -> float:
    """
    计算目标字符串与参考文本列表中所有文本的 LCS 相似度的最大值。
    使用顺序计算（外层已有 batch 级别并行，避免过度并行）。
    
    Args:
        target_str: 目标字符串（例如：solution_str）
        reference_texts: 参考文本列表（例如：majority_texts）
        normalize: 是否归一化到 [0, 1] 区间
    
    Returns:
        最大的 LCS 相似度分数
    
    Examples:
        >>> texts = ["hello world", "hello python", "goodbye"]
        >>> max_lcs_similarity_with_texts("hello there", texts, normalize=True)
        0.8  # 与 "hello world" 的相似度最高
    """
    import time
    import sys
    
    if not target_str or not reference_texts:
        return 0.0
    
    start_time = time.time()
    max_score = 0.0
    
    # 顺序计算（避免嵌套并行）
    for idx, ref_text in enumerate(reference_texts):
        if ref_text:  # 跳过空字符串
            lcs_start = time.time()
            score = lcs_similarity_score(target_str, ref_text, normalize=normalize)
            lcs_elapsed = time.time() - lcs_start
            
            # 只打印慢计算（>0.1秒）
            if lcs_elapsed > 0.1:
                print(f"[PERF]   Text-LCS #{idx+1}/{len(reference_texts)} took {lcs_elapsed:.4f}s | "
                      f"target_len={len(target_str)} ref_len={len(ref_text)} | score={score:.4f}", 
                      flush=True)
            
            max_score = max(max_score, score)
    
    total_time = time.time() - start_time
    if total_time > 0.05:  # 只打印显著的计算
        print(f"[PERF]   Total Text-LCS time: {total_time:.4f}s for {len(reference_texts)} texts | "
              f"avg={total_time/len(reference_texts):.4f}s per text", flush=True)
    
    return max_score


def _lcs_python(seq1: List[int], seq2: List[int]) -> int:
    """
    纯 Python 实现的 LCS 长度计算（Fallback 版本）。
    当 Numba 不可用时使用。
    """
    m, n = len(seq1), len(seq2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev
        curr = [0] * (n + 1)
    
    return prev[n]


def token_lcs_similarity(
    token_ids_1: List[int],
    token_ids_2: List[int],
    max_tokens: int = 1500,
    normalize: bool = True
) -> float:
    """
    基于 token IDs 的 LCS 相似度。
    
    Args:
        token_ids_1: 第一个序列的 token IDs
        token_ids_2: 第二个序列的 token IDs
        max_tokens: 最大 token 长度（防止过长序列导致性能问题）
        normalize: 是否归一化到 [0, 1]
    
    Returns:
        LCS 相似度分数 [0, 1]
    
    Examples:
        >>> ids1 = [100, 200, 300, 400]
        >>> ids2 = [100, 250, 300, 450]
        >>> token_lcs_similarity(ids1, ids2)
        0.5  # LCS=[100, 300], len=2, avg_len=4
    """
    if not token_ids_1 or not token_ids_2:
        return 0.0
    
    # 截断过长序列（性能优化）
    if len(token_ids_1) > max_tokens:
        token_ids_1 = token_ids_1[:max_tokens]
    if len(token_ids_2) > max_tokens:
        token_ids_2 = token_ids_2[:max_tokens]
    
    # 尝试使用 Numba 加速版本（6-10× 加速）
    try:
        from .utils_numba import lcs_length_numba, NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            # Numba 需要 NumPy 数组，提前转换
            import numpy as np
            arr1 = np.array(token_ids_1, dtype=np.int32)
            arr2 = np.array(token_ids_2, dtype=np.int32)
            lcs_len = lcs_length_numba(arr1, arr2)
        else:
            lcs_len = _lcs_python(token_ids_1, token_ids_2)
    except ImportError:
        # Numba 未安装，使用纯 Python 版本
        lcs_len = _lcs_python(token_ids_1, token_ids_2)
    
    if not normalize:
        return float(lcs_len)
    
    # 归一化：LCS 长度 / 两个序列的平均长度
    avg_len = (len(token_ids_1) + len(token_ids_2)) / 2.0
    return lcs_len / avg_len if avg_len > 0 else 0.0


def max_token_lcs_similarity(
    target_token_ids: List[int],
    reference_token_ids_list: List[List[int]],
    max_tokens: int = 1500
) -> float:
    """
    计算目标 token 序列与多个参考序列的最大 LCS 相似度。
    使用顺序计算（外层已有 batch 级别并行，避免过度并行）。
    
    Args:
        target_token_ids: 目标 token 序列
        reference_token_ids_list: 参考 token 序列列表
        max_tokens: 最大 token 长度
    
    Returns:
        最大的 LCS 相似度分数 [0, 1]
    """
    import time
    import sys
    
    if not target_token_ids or not reference_token_ids_list:
        return 0.0
    
    start_time = time.time()
    max_score = 0.0
    
    # 顺序计算（避免嵌套并行）
    for idx, ref_ids in enumerate(reference_token_ids_list):
        if ref_ids:  # 跳过空序列
            lcs_start = time.time()
            score = token_lcs_similarity(target_token_ids, ref_ids, max_tokens)
            lcs_elapsed = time.time() - lcs_start
            
            # 只打印慢计算（>0.1秒）
            if lcs_elapsed > 0.1:
                print(f"[PERF]   Token-LCS #{idx+1}/{len(reference_token_ids_list)} took {lcs_elapsed:.4f}s | "
                      f"target_len={len(target_token_ids)} ref_len={len(ref_ids)} | score={score:.4f}", 
                      flush=True)
            
            max_score = max(max_score, score)
    
    total_time = time.time() - start_time
    if total_time > 0.1:  # 只打印显著的计算
        print(f"[PERF]   Total Token-LCS time: {total_time:.4f}s for {len(reference_token_ids_list)} seqs | "
              f"avg={total_time/len(reference_token_ids_list):.4f}s per seq", flush=True)
    
    return max_score


def compute_process_reward(
    solution_str: str,
    majority_texts: Optional[List[str]],
    weight: float = 0.5,
    normalize: bool = True,
    solution_token_ids: Optional[List[int]] = None,
    majority_token_ids_list: Optional[List[List[int]]] = None,
    max_tokens: int = 1500
) -> float:
    """
    计算过程奖励：solution 与 majority_texts 的最大 LCS 相似度。
    优先使用 token IDs（更快更准确），如果没有则使用文本。
    
    Args:
        solution_str: 当前模型的输出文本（fallback）
        majority_texts: 多数投票的文本列表（fallback）
        weight: 过程奖励的权重
        normalize: 是否归一化到 [0, 1]
        solution_token_ids: 当前模型输出的 token IDs（推荐）
        majority_token_ids_list: 多数投票的 token IDs 列表（推荐）
        max_tokens: token 序列的最大长度（性能优化）
    
    Returns:
        过程奖励分数 [0, weight]
    
    Examples:
        >>> # 使用 token IDs（推荐）
        >>> solution_ids = [100, 200, 300]
        >>> majority_ids = [[100, 250, 300], [100, 200, 350]]
        >>> compute_process_reward(None, None, 1.0, True, solution_ids, majority_ids)
        0.75
    """
    # 优先使用 token IDs（更快更准确）
    if solution_token_ids is not None and majority_token_ids_list is not None:
        if len(majority_token_ids_list) == 0:
            return 0.0
        max_similarity = max_token_lcs_similarity(
            solution_token_ids,
            majority_token_ids_list,
            max_tokens=max_tokens
        )
        return max_similarity * weight
    
    # Fallback：使用文本（向后兼容）
    if majority_texts is None or len(majority_texts) == 0:
        return 0.0
    
    max_similarity = max_lcs_similarity_with_texts(
        solution_str, 
        majority_texts, 
        normalize=normalize
    )
    
    return max_similarity * weight


def token_level_lcs_similarity(str1: str, str2: str, tokenize_by: str = "word") -> float:
    """
    基于 token（词或字符）级别的 LCS 相似度。
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
        tokenize_by: 分词方式，"word" 按空格分词，"char" 按字符
    
    Returns:
        Token 级别的 LCS 相似度 [0, 1]
    
    Examples:
        >>> token_level_lcs_similarity("hello world", "hello python", tokenize_by="word")
        0.5  # 两个词中有一个相同
    """
    if tokenize_by == "word":
        tokens1 = str1.split()
        tokens2 = str2.split()
    elif tokenize_by == "char":
        tokens1 = list(str1)
        tokens2 = list(str2)
    else:
        raise ValueError(f"Unknown tokenize_by: {tokenize_by}")
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # 使用 token 列表计算 LCS
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_len = dp[m][n]
    avg_len = (len(tokens1) + len(tokens2)) / 2.0
    
    if avg_len == 0:
        return 0.0
    
    return lcs_len / avg_len


# 可以添加更多工具函数...

def compute_weighted_process_reward(
    solution_str: str,
    majority_texts: Optional[List[str]],
    char_weight: float = 0.3,
    word_weight: float = 0.7,
) -> float:
    """
    计算加权的过程奖励，结合字符级和词级的 LCS 相似度。
    
    Args:
        solution_str: 当前模型的输出文本
        majority_texts: 多数投票的文本列表
        char_weight: 字符级相似度的权重
        word_weight: 词级相似度的权重
    
    Returns:
        加权过程奖励分数 [0, 1]
    """
    if majority_texts is None or len(majority_texts) == 0:
        return 0.0
    
    # 字符级最大相似度
    max_char_sim = 0.0
    for ref_text in majority_texts:
        if ref_text:
            sim = lcs_similarity_score(solution_str, ref_text, normalize=True)
            max_char_sim = max(max_char_sim, sim)
    
    # 词级最大相似度
    max_word_sim = 0.0
    for ref_text in majority_texts:
        if ref_text:
            sim = token_level_lcs_similarity(solution_str, ref_text, tokenize_by="word")
            max_word_sim = max(max_word_sim, sim)
    
    # 加权组合
    weighted_reward = char_weight * max_char_sim + word_weight * max_word_sim
    
    return weighted_reward
