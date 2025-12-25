# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""
import random
from latex2sympy2_extended import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
import traceback

from .math_utils import extract_boxed_answer, is_latex_equal, grade_answer_mathd, grade_answer_sympy, timeout_ours

"""
This code is adapted from Entropy Machanism Recipe (https://github.com/volcengine/verl/tree/main/recipe/entropy/).
"""

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct

@timeout_ours(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    try:
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except TimeoutError:
        return expression_string
    except Exception as e:
        try:
            sympy_expr = latex2sympy(expression_string)
            simplified_expr = simplify(sympy_expr)
            return str(simplified_expr)
        except TimeoutError:
            return expression_string
        except Exception as e:
            return expression_string

def compute_score(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }

def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    import time
    import sys  # 用于强制刷新输出
    try:
        from .utils import compute_process_reward
        
        # 从 extra_info 中提取参数
        majority_texts = None
        majority_token_ids = None
        solution_token_ids = None
        process_reward_weight = 0.0  # 默认权重
        process_reward_strategy = "avg"  # 默认策略
        
        # 🆕 对比学习参数
        use_contrastive = False
        contrastive_temperature = 0.1
        
        if extra_info is not None and isinstance(extra_info, dict):
            majority_texts = extra_info.get("majority_texts", None)
            majority_token_ids = extra_info.get("majority_token_ids", None)
            solution_token_ids = extra_info.get("solution_token_ids", None)
            # 从 extra_info 中读取配置
            process_reward_weight = extra_info.get("process_reward_weight", 0.0)
            process_reward_strategy = extra_info.get("process_reward_strategy", "avg")
            process_lcs_norm = extra_info.get("process_lcs_norm", "avg")
            process_lcs_max_tokens = extra_info.get("process_lcs_max_tokens", 3000)
            # 🆕 对比学习配置
            use_contrastive = extra_info.get("use_contrastive_process_reward", False)
            contrastive_temperature = extra_info.get("contrastive_temperature", 0.1)
            
            
            # print(f"[DEBUG] process_reward_weight: {process_reward_weight}, process_reward_strategy: {process_reward_strategy}, "
            #       f"process_lcs_norm: {process_lcs_norm}, process_lcs_max_tokens: {process_lcs_max_tokens}, "
            #       f"use_contrastive: {use_contrastive}")

        # 计算原始的正确性分数
        res = compute_score(solution_str, str(ground_truth))
        
        # 提前计算过程奖励（基于 LCS 相似度）
        lcs_similarity = 0.0
        if process_reward_weight > 0:
            # 检查是否提供了 token IDs（必需）
            if solution_token_ids is None or majority_token_ids is None:
                raise ValueError(
                    f"Process reward is enabled (weight={process_reward_weight}), but token IDs are missing. "
                    f"solution_token_ids: {'None' if solution_token_ids is None else f'len={len(solution_token_ids)}'}, "
                    f"majority_token_ids: {'None' if majority_token_ids is None else f'len={len(majority_token_ids)}'}"
                )
            
            # 如果 majority_token_ids 为空列表，LCS 相似度为 0
            if len(majority_token_ids) > 0:
                if not use_contrastive:
                    # 🔵 传统方式：max LCS with majority
                    lcs_similarity = compute_process_reward(
                        solution_str=solution_str,
                        majority_texts=majority_texts,
                        weight=1.0,  # 这里不使用 weight，只获取原始相似度
                        normalize=True,
                        solution_token_ids=solution_token_ids,
                        majority_token_ids_list=majority_token_ids,
                        max_tokens=process_lcs_max_tokens,  # 性能优化：限制最大 token 长度
                        norm_strategy=process_lcs_norm  # LCS 归一化策略
                    )
                else:
                    # 🆕 对比学习方式：使用 batch 级别采样的负样本
                    sampled_negative_token_ids = extra_info.get("sampled_negative_token_ids", None)
                    
                    if sampled_negative_token_ids is None or len(sampled_negative_token_ids) == 0:
                        # Fallback: 如果没有负样本，回退到传统方法
                        print("[WARNING] Contrastive learning enabled but no sampled_negative_token_ids found, "
                              "falling back to traditional method")
                        lcs_similarity = compute_process_reward(
                            solution_str=solution_str,
                            majority_texts=majority_texts,
                            weight=1.0,
                            normalize=True,
                            solution_token_ids=solution_token_ids,
                            majority_token_ids_list=majority_token_ids,
                            max_tokens=process_lcs_max_tokens,
                            norm_strategy=process_lcs_norm
                        )
                    else:
                        # 使用 batch 级别采样好的负样本计算对比学习 reward
                        lcs_similarity = compute_contrastive_process_reward(
                            solution_token_ids=solution_token_ids,
                            majority_token_ids=majority_token_ids,
                            sampled_negative_token_ids=sampled_negative_token_ids,
                            temperature=contrastive_temperature,
                            max_tokens=process_lcs_max_tokens,
                            norm_strategy=process_lcs_norm
                        )
        
        # 根据策略组合原始分数和过程奖励
        if isinstance(res, dict):
            original_score = res["score"]
            
            # 只有当原始分数不是 1.0 且 process_reward_weight > 0 时，才补充 process reward
            if original_score != 1.0 and process_reward_weight > 0:
                    
                # 根据策略组合原始分数和过程奖励
                if process_reward_strategy == "sum":
                    # 策略1: 直接相加
                    final_score = original_score + process_reward_weight * lcs_similarity
                elif process_reward_strategy == "avg":  # "avg" 或其他值默认使用加权平均
                    # 策略2: 加权平均
                    final_score = (1-process_reward_weight) * original_score + process_reward_weight * lcs_similarity
                elif process_reward_strategy == "neg":
                    # 策略3: 负数奖励
                    final_score = -1 + process_reward_weight * lcs_similarity
                else:
                    final_score = original_score
                    print(f"[WARNING] Invalid process_reward_strategy: {process_reward_strategy}, using original score")
                
                res["score"] = final_score
            else:
                # 原始分数已经是 1.0 或 process_reward_weight == 0，直接使用原始分数
                final_score = original_score
                res["score"] = final_score
            
            # print(f"[DEBUG] final_score: {final_score}, original_score: {original_score}, lcs_similarity: {lcs_similarity}, process_reward_weight: {process_reward_weight}, process_reward_strategy: {process_reward_strategy}")
            # 保存详细信息用于分析
            res["original_score"] = original_score  # 原始正确性分数
            res["lcs_similarity"] = lcs_similarity  # LCS 相似度 (0-1)
            res["process_reward"] = process_reward_weight * lcs_similarity  # 加权后的过程奖励
            res["majority_texts_count"] = len(majority_texts) if majority_texts else 0
            
            return res
        elif isinstance(res, (int, float, bool)):
            # 如果返回的是简单数值
            original_score = float(res)
            if process_reward_strategy == "sum":
                return original_score + process_reward_weight * lcs_similarity
            elif process_reward_strategy == "avg":
                return original_score * (1 - process_reward_weight) + process_reward_weight * lcs_similarity
            elif process_reward_strategy == "neg":
                return -1 + process_reward_weight * lcs_similarity
            else:
                print(f"[WARNING] Invalid process_reward_strategy: {process_reward_strategy}, using original score")
                return original_score
        else:
            original_score = float(res[0])
            if process_reward_strategy == "sum":
                return original_score + process_reward_weight * lcs_similarity
            elif process_reward_strategy == "avg":
                return original_score * (1 - process_reward_weight) + process_reward_weight * lcs_similarity
            elif process_reward_strategy == "neg":
                return -1 + process_reward_weight * lcs_similarity
            else:
                print(f"[WARNING] Invalid process_reward_strategy: {process_reward_strategy}, using original score")
                return original_score
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise


def reward_func_batch(
    data_sources, solution_strs, ground_truths, extra_infos=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    批量版本的 reward_func，适配 BatchRewardManager。
    使用 ThreadPoolExecutor 并行处理，保持样本顺序。
    
    Args:
        data_sources: List[str] - 数据源列表
        solution_strs: List[str] - 答案字符串列表
        ground_truths: List[str] - 真实答案列表
        extra_infos: List[dict] - 额外信息列表
    
    Returns:
        List[dict] - 每个样本的评分结果（按原始顺序）
    """
    import time
    import sys
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    batch_size = len(solution_strs)
    
    if extra_infos is None:
        extra_infos = [None] * batch_size
    
    batch_start_time = time.time()
    
    # 动态获取 CPU 核心数
    cpu_count = os.cpu_count() or 48
    max_workers = min(cpu_count, batch_size)
    
    print(f"[PERF] reward_func_batch processing {batch_size} samples with {max_workers} workers...", flush=True)
    
    # 🔥 Prompt 级别的负样本采样（确保同一 prompt 的所有 responses 使用相同的负样本）
    if extra_infos is not None and len(extra_infos) > 0 and isinstance(extra_infos[0], dict):
        use_contrastive = extra_infos[0].get("use_contrastive_process_reward", False)
        
        if use_contrastive:
            import random
            print(f"[CONTRASTIVE] Performing prompt-level negative sampling...", flush=True)
            
            # 按 prompt 分组（使用 group_solution_token_ids 的 id 作为 key）
            prompt_groups = {}  # {group_id: [sample_indices]}
            group_to_group_solution = {}  # {group_id: group_solution_token_ids}
            group_to_majority = {}  # {group_id: majority_token_ids}
            
            for idx, extra_info in enumerate(extra_infos):
                if extra_info is None:
                    continue
                
                group_solution_token_ids = extra_info.get("group_solution_token_ids", None)

                majority_token_ids = extra_info.get("majority_token_ids", None)
                
                if group_solution_token_ids is not None:
                    # 使用 group_solution_token_ids 的 id 作为 prompt 的标识
                    group_id = extra_info.get("group_key", None)

                    if group_id not in prompt_groups:
                        prompt_groups[group_id] = []
                        group_to_group_solution[group_id] = group_solution_token_ids
                        group_to_majority[group_id] = majority_token_ids
                    
                    prompt_groups[group_id].append(idx)
            # 🔍 Debug: 输出 prompt groups 的数目和 group 0 的长度
            print(f"[DEBUG] Number of prompt groups: {len(prompt_groups)}", flush=True)
            if len(prompt_groups) > 0:
                first_group_id = list(prompt_groups.keys())[0]
                first_group_length = len(prompt_groups[first_group_id])
                print(f"[DEBUG] Length of first group (group_id={first_group_id}): {first_group_length}", flush=True)
            # 为每个 prompt 采样负样本
            max_neg_samples = 5
            for group_id, sample_indices in prompt_groups.items():
                group_solution_token_ids = group_to_group_solution[group_id]
                majority_token_ids = group_to_majority[group_id]
                
                if majority_token_ids is None or len(majority_token_ids) == 0:
                    continue
                
                # 🆕 获取 n_samples_per_prompt（用于训练的回复数量）
                # 从 extra_info 中读取，或者使用 sample_indices 的长度作为近似
                first_sample_idx = sample_indices[0] if sample_indices else 0
                n_samples_per_prompt = extra_infos[first_sample_idx].get("n_samples_per_prompt", len(sample_indices))
                n_votes_per_prompt = len(group_solution_token_ids)
                
                # 构建 majority_sets
                majority_sets = [set(maj_ids) for maj_ids in majority_token_ids]
                
                # 🔥 只从未被选中的回复（indices n_samples_per_prompt 到 n_votes_per_prompt-1）中选择负样本
                # 这样可以避免负样本与用于 GRPO 训练的回复重复
                negative_candidates = []
                negative_candidates_2 = []
                for idx, group_token_ids in enumerate(group_solution_token_ids):
                    group_set = set(group_token_ids)
                    is_negative = all(group_set != maj_set for maj_set in majority_sets)
                    if is_negative:
                        if idx >= n_samples_per_prompt:
                            negative_candidates.append(group_token_ids)
                        else:
                            negative_candidates_2.append(group_token_ids)
                
                # 🔍 Debug: 输出负样本候选的数量
                print(f"[DEBUG] Group {group_id}: negative_candidates from unselected={len(negative_candidates)} "
                      f"(indices {n_samples_per_prompt}-{n_votes_per_prompt-1})", flush=True)
                
                # 采样负样本（从未被选中的回复中随机采样）
                if len(negative_candidates) > max_neg_samples:
                    sampled_negatives = random.sample(negative_candidates, max_neg_samples)
                else:
                    # 未选中的不够，用训练样本中的负样本补充
                    needed = max_neg_samples - len(negative_candidates)
                    sampled_negatives = negative_candidates + negative_candidates_2[:needed]
                # 将采样的负样本添加到该 prompt 的所有 responses
                for idx in sample_indices:
                    extra_infos[idx]["sampled_negative_token_ids"] = sampled_negatives
    
    # 定义单个样本处理函数（返回索引和结果）
    def process_single_sample(idx):
        result = reward_func(
            data_source=data_sources[idx] if isinstance(data_sources, list) else data_sources,
            solution_str=solution_strs[idx],
            ground_truth=ground_truths[idx],
            extra_info=extra_infos[idx],
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore
        )
        return idx, result
    
    # 并行处理，使用字典保持顺序
    results_dict = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_sample, i): i for i in range(batch_size)}
        
        for future in as_completed(futures):
            idx, result = future.result()
            results_dict[idx] = result
    
    # 按原始索引顺序组装结果
    results = [results_dict[i] for i in range(batch_size)]
    
    batch_elapsed = time.time() - batch_start_time
    print(f"[PERF] reward_func_batch completed: {batch_elapsed:.4f}s for {batch_size} samples "
          f"(avg={batch_elapsed/batch_size:.4f}s/sample) with {max_workers} workers", flush=True)
    sys.stdout.flush()
    
    return results


def compute_contrastive_process_reward(
    solution_token_ids,
    majority_token_ids,
    sampled_negative_token_ids,
    temperature=0.1,
    max_tokens=3000,
    norm_strategy="avg"
):
    """
    计算对比学习风格的过程奖励（使用预采样的负样本）
    
    思路：
    1. 使用 majority_token_ids 区分当前样本是正还是负
    2. 计算 max_pos_sim（与 majority 的最大相似度）
    3. 计算 avg_neg_sim（与预采样负样本的平均相似度）
    4. 使用 SimCLR 风格的对比学习公式
    
    Args:
        solution_token_ids: 当前 solution 的 token IDs
        majority_token_ids: majority 答案的 token IDs 列表（用于判断正负样本）
        sampled_negative_token_ids: 预采样的负样本 token IDs 列表
        temperature: 温度参数（默认 0.1）
        max_tokens: LCS 计算的最大 token 长度
        norm_strategy: LCS 归一化策略
    
    Returns:
        float: 对比学习奖励 (0-1)
    """
    from .utils import token_lcs_similarity
    import numpy as np
    
    # ========== 步骤1: 判断当前样本是否为正样本 ==========
    solution_set = set(solution_token_ids)
    majority_sets = [set(maj_ids) for maj_ids in majority_token_ids]
    is_positive = any(solution_set == maj_set for maj_set in majority_sets)
    
    # ========== 步骤2: 检查负样本是否存在 ==========
    if sampled_negative_token_ids is None or len(sampled_negative_token_ids) == 0:
        # Fallback: 没有负样本时返回默认值
        return 1.0 if is_positive else 0.5
    
    # ========== 步骤3: 计算与正样本的最大相似度 ==========
    max_pos_sim = 0.0
    
    if is_positive:
        # 当前是正样本：与 majority 中的其他样本比较
        for maj_ids in majority_token_ids:
            if maj_ids != solution_token_ids:  # 排除自己
                sim = token_lcs_similarity(
                    solution_token_ids, maj_ids,
                    max_tokens=max_tokens,
                    normalize=True,
                    norm_strategy=norm_strategy
                )
                max_pos_sim = max(max_pos_sim, sim)
    else:
        # 当前是负样本：与所有 majority 样本比较
        for maj_ids in majority_token_ids:
            sim = token_lcs_similarity(
                solution_token_ids, maj_ids,
                max_tokens=max_tokens,
                normalize=True,
                norm_strategy=norm_strategy
            )
            max_pos_sim = max(max_pos_sim, sim)
    
    # ========== 步骤4: 计算与负样本的平均相似度 ==========
    neg_sims = []
    for neg_ids in sampled_negative_token_ids:
        # 🛡️ 鲁棒性：排除与自己相同的负样本（理论上不应该存在，但防止边界情况）
        if neg_ids == solution_token_ids or set(neg_ids) == solution_set:
            continue
        
        sim = token_lcs_similarity(
            solution_token_ids, neg_ids,
            max_tokens=max_tokens,
            normalize=True,
            norm_strategy=norm_strategy
        )
        neg_sims.append(sim)
    
    avg_neg_sim = np.mean(neg_sims) if len(neg_sims) > 0 else 0.0
    
    # ========== 步骤5: 计算对比学习分数（SimCLR 风格）==========
    if avg_neg_sim > 0 and max_pos_sim > 0:
        numerator = np.exp(max_pos_sim / temperature)
        denominator = numerator + np.exp(avg_neg_sim / temperature)
        contrastive_score = numerator / denominator
    else:
        # 边界情况处理
        contrastive_score = 1.0 if is_positive else 0.0
    
    return float(contrastive_score)