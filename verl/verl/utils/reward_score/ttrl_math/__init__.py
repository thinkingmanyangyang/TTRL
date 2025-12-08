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
        process_reward_weight = 1.0  # 默认权重
        process_reward_strategy = "avg"  # 默认策略
        
        if extra_info is not None and isinstance(extra_info, dict):
            majority_texts = extra_info.get("majority_texts", None)
            majority_token_ids = extra_info.get("majority_token_ids", None)  # 新增
            solution_token_ids = extra_info.get("solution_token_ids", None)  # 新增
            # 简化调试输出：只显示是否存在和前几个token
            # if solution_token_ids:
            #     print(f"[DEBUG] solution_token_ids: exists, len={len(solution_token_ids)}, first_5={solution_token_ids[:5]}")
            # else:
            #     print(f"[DEBUG] solution_token_ids: None or empty")
            
            # if majority_token_ids:
            #     print(f"[DEBUG] majority_token_ids: exists, count={len(majority_token_ids)}, first_ids_len={len(majority_token_ids[0]) if len(majority_token_ids) > 0 else 0}, first_5={majority_token_ids[0][:5] if len(majority_token_ids) > 0 else []}")
            # else:
            #     print(f"[DEBUG] majority_token_ids: None or empty")
            # 从 extra_info 中读取配置
            process_reward_weight = extra_info.get("process_reward_weight", 1.0)
            process_reward_strategy = extra_info.get("process_reward_strategy", "avg")

        # 计算原始的正确性分数
        res = compute_score(solution_str, str(ground_truth))
        
        # 计算过程奖励（基于 LCS 相似度）
        lcs_similarity = 0.0
        
        # 根据策略组合原始分数和过程奖励
        if isinstance(res, dict):
            original_score = res["score"]
            
            # 只有当原始分数不是 1.0 时，才计算和补充 process reward
            if original_score != 1.0:
                # 优先使用 token IDs（更快更准确）
                use_token_ids = (solution_token_ids is not None and majority_token_ids is not None and 
                                len(majority_token_ids) > 0)
                use_texts = majority_texts is not None and len(majority_texts) > 0
                
                if use_token_ids or use_texts:
                    lcs_similarity = compute_process_reward(
                        solution_str=solution_str,
                        majority_texts=majority_texts[:8],
                        weight=1.0,  # 这里不使用 weight，只获取原始相似度
                        normalize=True,
                        solution_token_ids=solution_token_ids,  # 新增：优先使用 token IDs
                        majority_token_ids_list=majority_token_ids[:8],  # 新增
                        max_tokens=2000  # 性能优化：限制最大 token 长度
                    )
                    
                # 根据策略组合原始分数和过程奖励
                if process_reward_strategy == "sum":
                    # 策略1: 直接相加
                    final_score = original_score + process_reward_weight * lcs_similarity
                else:  # "avg" 或其他值默认使用加权平均
                    # 策略2: 加权平均
                    final_score = (1-process_reward_weight) * original_score + process_reward_weight * lcs_similarity
                
                res["score"] = final_score
            else:
                # 原始分数已经是 1.0，不需要补充 process reward，直接使用原始分数
                res["score"] = original_score
            
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
            else:
                return original_score * (1 - process_reward_weight) + process_reward_weight * lcs_similarity
        else:
            original_score = float(res[0])
            if process_reward_strategy == "sum":
                return original_score + process_reward_weight * lcs_similarity
            else:
                return original_score * (1 - process_reward_weight) + process_reward_weight * lcs_similarity
            
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