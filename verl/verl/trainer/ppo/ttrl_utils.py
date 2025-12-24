# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
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
from typing import List
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def _ensure_extra_info(data_item):
    """
    确保 data_item 的 extra_info 字段存在且为字典。
    
    Args:
        data_item: DataProtoItem 对象
        
    Returns:
        extra_info 字典的引用
    """
    if "extra_info" not in data_item.non_tensor_batch:
        data_item.non_tensor_batch["extra_info"] = {}
    elif not isinstance(data_item.non_tensor_batch["extra_info"], dict):
        data_item.non_tensor_batch["extra_info"] = {}
    return data_item.non_tensor_batch["extra_info"]


def apply_ttrl_gt(
    batch, 
    gen_batch_output, 
    n, 
    tokenizer, 
    process_reward_weight=0, 
    process_reward_strategy="avg", 
    process_lcs_norm="avg",
    process_lcs_max_tokens=3000,
    use_contrastive_process_reward=False,
    contrastive_temperature=0.1
):
    """
    Apply the majority vote ground truth to the batch.
    
    Args:
        batch: 批次数据
        gen_batch_output: 生成的批次输出
        n: 每个 prompt 的样本数
        tokenizer: 分词器
        process_reward_weight: 过程奖励的权重
        process_reward_strategy: 过程奖励的组合策略（"sum" 或 "avg"）
        process_lcs_norm: LCS 归一化策略（"max", "avg", "min", "l1", "l2"）
        process_lcs_max_tokens: LCS 计算的最大 token 长度（性能优化）
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # 保存文本和 token IDs（用于后续的 token-based LCS）
            model_outputs.append({
                'text': response_str,
                'token_ids': valid_response_ids.tolist()
            })
            # extra_info = _ensure_extra_info(data_item)
            # extra_info["solution_token_ids"] = valid_response_ids.tolist()

    majority_gt_list, majority_ratio_list, majority_items_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        # 提取文本和 token IDs
        majority_items = majority_items_list[i]
        majority_texts = [item['text'] for item in majority_items]
        majority_token_ids = [item['token_ids'] for item in majority_items]

        data_item.non_tensor_batch["reward_model"]["majority_texts"] = majority_texts

        # 将投票结果添加到 batch（prompt 级别）的 extra_info
        extra_info = _ensure_extra_info(data_item)
        extra_info["majority_texts"] = majority_texts
        extra_info["majority_token_ids"] = majority_token_ids
        extra_info["process_reward_weight"] = process_reward_weight
        extra_info["process_reward_strategy"] = process_reward_strategy
        extra_info["process_lcs_norm"] = process_lcs_norm
        extra_info["process_lcs_max_tokens"] = process_lcs_max_tokens
        
        # 🆕 收集这个 group 的所有 solution token_ids（用于对比学习）
        start = i * n
        group_solution_token_ids = []
        for j in range(n):
            solution_token_ids = model_outputs[start + j]['token_ids']
            group_solution_token_ids.append(solution_token_ids)
        
        # 🆕 对比学习字段
        extra_info["use_contrastive_process_reward"] = use_contrastive_process_reward
        extra_info["contrastive_temperature"] = contrastive_temperature
        extra_info["group_solution_token_ids"] = group_solution_token_ids
        extra_info["group_key"] = id(group_solution_token_ids)

        # for j in range(n):
        #     gen_data_item = gen_batch_output[start + j]
            
        #     extra_info = _ensure_extra_info(gen_data_item)
            
        #     extra_info["majority_texts"] = majority_texts
        #     extra_info["majority_token_ids"] = majority_token_ids
        #     extra_info["process_reward_weight"] = process_reward_weight
        #     extra_info["process_reward_strategy"] = process_reward_strategy
        #     extra_info["process_lcs_norm"] = process_lcs_norm
        #     extra_info["process_lcs_max_tokens"] = process_lcs_max_tokens
        #     # 🆕 对比学习字段
        #     extra_info["use_contrastive_process_reward"] = use_contrastive_process_reward
        #     extra_info["contrastive_temperature"] = contrastive_temperature
        #     extra_info["group_solution_token_ids"] = group_solution_token_ids

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch, gen_batch_output


def _batch_majority_vote(model_outputs: List[dict], n: int) -> tuple[List[str], List[float], List[List[dict]]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of dict with 'text' and 'token_ids'
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
        majority_items_list: list of list of dict (每个prompt对应的多数答案的原始数据列表，包含text和token_ids)
    """
    majority_gt_list = []
    majority_ratio_list = []
    majority_items_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio, prompt_majority_items = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        majority_items_list.append(prompt_majority_items)
        
    return majority_gt_list, majority_ratio_list, majority_items_list


def _majority_vote(model_outputs: List[dict]) -> tuple[str, float, List[dict]]:
    """
    Args:
        model_outputs: List of dict with 'text' and 'token_ids'
    Returns:
        majority_answer: str
        majority_ratio: float
        majority_items: List of dict with 'text' and 'token_ids'
    """
    assert len(model_outputs) > 0
    
    # 提取和简化答案，同时保留原始数据的映射关系
    answer_to_items = {}  # 简化答案 -> 原始数据列表的映射
    model_answers = []
    
    for output_item in model_outputs:
        generated_text = output_item['text']
        extracted_answer = extract_answer(generated_text)
        if extracted_answer is not None:
            simplified_answer = simplify_expression_string(extracted_answer)
            model_answers.append(simplified_answer)
            
            # 记录该简化答案对应的原始数据（包含 text 和 token_ids）
            if simplified_answer not in answer_to_items:
                answer_to_items[simplified_answer] = []
            answer_to_items[simplified_answer].append(output_item)
    
    if len(model_answers) == 0:
        return "None", 0.0, []
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    # 获取多数答案对应的所有原始数据（包含 text 和 token_ids）
    majority_items = answer_to_items.get(majority_answer, [])
    
    return majority_answer, majority_ratio, majority_items


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric