import torch
import torch.nn.functional as F

@torch.jit.script
def _parallel_dp_step(current_dp: torch.Tensor, P: torch.Tensor, max_steps: int) -> torch.Tensor:
    """
    将 DP 循环用 JIT 编译，减少 Python 解释器开销。
    """
    for t in range(max_steps):
        P_t = P[:, t, :]
        if t == 0:
            current_dp = P_t
        else:
            # DP 转移: dp[t][j] = P_t[j] + max_{k<=j}(dp[t-1][k] - P_t[k])
            prev_Q = current_dp - P_t
            # cummax 也是 GPU 优化的扫描算子
            max_Q, _ = torch.cummax(prev_Q, dim=-1)
            current_dp = P_t + max_Q
    return current_dp

def fast_parallel_step_lcs_score(
    maj_input_ids: torch.Tensor,      # [B, L_maj]
    maj_entropies: torch.Tensor,      # [B, L_maj]
    stud_input_ids: torch.Tensor,     # [B, L_stu]
    entropy_threshold: float = 2.0,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    并行计算步骤对齐的 LCS 分数（优化版本，使用 int8/int16 节省显存，支持 LCS 范围 0-3072）。
    
    注意：虽然尽量使用整数类型，但以下操作必须使用 float：
    - torch.bmm: 只支持 float 类型（float16/float32）
    - torch.cumsum: 只支持 float 类型
    - torch.cummax: 只支持 float 类型
    因此计算时临时使用 float16/float32，但存储时尽量使用 int8/int16。
    
    Args:
        maj_input_ids: [B, L_maj] 标准答案的 token IDs
        maj_entropies: [B, L_maj] 标准答案的熵值（用于步骤划分）
        stud_input_ids: [B, L_stu] 学生答案的 token IDs
        entropy_threshold: 熵阈值，用于步骤划分
        pad_token_id: Padding token ID
    
    Returns:
        scores: [B] 归一化后的相似度分数
    """
    # 1. 基础维度信息
    B, L_maj = maj_input_ids.shape
    _, L_stu = stud_input_ids.shape
    device = maj_input_ids.device

    # ---------------------------------------------------------
    # 2. 步骤划分 (Step Segmentation)
    # ---------------------------------------------------------
    # 逻辑：熵 > 阈值 视为步骤切分点
    is_split = (maj_entropies > entropy_threshold).long()
    non_pad = (maj_input_ids != pad_token_id).long()
    is_split = is_split & non_pad
    is_split[:, -1] = 1  # 强制最后一个位置是切分点
    
    step_ids = torch.cumsum(is_split, dim=1) - is_split
    max_steps = step_ids.max().item() + 1
    
    # [B, L_maj, MaxSteps]
    # 优化：使用 int8 存储 step_masks（0 或 1），节省显存
    step_masks = F.one_hot(step_ids, num_classes=max_steps).to(dtype=torch.int8)
    # 过滤 Padding：将 non_pad 转为 int8 后相乘
    non_pad_int8 = non_pad.to(dtype=torch.int8).unsqueeze(-1)
    step_masks = step_masks * non_pad_int8

    # ---------------------------------------------------------
    # 3. 极速矩阵乘法 (使用 int8 节省显存)
    # ---------------------------------------------------------
    # 关键优化：使用 int8 存储输入（0 或 1），节省显存
    # 注意：PyTorch 的 BMM 只支持 float 类型，所以计算时临时转为 float16
    # 优化：BMM 结果直接转 float32 用于后续 DP，避免 int16 中间转换（减少一次类型转换开销）
    # [B, L_maj, 1]
    maj_exp = maj_input_ids.unsqueeze(2) 
    # [B, 1, L_stu]
    stud_exp = stud_input_ids.unsqueeze(1)
    
    # 生成匹配矩阵 [B, L_maj, L_stu]
    # 注意：这里我们转置了逻辑，为了后续 MatMul: (L_stu x L_maj) @ (L_maj x MaxSteps)
    # 所以我们先生成 [B, L_stu, L_maj]
    raw_matches = (maj_exp == stud_exp).transpose(1, 2)  # -> [B, L_stu, L_maj]
    
    # 过滤 Padding
    stud_non_pad = (stud_input_ids != pad_token_id).unsqueeze(-1)  # [B, L_stu, 1]
    raw_matches = raw_matches & stud_non_pad
    
    # 优化：使用 int8 存储（0 或 1），节省显存
    # 注意：int8 范围是 -128 到 127，0 和 1 完全在范围内
    raw_matches_int8 = raw_matches.to(dtype=torch.int8)
    
    # BMM 只支持 float 类型（不支持 int），所以计算时临时转为 float16（节省显存）
    # BMM: [B, L_stu, L_maj] @ [B, L_maj, MaxSteps] -> [B, L_stu, MaxSteps]
    # 物理意义：对于每个学生的每个词(Row)，它在 Standard 的哪些 Step(Col) 里出现过？
    # 这一步是整个算法的计算核心，BMM 是 GPU 上最快的算子
    step_token_matches = torch.bmm(
        raw_matches_int8.to(dtype=torch.float16),  # 临时转 float16 用于计算（BMM 不支持 int）
        step_masks.to(dtype=torch.float16)         # 临时转 float16 用于计算
    )
    
    # 优化：直接转 float32 用于后续 DP 计算，避免 int16 中间转换（减少一次类型转换）
    # 注意：虽然想用整数存储，但 int16->float32 的转换是多余的，直接 float16->float32 更快
    # 显存权衡：float32 (4 bytes) vs int16 (2 bytes)，但 DP 阶段需要 float 类型
    # 由于 DP 阶段需要 float，提前转换可以避免后续的 int16->float32 转换开销
    step_token_matches = step_token_matches.float()  # float16 -> float32，直接用于 DP

    # ---------------------------------------------------------
    # 4. 并行 DP
    # ---------------------------------------------------------
    # 调整维度: [B, L_stu, MaxSteps] -> [B, MaxSteps, L_stu]
    step_token_matches = step_token_matches.permute(0, 2, 1)  # -> [B, MaxSteps, L_stu]
    
    # 前缀和
    P = torch.cumsum(step_token_matches, dim=-1)  # [B, MaxSteps, L_stu]
    
    # 初始化 DP (JIT 加速)
    current_dp = torch.zeros(B, L_stu, device=device)
    current_dp = _parallel_dp_step(current_dp, P, max_steps)
            
    # ---------------------------------------------------------
    # 5. 结果
    # ---------------------------------------------------------
    final_scores = current_dp[:, -1]  # [B]
    maj_lens = non_pad.sum(dim=1)  # [B]
    return final_scores / (maj_lens + 1e-6)  # [B]

