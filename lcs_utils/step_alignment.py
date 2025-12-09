#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤对齐算法：基于Token级LCS实现A和B的步骤硬对齐
给定A的固定切分和token级匹配对，找B的最优切分使得步骤匹配最大化
"""

import numpy as np
from typing import List, Tuple, Optional


def weighted_lcs(X: List[str], Y: List[str], wx: np.ndarray, wy: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """
    计算加权最长公共子序列
    
    Args:
        X: A的token序列
        Y: B的token序列
        wx: A的token权重数组
        wy: B的token权重数组
    
    Returns:
        (匹配分数, 匹配对列表)
        匹配对格式：[(p, q), ...] 其中p是X的索引(1-based)，q是Y的索引(1-based)
    """
    n, m = len(X), len(Y)
    
    # DP矩阵
    L = np.zeros((n + 1, m + 1), dtype=float)
    # 回溯指针：0=上方(删除), 1=左方(插入), 2=左上(匹配)
    P = np.zeros((n + 1, m + 1), dtype=int)
    
    # 填表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if X[i-1] == Y[j-1]:  # 匹配
                w_match = 0.5 * (wx[i-1] + wy[j-1])
                match_score = L[i-1][j-1] + w_match
                delete_score = L[i-1][j]
                insert_score = L[i][j-1]
                
                if match_score >= delete_score and match_score >= insert_score:
                    L[i][j] = match_score
                    P[i][j] = 2  # 匹配
                elif delete_score >= insert_score:
                    L[i][j] = delete_score
                    P[i][j] = 0  # 删除
                else:
                    L[i][j] = insert_score
                    P[i][j] = 1  # 插入
            else:  # 不匹配
                if L[i-1][j] >= L[i][j-1]:
                    L[i][j] = L[i-1][j]
                    P[i][j] = 0  # 删除
                else:
                    L[i][j] = L[i][j-1]
                    P[i][j] = 1  # 插入
    
    # 回溯匹配对
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if P[i][j] == 2:  # 匹配
            pairs.append((i, j))  # 1-based索引
            i -= 1
            j -= 1
        elif P[i][j] == 0:  # 删除
            i -= 1
        else:  # 插入
            j -= 1
    
    pairs.reverse()
    return float(L[n][m]), pairs


def optimal_step_alignment(
    A_steps: List[Tuple[int, int]],
    B_length: int,
    token_pairs: List[Tuple[int, int]],
    wx: np.ndarray,
    wy: np.ndarray,
    verbose: bool = False
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    找到B的最优k段切分，使得与A的k个步骤匹配分数最大
    
    Args:
        A_steps: A的步骤切分 [(L_1, R_1), ..., (L_k, R_k)]，1-based索引
        B_length: B的token总数
        token_pairs: token级匹配对 [(p, q), ...]，1-based索引
        wx: A的token权重
        wy: B的token权重
        verbose: 是否打印调试信息
    
    Returns:
        (B的最优切分, 最大匹配分数)
        B的切分格式：[(U_1, V_1), ..., (U_k, V_k)]，1-based索引
    """
    k = len(A_steps)
    n = B_length
    
    if verbose:
        print(f"\n=== 步骤对齐开始 ===")
        print(f"A有{k}个步骤: {A_steps}")
        print(f"B有{n}个token，待切分为{k}个步骤")
        print(f"Token匹配对数: {len(token_pairs)}")
    
    # ========== 步骤1：预处理Match函数（前缀和优化）==========
    # PrefixMatch[i][q] = A步骤i与B前q个token的累积匹配权重
    PrefixMatch = np.zeros((k, n + 1), dtype=float)
    
    # 遍历所有匹配对
    for (p, q) in token_pairs:
        w = 0.5 * (wx[p-1] + wy[q-1])
        
        # 找p属于哪个A步骤
        for step_idx in range(k):
            L, R = A_steps[step_idx]
            if L <= p <= R:
                PrefixMatch[step_idx][q] += w
                break
    
    # 累积前缀和
    for step_idx in range(k):
        for q in range(1, n + 1):
            PrefixMatch[step_idx][q] += PrefixMatch[step_idx][q-1]
    
    # 查询函数：O(1)时间
    def Match(step_idx: int, u: int, v: int) -> float:
        """A的第step_idx个步骤(0-indexed)与B区间[u,v](1-indexed)的匹配分数"""
        if u > v or u < 1 or v > n or step_idx < 0 or step_idx >= k:
            return 0.0
        return PrefixMatch[step_idx][v] - (PrefixMatch[step_idx][u-1] if u > 1 else 0.0)
    
    if verbose:
        print("\n前缀和矩阵构建完成")
        print("示例查询 Match(步骤0, B[1,3]):", Match(0, 1, 3))
    
    # ========== 步骤2：区间划分DP ==========
    INF = float('-inf')
    # F[i][j] = A的前i个步骤对应B的前j个token的最大匹配分数
    F = np.full((k + 1, n + 1), INF, dtype=float)
    # Split[i][j] = F[i][j]的最优切分点
    Split = np.full((k + 1, n + 1), -1, dtype=int)
    
    # 边界
    F[0][0] = 0.0
    
    # DP填表
    for i in range(1, k + 1):  # A的前i个步骤
        for j in range(i, n + 1):  # B的前j个token（至少需要i个token才能切i段）
            # 枚举最后一个切分点
            for last in range(i - 1, j):
                # A的第i个步骤(0-indexed是i-1)对应B的[last+1, j]
                if F[i-1][last] > INF:  # 前置状态有效
                    match_score = Match(i - 1, last + 1, j)
                    total = F[i-1][last] + match_score
                    
                    if total > F[i][j]:
                        F[i][j] = total
                        Split[i][j] = last
    
    if verbose:
        print(f"\nDP填表完成")
        print(f"最优分数 F[{k}][{n}] = {F[k][n]:.4f}")
    
    # ========== 步骤3：回溯切分点 ==========
    if F[k][n] <= INF:
        if verbose:
            print("无法找到有效切分！")
        return None, INF
    
    B_steps = []
    i, j = k, n
    while i > 0:
        last = Split[i][j]
        # 第i个A步骤对应B的[last+1, j]
        B_steps.append((last + 1, j))
        
        if verbose:
            match_score = Match(i - 1, last + 1, j)
            print(f"  A步骤{i} {A_steps[i-1]} ↔ B段[{last+1}, {j}]  (匹配分数={match_score:.4f})")
        
        j = last
        i -= 1
    
    B_steps.reverse()
    
    if verbose:
        print(f"\n=== 对齐完成 ===")
        print(f"B的最优切分: {B_steps}")
    
    return B_steps, float(F[k][n])


def visualize_alignment(
    X: List[str],
    Y: List[str],
    A_steps: List[Tuple[int, int]],
    B_steps: List[Tuple[int, int]],
    token_pairs: List[Tuple[int, int]]
):
    """可视化对齐结果"""
    print("\n" + "="*80)
    print("对齐可视化".center(80))
    print("="*80)
    
    k = len(A_steps)
    for i in range(k):
        L_A, R_A = A_steps[i]
        L_B, R_B = B_steps[i]
        
        # A的步骤内容
        A_tokens = X[L_A-1:R_A]
        A_str = " ".join(A_tokens)
        
        # B的步骤内容
        B_tokens = Y[L_B-1:R_B]
        B_str = " ".join(B_tokens)
        
        # 这两个步骤之间的匹配对
        matches_in_step = []
        for (p, q) in token_pairs:
            if L_A <= p <= R_A and L_B <= q <= R_B:
                matches_in_step.append((X[p-1], Y[q-1]))
        
        print(f"\n【步骤 {i+1}】")
        print(f"  A[{L_A:2d}-{R_A:2d}]: {A_str}")
        print(f"  B[{L_B:2d}-{R_B:2d}]: {B_str}")
        print(f"  匹配对数: {len(matches_in_step)}")
        if matches_in_step:
            print(f"  匹配详情: {matches_in_step[:5]}" + (" ..." if len(matches_in_step) > 5 else ""))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("步骤对齐算法测试\n")
    
    # ========== 测试用例1：简单数学推理 ==========
    print("\n" + "="*80)
    print("测试用例1：数学推理步骤对齐")
    print("="*80)
    
    # A的回复（已切分为3个步骤）
    X = ["设", "x", "为", "变量",  # 步骤1
         "移项", "得到", "2x=8",  # 步骤2
         "因此", "x=4"]            # 步骤3
    
    A_steps = [
        (1, 4),   # 步骤1: "设x为变量"
        (5, 7),   # 步骤2: "移项得到2x=8"
        (8, 9)    # 步骤3: "因此x=4"
    ]
    
    # B的回复（未切分）
    Y = ["首先", "设", "x", "然后", "移项", "整理", 
         "得到", "2x", "等于", "8", "所以", "x", "是", "4"]
    
    # Token权重（简单起见，都设为1，实际可以用熵）
    wx = np.ones(len(X))
    wy = np.ones(len(Y))
    
    # 计算token级LCS
    print("\n[1] 计算Token级加权LCS...")
    lcs_score, token_pairs = weighted_lcs(X, Y, wx, wy)
    print(f"LCS匹配分数: {lcs_score:.4f}")
    print(f"匹配对 ({len(token_pairs)}个):")
    for (p, q) in token_pairs:
        print(f"  ({p:2d}, {q:2d}): '{X[p-1]}' ↔ '{Y[q-1]}'")
    
    # 找B的最优切分
    print("\n[2] 找B的最优切分...")
    B_steps, align_score = optimal_step_alignment(
        A_steps, len(Y), token_pairs, wx, wy, verbose=True
    )
    
    # 可视化
    visualize_alignment(X, Y, A_steps, B_steps, token_pairs)
    
    # ========== 测试用例2：代码推理 ==========
    print("\n\n" + "="*80)
    print("测试用例2：代码推理步骤对齐")
    print("="*80)
    
    # A的回复（算法步骤）
    X2 = ["初始化", "数组", "arr",       # 步骤1
          "遍历", "每个", "元素",         # 步骤2
          "累加", "到", "sum",           # 步骤3
          "返回", "结果"]                # 步骤4
    
    A_steps2 = [
        (1, 3),    # 初始化
        (4, 6),    # 遍历
        (7, 9),    # 累加
        (10, 11)   # 返回
    ]
    
    # B的回复（不同表述）
    Y2 = ["先", "初始化", "一个", "arr", "数组", 
          "接着", "for", "循环", "遍历", "所有", "元素",
          "把", "每个", "累加", "sum", "变量",
          "最后", "return", "sum"]
    
    wx2 = np.ones(len(X2))
    wy2 = np.ones(len(Y2))
    
    print("\n[1] 计算Token级LCS...")
    lcs_score2, token_pairs2 = weighted_lcs(X2, Y2, wx2, wy2)
    print(f"LCS匹配分数: {lcs_score2:.4f}")
    print(f"匹配对数: {len(token_pairs2)}")
    
    print("\n[2] 找B的最优切分...")
    B_steps2, align_score2 = optimal_step_alignment(
        A_steps2, len(Y2), token_pairs2, wx2, wy2, verbose=True
    )
    
    visualize_alignment(X2, Y2, A_steps2, B_steps2, token_pairs2)
    
    # ========== 测试用例3：带权重（模拟熵） ==========
    print("\n\n" + "="*80)
    print("测试用例3：带熵权重的对齐")
    print("="*80)
    
    X3 = ["关键", "推理", "步骤", "结论"]
    A_steps3 = [(1, 2), (3, 4)]
    
    Y3 = ["首先", "关键", "的", "推理", "然后", "步骤", "最后", "结论"]
    
    # 模拟熵权重（关键词高熵，停用词低熵）
    wx3 = np.array([2.0, 2.5, 1.5, 2.0])  # "关键"、"推理"权重高
    wy3 = np.array([0.5, 2.0, 0.3, 2.5, 0.5, 1.5, 0.5, 2.0])
    
    print("\n[1] Token权重（模拟熵）:")
    print(f"X权重: {wx3}")
    print(f"Y权重: {wy3}")
    
    lcs_score3, token_pairs3 = weighted_lcs(X3, Y3, wx3, wy3)
    print(f"\nLCS匹配分数: {lcs_score3:.4f}")
    
    B_steps3, align_score3 = optimal_step_alignment(
        A_steps3, len(Y3), token_pairs3, wx3, wy3, verbose=True
    )
    
    visualize_alignment(X3, Y3, A_steps3, B_steps3, token_pairs3)
    
    # ========== 性能测试 ==========
    print("\n\n" + "="*80)
    print("性能测试")
    print("="*80)
    
    import time
    
    # 模拟较长序列
    n_large = 100
    k_large = 5
    X_large = [f"token_A_{i}" for i in range(n_large)]
    Y_large = [f"token_B_{i}" for i in range(n_large)]
    
    # 随机切分A
    step_size = n_large // k_large
    A_steps_large = [(i*step_size+1, (i+1)*step_size) for i in range(k_large)]
    A_steps_large[-1] = (A_steps_large[-1][0], n_large)  # 最后一步到末尾
    
    # 随机生成匹配对
    np.random.seed(42)
    n_pairs = 40
    token_pairs_large = []
    for _ in range(n_pairs):
        p = np.random.randint(1, n_large + 1)
        q = np.random.randint(1, n_large + 1)
        token_pairs_large.append((p, q))
    
    wx_large = np.random.uniform(0.5, 2.0, n_large)
    wy_large = np.random.uniform(0.5, 2.0, n_large)
    
    print(f"\n序列长度: A={n_large}, B={n_large}")
    print(f"步骤数: k={k_large}")
    print(f"匹配对数: {n_pairs}")
    
    start = time.time()
    B_steps_large, score_large = optimal_step_alignment(
        A_steps_large, n_large, token_pairs_large, wx_large, wy_large, verbose=False
    )
    elapsed = time.time() - start
    
    print(f"\n运行时间: {elapsed:.4f}秒")
    print(f"最优分数: {score_large:.4f}")
    print(f"B的切分: {B_steps_large}")
    
    print("\n" + "="*80)
    print("所有测试完成！")
    print("="*80)

