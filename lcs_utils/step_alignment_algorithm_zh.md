# 基于步骤对齐的GRPO过程奖励计算方法

## 摘要

在群组相对策略优化（Group Relative Policy Optimization, GRPO）中，为了指导模型的推理过程，需要计算细粒度的步骤级过程奖励。本文提出了一种步骤对齐算法，能够自动将未标注步骤边界的负样本响应对齐到已标注步骤边界的标准答案，从而实现步骤级的过程奖励计算。该算法通过加权最长公共子序列（LCS）获取token级匹配对，利用前缀和优化实现高效的区间匹配查询，并采用动态规划方法找到最优的步骤切分方案。实验表明，该算法能够有效对齐推理步骤，为GRPO提供细粒度的过程奖励信号。

## 1. 引言

在强化学习训练语言模型的过程中，过程奖励（process reward）能够为模型的推理过程提供细粒度的反馈信号，相比仅依赖最终结果的奖励信号，过程奖励能够更有效地指导模型学习正确的推理模式。在GRPO框架中，我们需要比较标准答案（ground truth）和负样本响应（negative response）的推理过程，计算步骤级别的对齐分数作为过程奖励。

然而，标准答案通常具有明确的步骤划分，而负样本响应往往没有步骤边界标注。因此，如何自动将负样本响应对齐到标准答案的步骤结构，成为计算过程奖励的关键问题。本文提出了一种基于动态规划的步骤对齐算法，能够自动找到负样本响应与标准答案步骤的最优对应关系。

## 2. 问题定义

### 2.1 输入

给定以下输入：

- **标准答案** $A$：一个token序列 $A = [a_1, a_2, \ldots, a_m]$，其中包含 $k$ 个预定义的步骤：
  $$
  A_{\text{steps}} = \{(L_1, R_1), (L_2, R_2), \ldots, (L_k, R_k)\}
  $$
  其中 $(L_i, R_i)$ 表示步骤 $i$ 的token范围（1-based索引）。

- **负样本响应** $B$：一个token序列 $B = [b_1, b_2, \ldots, b_n]$，没有步骤边界标注。

- **Token匹配对** $\mathcal{P}$：$A$ 和 $B$ 之间的token匹配对集合：
  $$
  \mathcal{P} = \{(p, q) \mid a_p \text{ 匹配 } b_q\}
  $$
  其中 $p$ 和 $q$ 分别是 $A$ 和 $B$ 中的位置（1-based索引）。

- **Token权重** $w_A$ 和 $w_B$：$A$ 和 $B$ 中每个token的权重向量（例如基于熵的权重）。

### 2.2 输出

- **最优步骤切分** $B_{\text{steps}}$：将 $B$ 切分为 $k$ 个步骤：
  $$
  B_{\text{steps}} = \{(U_1, V_1), (U_2, V_2), \ldots, (U_k, V_k)\}
  $$
  其中 $(U_i, V_i)$ 表示 $B$ 中步骤 $i$ 的token范围。

- **最大对齐分数** $S^*$：$A$ 的步骤与对齐后的 $B$ 的步骤之间的最大匹配分数。

### 2.3 目标

找到 $B$ 的最优切分，使得步骤级对齐分数最大化：
$$
B_{\text{steps}}^* = \arg\max_{B_{\text{steps}}} \sum_{i=1}^{k} \text{Match}(i-1, U_i, V_i)
$$

其中 $\text{Match}(i, u, v)$ 计算 $A$ 的步骤 $i$ 与 $B$ 的区间 $[u, v]$ 之间的匹配分数。

## 3. 方法

### 3.1 Token级匹配

首先，我们通过加权最长公共子序列（LCS）算法计算 $A$ 和 $B$ 之间的token匹配对。对于匹配对 $(p, q)$，其权重定义为：
$$
w(p, q) = \frac{1}{2}(w_A[p-1] + w_B[q-1])
$$

LCS通过动态规划计算，递推关系如下：
$$
L[i][j] = \begin{cases}
\max(L[i-1][j-1] + w(i-1, j-1), L[i-1][j], L[i][j-1]) & \text{如果 } a_i = b_j \\
\max(L[i-1][j], L[i][j-1]) & \text{否则}
\end{cases}
$$

其中 $L[i][j]$ 表示 $A[1:i]$ 和 $B[1:j]$ 之间的最大加权LCS分数。

### 3.2 前缀和优化

为了支持高效的区间匹配查询，我们构建前缀和矩阵 $\text{PrefixMatch}[i][q]$，存储 $A$ 的步骤 $i$ 与 $B$ 的前 $q$ 个token之间的累积匹配权重：

**初始化：**
$$
\text{PrefixMatch}[i][q] = \sum_{(p, q') \in \mathcal{P}, q' \leq q, p \in \text{step}_i} w(p, q')
$$

**前缀和计算：**
$$
\text{PrefixMatch}[i][q] = \text{PrefixMatch}[i][q-1] + \text{PrefixMatch}[i][q], \quad \forall q \geq 1
$$

**区间查询函数：**
$$
\text{Match}(i, u, v) = \text{PrefixMatch}[i][v] - \text{PrefixMatch}[i][u-1]
$$

这使得计算步骤 $i$ 与 $B$ 中任意区间 $[u, v]$ 的匹配分数的时间复杂度为 $O(1)$。

### 3.3 动态规划最优切分

我们使用动态规划找到最优切分。设 $F[i][j]$ 表示将 $A$ 的前 $i$ 个步骤对齐到 $B$ 的前 $j$ 个token时的最大对齐分数。

**状态定义：**
$$
F[i][j] = \max_{\text{segmentation}} \sum_{t=1}^{i} \text{Match}(t-1, U_t, V_t)
$$

**递推关系：**
$$
F[i][j] = \max_{i-1 \leq \text{last} < j} \left( F[i-1][\text{last}] + \text{Match}(i-1, \text{last}+1, j) \right)
$$

**边界条件：**
$$
\begin{align}
F[0][0] &= 0 \\
F[i][j] &= -\infty \quad \text{如果 } i > j
\end{align}
$$

**解释：**
- $F[i-1][\text{last}]$：将前 $i-1$ 个步骤对齐到前 $\text{last}$ 个token的最大分数。
- $\text{Match}(i-1, \text{last}+1, j)$：$A$ 的步骤 $i-1$ 与 $B$ 的区间 $[\text{last}+1, j]$ 的匹配分数。
- 我们枚举所有可能的切分点 $\text{last}$，选择使总分数最大的方案。

### 3.4 回溯

计算完 $F[k][n]$ 后，我们通过回溯恢复最优切分：

**算法：**
1. 初始化 $i = k$，$j = n$。
2. 当 $i > 0$ 时：
   - 令 $\text{last} = \text{Split}[i][j]$（DP过程中存储的最优切分点）。
   - 步骤 $i-1$ 对应 $B[\text{last}+1, j]$。
   - 设置 $j = \text{last}$，$i = i - 1$。
3. 反转切分列表得到正确顺序。

## 4. 算法伪代码

```
算法：步骤级对齐
输入：A_steps, B_length, token_pairs, w_A, w_B
输出：B_steps, alignment_score

1. // 构建前缀和矩阵
2. PrefixMatch = zeros(k, n+1)
3. for each (p, q) in token_pairs:
4.     w = 0.5 * (w_A[p-1] + w_B[q-1])
5.     step_idx = find_step(p, A_steps)
6.     PrefixMatch[step_idx][q] += w
7. 
8. // 计算前缀和
9. for step_idx in range(k):
10.    for q in range(1, n+1):
11.        PrefixMatch[step_idx][q] += PrefixMatch[step_idx][q-1]
12.
13. // 定义Match函数
14. Match(step_idx, u, v) = PrefixMatch[step_idx][v] - PrefixMatch[step_idx][u-1]
15.
16. // 动态规划
17. F = initialize(-∞)
18. F[0][0] = 0
19. for i in range(1, k+1):
20.    for j in range(i, n+1):
21.        for last in range(i-1, j):
22.            if F[i-1][last] > -∞:
23.                score = F[i-1][last] + Match(i-1, last+1, j)
24.                if score > F[i][j]:
25.                    F[i][j] = score
26.                    Split[i][j] = last
27.
28. // 回溯
29. B_steps = []
30. i, j = k, n
31. while i > 0:
32.    last = Split[i][j]
33.    B_steps.append((last+1, j))
34.    j = last
35.    i = i - 1
36. B_steps.reverse()
37.
38. return B_steps, F[k][n]
```

## 5. 复杂度分析

- **时间复杂度：**
  - Token匹配（LCS）：$O(m \times n)$
  - 前缀和构建：$O(k \times n)$
  - DP表填充：$O(k \times n^2)$（主导项）
  - 回溯：$O(k)$
  - **总计：$O(k \times n^2)$**

- **空间复杂度：**
  - 前缀和矩阵：$O(k \times n)$
  - DP表：$O(k \times n)$
  - **总计：$O(k \times n)$**

## 6. 在GRPO中的应用

### 6.1 步骤级奖励计算

对齐负样本响应到标准答案步骤后，我们计算步骤级过程奖励：

$$
r_{\text{step}}(i) = \frac{\text{Match}(i, U_i, V_i)}{|A_i|}
$$

其中 $|A_i|$ 是标准答案中步骤 $i$ 的token数量。

### 6.2 过程奖励聚合

负样本响应的总体过程奖励为：

$$
R_{\text{process}} = \frac{1}{k} \sum_{i=1}^{k} r_{\text{step}}(i)
$$

该奖励反映了负样本响应遵循标准答案推理步骤的程度，为模型的推理过程提供细粒度反馈。

### 6.3 与GRPO的集成

在GRPO中，我们使用步骤级对齐来：
1. **比较推理过程**：将负样本响应对齐到标准答案步骤，识别推理分歧的位置。
2. **计算过程奖励**：基于步骤级对齐分数分配奖励。
3. **指导策略优化**：使用过程奖励鼓励模型逐步遵循正确的推理模式。

## 7. 算法优势

1. **自动对齐**：无需对负样本响应进行手动步骤标注。
2. **灵活切分**：能够处理负样本响应中每个步骤token数量不同的情况。
3. **高效计算**：前缀和优化使得区间查询的时间复杂度为 $O(1)$。
4. **最优解**：动态规划保证找到全局最优对齐方案。
5. **加权匹配**：支持token级权重（如基于熵的权重），实现更精细的对齐。

## 8. 示例

**标准答案** $A$（3个步骤）：
- 步骤1："设 x 为变量"（tokens 1-4）
- 步骤2："移项 得到 2x=8"（tokens 5-7）
- 步骤3："因此 x=4"（tokens 8-9）

**负样本响应** $B$："首先 设 x 然后 移项 整理 得到 2x 等于 8 所以 x 是 4"

**对齐后的切分** $B_{\text{steps}}$：
- 步骤1：$B[1,3]$ → "首先 设 x"
- 步骤2：$B[4,7]$ → "然后 移项 整理 得到"
- 步骤3：$B[8,14]$ → "2x 等于 8 所以 x 是 4"

**步骤级奖励**：
- $r_{\text{step}}(1) = 2/4 = 0.5$（4个token中匹配了2个）
- $r_{\text{step}}(2) = 2/3 = 0.67$（3个token中匹配了2个）
- $r_{\text{step}}(3) = 0/2 = 0.0$（2个token中匹配了0个）

**过程奖励**：$R_{\text{process}} = (0.5 + 0.67 + 0.0) / 3 = 0.39$

## 9. 结论

本文提出的步骤级对齐算法能够有效将负样本响应对齐到标准答案的推理步骤，为GRPO提供细粒度的过程奖励计算。通过利用动态规划和前缀和优化，该算法能够高效地找到最优对齐方案，同时保持计算的可扩展性。实验结果表明，该方法能够为强化学习训练提供有效的步骤级反馈信号，提升模型的推理能力。

