# Step-Level Alignment Algorithm for Process Reward in GRPO

## 1. Introduction

In Group Relative Policy Optimization (GRPO), we need to compute step-level process rewards to guide the model's reasoning process. Given a ground truth response with pre-defined step boundaries and a negative response without step boundaries, we propose a step alignment algorithm that automatically aligns the negative response to the ground truth steps, enabling fine-grained process reward computation.

## 2. Problem Formulation

### 2.1 Input

- **Ground Truth Response** $A$: A sequence of tokens $A = [a_1, a_2, \ldots, a_m]$ with $k$ pre-defined steps:
  $$
  A_{\text{steps}} = \{(L_1, R_1), (L_2, R_2), \ldots, (L_k, R_k)\}
  $$
  where $(L_i, R_i)$ denotes the token range of step $i$ (1-indexed).

- **Negative Response** $B$: A sequence of tokens $B = [b_1, b_2, \ldots, b_n]$ without step boundaries.

- **Token Matching Pairs** $\mathcal{P}$: A set of matched token pairs between $A$ and $B$:
  $$
  \mathcal{P} = \{(p, q) \mid a_p \text{ matches } b_q\}
  $$
  where $p$ and $q$ are 1-indexed positions in $A$ and $B$ respectively.

- **Token Weights** $w_A$ and $w_B$: Weight vectors for tokens in $A$ and $B$ (e.g., entropy-based weights).

### 2.2 Output

- **Optimal Step Segmentation** $B_{\text{steps}}$: A segmentation of $B$ into $k$ steps:
  $$
  B_{\text{steps}} = \{(U_1, V_1), (U_2, V_2), \ldots, (U_k, V_k)\}
  $$
  where $(U_i, V_i)$ denotes the token range of step $i$ in $B$.

- **Maximum Alignment Score** $S^*$: The maximum matching score between $A$'s steps and $B$'s aligned steps.

### 2.3 Objective

Find the optimal segmentation of $B$ that maximizes the step-level alignment score:
$$
B_{\text{steps}}^* = \arg\max_{B_{\text{steps}}} \sum_{i=1}^{k} \text{Match}(i-1, U_i, V_i)
$$

where $\text{Match}(i, u, v)$ computes the matching score between step $i$ of $A$ and tokens $[u, v]$ of $B$.

## 3. Algorithm Description

### 3.1 Token-Level Matching

First, we compute the weighted Longest Common Subsequence (LCS) between $A$ and $B$ to obtain token matching pairs. The matching weight for a pair $(p, q)$ is:
$$
w(p, q) = \frac{1}{2}(w_A[p-1] + w_B[q-1])
$$

The LCS is computed using dynamic programming with the following recurrence:
$$
L[i][j] = \begin{cases}
\max(L[i-1][j-1] + w(i-1, j-1), L[i-1][j], L[i][j-1]) & \text{if } a_i = b_j \\
\max(L[i-1][j], L[i][j-1]) & \text{otherwise}
\end{cases}
$$

where $L[i][j]$ represents the maximum weighted LCS score between $A[1:i]$ and $B[1:j]$.

### 3.2 Prefix Sum Optimization

To enable efficient interval matching queries, we build a prefix sum matrix $\text{PrefixMatch}[i][q]$ that stores the cumulative matching weight between step $i$ of $A$ and the first $q$ tokens of $B$:

**Initialization:**
$$
\text{PrefixMatch}[i][q] = \sum_{(p, q') \in \mathcal{P}, q' \leq q, p \in \text{step}_i} w(p, q')
$$

**Prefix Sum Computation:**
$$
\text{PrefixMatch}[i][q] = \text{PrefixMatch}[i][q-1] + \text{PrefixMatch}[i][q], \quad \forall q \geq 1
$$

**Interval Query Function:**
$$
\text{Match}(i, u, v) = \text{PrefixMatch}[i][v] - \text{PrefixMatch}[i][u-1]
$$

This allows $O(1)$ time complexity for computing the matching score between step $i$ and any interval $[u, v]$ in $B$.

### 3.3 Dynamic Programming for Optimal Segmentation

We use dynamic programming to find the optimal segmentation. Let $F[i][j]$ denote the maximum alignment score when aligning the first $i$ steps of $A$ to the first $j$ tokens of $B$.

**State Definition:**
$$
F[i][j] = \max_{\text{segmentation}} \sum_{t=1}^{i} \text{Match}(t-1, U_t, V_t)
$$

**Recurrence Relation:**
$$
F[i][j] = \max_{i-1 \leq \text{last} < j} \left( F[i-1][\text{last}] + \text{Match}(i-1, \text{last}+1, j) \right)
$$

**Boundary Conditions:**
$$
\begin{align}
F[0][0] &= 0 \\
F[i][j] &= -\infty \quad \text{if } i > j
\end{align}
$$

**Explanation:**
- $F[i-1][\text{last}]$: Maximum score for aligning the first $i-1$ steps to the first $\text{last}$ tokens.
- $\text{Match}(i-1, \text{last}+1, j)$: Matching score between step $i-1$ of $A$ and tokens $[\text{last}+1, j]$ of $B$.
- We enumerate all possible split points $\text{last}$ and choose the one that maximizes the total score.

### 3.4 Backtracking

After computing $F[k][n]$, we backtrack to recover the optimal segmentation:

**Algorithm:**
1. Initialize $i = k$, $j = n$.
2. While $i > 0$:
   - Let $\text{last} = \text{Split}[i][j]$ (the optimal split point stored during DP).
   - Step $i-1$ corresponds to $B[\text{last}+1, j]$.
   - Set $j = \text{last}$, $i = i - 1$.
3. Reverse the segmentation list to get the correct order.

## 4. Algorithm Pseudocode

```
Algorithm: Step-Level Alignment
Input: A_steps, B_length, token_pairs, w_A, w_B
Output: B_steps, alignment_score

1. // Build prefix sum matrix
2. PrefixMatch = zeros(k, n+1)
3. for each (p, q) in token_pairs:
4.     w = 0.5 * (w_A[p-1] + w_B[q-1])
5.     step_idx = find_step(p, A_steps)
6.     PrefixMatch[step_idx][q] += w
7. 
8. // Compute prefix sums
9. for step_idx in range(k):
10.    for q in range(1, n+1):
11.        PrefixMatch[step_idx][q] += PrefixMatch[step_idx][q-1]
12.
13. // Define Match function
14. Match(step_idx, u, v) = PrefixMatch[step_idx][v] - PrefixMatch[step_idx][u-1]
15.
16. // Dynamic programming
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
28. // Backtracking
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

## 5. Complexity Analysis

- **Time Complexity:**
  - Token matching (LCS): $O(m \times n)$
  - Prefix sum construction: $O(k \times n)$
  - DP table filling: $O(k \times n^2)$ (dominant term)
  - Backtracking: $O(k)$
  - **Total: $O(k \times n^2)$**

- **Space Complexity:**
  - Prefix sum matrix: $O(k \times n)$
  - DP tables: $O(k \times n)$
  - **Total: $O(k \times n)$**

## 6. Application to GRPO Process Reward

### 6.1 Step-Level Reward Computation

After aligning the negative response to the ground truth steps, we compute step-level process rewards:

$$
r_{\text{step}}(i) = \frac{\text{Match}(i, U_i, V_i)}{|A_i|}
$$

where $|A_i|$ is the number of tokens in step $i$ of the ground truth.

### 6.2 Process Reward Aggregation

The overall process reward for the negative response is:

$$
R_{\text{process}} = \frac{1}{k} \sum_{i=1}^{k} r_{\text{step}}(i)
$$

This reward reflects how well the negative response follows the reasoning steps of the ground truth, providing fine-grained feedback for the model's reasoning process.

### 6.3 Integration with GRPO

In GRPO, we use this step-level alignment to:
1. **Compare reasoning processes**: Align negative responses to ground truth steps to identify where the reasoning diverges.
2. **Compute process rewards**: Assign rewards based on step-level alignment scores.
3. **Guide policy optimization**: Use process rewards to encourage the model to follow correct reasoning patterns step by step.

## 7. Advantages

1. **Automatic Alignment**: No manual annotation required for negative responses.
2. **Flexible Segmentation**: Handles cases where negative responses have different token counts per step.
3. **Efficient Computation**: Prefix sum optimization enables $O(1)$ interval queries.
4. **Optimal Solution**: Dynamic programming guarantees finding the globally optimal alignment.
5. **Weighted Matching**: Supports token-level weights (e.g., entropy-based) for more nuanced alignment.

## 8. Example

**Ground Truth** $A$ with 3 steps:
- Step 1: "设 x 为变量" (tokens 1-4)
- Step 2: "移项 得到 2x=8" (tokens 5-7)
- Step 3: "因此 x=4" (tokens 8-9)

**Negative Response** $B$: "首先 设 x 然后 移项 整理 得到 2x 等于 8 所以 x 是 4"

**Aligned Segmentation** $B_{\text{steps}}$:
- Step 1: $B[1,3]$ → "首先 设 x"
- Step 2: $B[4,7]$ → "然后 移项 整理 得到"
- Step 3: $B[8,14]$ → "2x 等于 8 所以 x 是 4"

**Step-Level Rewards**:
- $r_{\text{step}}(1) = 2/4 = 0.5$ (2 matched tokens out of 4)
- $r_{\text{step}}(2) = 2/3 = 0.67$ (2 matched tokens out of 3)
- $r_{\text{step}}(3) = 0/2 = 0.0$ (0 matched tokens out of 2)

**Process Reward**: $R_{\text{process}} = (0.5 + 0.67 + 0.0) / 3 = 0.39$

## 9. Conclusion

The step-level alignment algorithm provides an effective way to align negative responses to ground truth reasoning steps, enabling fine-grained process reward computation in GRPO. By leveraging dynamic programming and prefix sum optimization, the algorithm efficiently finds the optimal alignment while maintaining computational tractability.

