# Pair-GRPO Family 精读

阅读日期：2026-06-01  
论文标题：A Unified Pair-GRPO Family: From Implicit to Explicit Preference Constraints for Stable and General RL Alignment  
作者：Hao Yu  
机构：Department of Automation, Tsinghua University  
arXiv：2605.06375v1，2026-05-07  
链接：[arXiv Abs](https://arxiv.org/abs/2605.06375)，[HTML](https://arxiv.org/html/2605.06375)，[PDF](https://arxiv.org/pdf/2605.06375)  
资料版本：arXiv PDF，SHA256 `afe7d39c9de89cbaaf78d6c399cb54b694370dfe665a71be005f9611df9b7386`  
源码版本：arXiv e-print，SHA256 `94ef19a29eaf9a1b6ecdc6dd6d34da9ee215fb2cdf841a33f6e689b3ec4ed3e1`

## 一句话结论

这篇论文提出了一个 **Pair-GRPO family**，试图把 GRPO 的 group-normalized scalar reward 改成 pairwise preference signal。它分成两个版本：

1. **Soft-Pair-GRPO**：把偏好对 `(a_p, a_r)` 直接编码成 `+1/-1`，仍然使用 GRPO/PPO 风格的 clipped surrogate 和 KL regularization。
2. **Hard-Pair-GRPO**：不再只给 reward，而是显式构造一个 target policy distribution，把概率质量从 rejected response 转给 preferred response，再做带 trust-region 的 KL fitting。

论文最值得读的地方是它把“偏好对齐中的相对偏好信号”和“GRPO trust-region 更新”连接起来。但我对它的理论严谨性比较谨慎：梯度同向这个核心想法是合理的；可是文中关于“严格方差降低”“单调改进保证”“irrelevant responses 梯度为零”等表述需要额外假设，正文和补充材料里也存在比例常数方向、方差协方差符号、代码可复现性等不够严密的地方。

## 1. 背景：为什么要改 GRPO

传统 RLHF 大致是：

```text
SFT
-> reward model from human preference pairs
-> PPO/GRPO 等 RL 算法最大化 reward
```

GRPO 的核心是对同一个 prompt 采样一组 responses：

```text
s -> {a_1, ..., a_K}
```

然后用 reward model 给每个 response 打分，并在组内标准化：

```text
r_i^GRPO = (R(a_i) - mu_R) / sigma_R
```

再用 PPO/GRPO clipped surrogate 更新策略。

论文认为这里有三个问题：

| 问题 | 解释 |
| --- | --- |
| 梯度方向噪声大 | scalar reward 的绝对值、尺度、归一化都会影响更新 |
| 连续 reward 信息冗余 | 人类偏好本质是 pairwise ordering，不一定需要绝对 reward magnitude |
| 全局 policy drift | reward-based objective 可能影响整个 action/response space，不只改变关键偏好对 |

因此论文提出：既然偏好数据本来就是 `a_p ≻ a_r`，就直接让算法围绕 pairwise preference 更新。

## 2. 符号

| 符号 | 含义 |
| --- | --- |
| `s` | prompt / state |
| `a` | response / action |
| `pi_theta(a|s)` | 当前策略 |
| `pi_old(a|s)` | 上一轮冻结的 reference policy |
| `(s, a_p, a_r)` | preference pair，`a_p` preferred，`a_r` rejected |
| `R(a)` | reward model 给 response 的 scalar reward |
| `rho(s,a)` | importance ratio，`pi_theta(a|s) / pi_old(a|s)` |
| `D_KL(P || Q)` | KL divergence |
| `beta` | trust-region KL 阈值或 penalty 系数 |
| `epsilon` | PPO/GRPO clip range |
| `delta_t` | Hard-Pair-GRPO 中概率质量转移步长 |

## 3. 标准 GRPO 的局部形式

论文采用的 GRPO objective 是：

```text
L_GRPO(theta)
= E[min(rho(s,a) r^GRPO,
        clip(rho(s,a), 1-epsilon, 1+epsilon) r^GRPO)]
```

其中：

```text
rho(s,a) = pi_theta(a|s) / pi_old(a|s)
r_i^GRPO = (R(a_i) - mu_R) / sigma_R
```

为了看清理论主线，可以先忽略 clipping 和 KL penalty，只看局部一阶梯度。因为在 `theta = theta_old` 附近：

```text
rho(s,a) = 1
grad_theta rho(s,a)
= grad_theta pi_theta(a|s) / pi_old(a|s)
= rho(s,a) grad_theta log pi_theta(a|s)
≈ grad_theta log pi_theta(a|s)
```

所以局部 policy-gradient 形式是：

```text
grad L_GRPO
≈ E[ r^GRPO(a) grad log pi_theta(a|s) ]
```

这和普通 policy gradient 一致：advantage / normalized reward 决定这条 sample 是增加概率还是降低概率。

## 4. Soft-Pair-GRPO

### 4.1 算法定义

给定偏好对：

```text
(s, a_p, a_r),  a_p ≻ a_r
```

Soft-Pair-GRPO 定义 binary pairwise reward：

```text
r^Soft(a_p) = +1
r^Soft(a_r) = -1
```

对应 objective：

```text
L_Soft(theta)
= E[min(rho(s,a) r^Soft,
        clip(rho(s,a), 1-epsilon, 1+epsilon) r^Soft)]
  + beta D_KL(pi_theta || pi_old)
```

实际更新流程是：

```text
1. 从 pi_old 收集 preference pairs
2. 对 preferred response 给 +1
3. 对 rejected response 给 -1
4. 用 clipped surrogate + KL penalty 更新 pi_theta
5. 同步 pi_old <- pi_theta
```

### 4.2 Soft-Pair-GRPO 的直观含义

它不是训练 reward model，也不是 DPO 那种直接二分类 loss，而是保留 GRPO 的 policy optimization 框架，只把 reward 换成最简单的相对偏好信号。

局部一阶梯度为：

```text
g_Soft
≈ grad log pi_theta(a_p|s) - grad log pi_theta(a_r|s)
```

如果使用梯度上升最大化 objective，它会：

```text
increase pi_theta(a_p|s)
decrease pi_theta(a_r|s)
```

如果代码里把它写成 loss 并做梯度下降，则符号整体反过来，但更新含义不变。

## 5. 理论推导 1：Soft-Pair-GRPO 与 GRPO 的梯度关系

用户要求理论推导，这里把关键定理完整拆开。

### 5.1 两响应 pair group 的最简单情形

假设同一 prompt 下只比较两个 response：

```text
Group = {a_p, a_r}
Delta R = R(a_p) - R(a_r) > 0
```

组内均值：

```text
mu_R = (R(a_p) + R(a_r)) / 2
```

如果用二点总体标准差：

```text
sigma_R
= sqrt( ((R(a_p)-mu_R)^2 + (R(a_r)-mu_R)^2) / 2 )
= |Delta R| / 2
```

于是：

```text
r_p^GRPO
= (R(a_p)-mu_R) / sigma_R
= (Delta R/2) / (Delta R/2)
= +1

r_r^GRPO
= (R(a_r)-mu_R) / sigma_R
= (-Delta R/2) / (Delta R/2)
= -1
```

因此在这个最简单情形下：

```text
g_GRPO = g_Soft
```

也就是说，如果 GRPO 的 group 本身就是一个偏好 pair，组内标准化会自然把 reward 变成 `+1/-1`。这能解释为什么 binary preference signal 可能不会破坏方向。

### 5.2 更一般的 pair 嵌入 group 情形

如果 group 里还有其他 responses，那么 `sigma_R` 是整个 group 的 reward 标准差，不一定等于 `Delta R/2`。

只看 pair 对梯度的贡献：

```text
g_GRPO_pair
≈ ((R(a_p)-mu_R)/sigma_R) grad log pi(a_p|s)
 + ((R(a_r)-mu_R)/sigma_R) grad log pi(a_r|s)
```

为了和 Soft-Pair 对齐，论文主要看 reward difference：

```text
Delta r_GRPO
= r_p^GRPO - r_r^GRPO
= (R(a_p)-R(a_r)) / sigma_R
= Delta R / sigma_R
```

Soft-Pair 的 difference 是：

```text
Delta r_Soft = (+1) - (-1) = 2
```

如果把 pairwise 更新方向定义为：

```text
h = grad log pi(a_p|s) - grad log pi(a_r|s)
```

那么在 pair-centered 近似下：

```text
g_GRPO_pair ≈ (Delta R / (2 sigma_R)) h
g_Soft      = h
```

于是：

```text
g_GRPO_pair ≈ C g_Soft
C = Delta R / (2 sigma_R) > 0
```

等价地：

```text
g_Soft ≈ (1/C) g_GRPO_pair
= (2 sigma_R / Delta R) g_GRPO_pair
```

只要 `Delta R > 0`，两者方向相同。

### 5.3 和论文定理写法的差异

正文的 theorem 写的是：

```text
grad L_Soft ≈ C(pi_old) grad L_GRPO
C = E[(R(a_p)-R(a_r))/(2 sigma_R)] > 0
```

但补充材料推导得到的是：

```text
grad J_GRPO ≈ C grad J_Soft
C = E[Delta R] / (2 sigma_R)
```

从上面的局部推导看，补充材料的方向更自然：

```text
GRPO = C * Soft
Soft = (1/C) * GRPO
```

这个倒置不影响“同向”结论，但会影响“尺度等价”的严格表述。我的判断是：

1. **核心方向性观点成立**：只要 preferred 的 reward 高于 rejected，binary pair signal 和 scalar reward signal 是同向的。
2. **比例常数写法不严谨**：正文和 supplement 的方向不一致，应以“正比例同向”而不是具体 `C` 为主。
3. **加入 KL penalty 后不再是简单整体 scalar multiple**：除非 KL 项也按同一比例缩放，否则 `reward gradient + beta KL gradient` 不是严格的整体正比例。
4. **进入 clipping 区间后等价也会被破坏**：PPO/GRPO clip 会把某些样本梯度截断，所以梯度等价只适合局部未 clipping 的一阶分析。

## 6. 理论推导 2：Soft-Pair-GRPO 的方向确定性

忽略 clipping，Soft objective 的 pair 部分是：

```text
L_pair(theta)
= rho(s,a_p) * (+1) + rho(s,a_r) * (-1)
= rho_p - rho_r
```

局部梯度：

```text
grad L_pair
= grad rho_p - grad rho_r
≈ grad log pi(a_p|s) - grad log pi(a_r|s)
```

梯度上升时：

```text
theta <- theta + eta grad L_pair
```

对 preferred response：

```text
log pi(a_p|s) 增大
pi(a_p|s) 增大
```

对 rejected response：

```text
log pi(a_r|s) 减小
pi(a_r|s) 减小
```

这就是论文所谓 deterministic gradient directionality。

需要注意：这个结论是局部的。真实神经网络参数共享，增加一个 response 的概率可能通过 softmax normalization 和表示共享影响其他 token / response；因此“只影响 pair，不影响其他 response”不是 Soft-Pair 能保证的。

## 7. 理论推导 3：Soft-Pair-GRPO 的方差讨论

论文声称：

```text
Var_Soft < Var_GRPO
```

直觉上，Soft-Pair 的优势是去掉了 reward magnitude 的噪声：

```text
GRPO: advantage depends on reward model scale and group statistics
Soft: advantage fixed as +1/-1
```

如果 reward model 的绝对值很 noisy，而 preference ordering 较可靠，那么 Soft 的确可能降低梯度噪声。

但论文给出的严格证明并不充分。补充材料写：

```text
g = g_p - g_r
Var(g) = Var(g_p) + Var(g_r) - 2 Cov(g_p, g_r)
```

然后说 Soft-Pair 中 `Cov(g_p, g_r) < 0`，所以方差降低。

这里符号上有问题：如果公式是 `Var(g_p - g_r)`，那么当 `Cov(g_p,g_r) < 0` 时：

```text
-2 Cov(g_p,g_r) > 0
```

方差反而变大。要让 `g_p - g_r` 的方差降低，需要 `Cov(g_p,g_r) > 0`。如果把 rejected response 的负号已经吸收到 `g_r` 里，公式又应该写成 `Var(g_p + g_r)`，此时协方差解释也要重新定义。

所以我会这样理解这条理论：

1. **经验直觉合理**：binary pair signal 可能比 noisy scalar reward 稳。
2. **严格不等式不成立为一般定理**：必须假设 reward noise、pair sampling、gradient covariance 的特定结构。
3. **论文证明有符号问题**：不能直接作为严格证明使用。

## 8. Hard-Pair-GRPO

### 8.1 目标分布构造

Hard-Pair-GRPO 想进一步解决 Soft 的问题：Soft 仍然是 reward-weighted policy gradient，可能导致全局 policy drift。

它对每个 preference pair 构造一个 target distribution：

```text
pi_tar(a_p|s) = pi_old(a_p|s) + delta_t
pi_tar(a_r|s) = pi_old(a_r|s) - delta_t
pi_tar(c|s)   = pi_old(c|s),  c not in {a_p, a_r}
```

其中：

```text
delta_t = delta_0 * gamma^t
delta_0 > 0
gamma in (0, 1)
```

也就是显式把一小块概率质量从 rejected response 转给 preferred response。

为了保证这是合法分布，需要：

```text
0 <= pi_old(a_r|s) - delta_t
pi_old(a_p|s) + delta_t <= 1
```

实践中需要 clamp 或自适应缩小 `delta_t`。

### 8.2 KL fitting objective

Hard-Pair-GRPO 不再用 `+1/-1` surrogate，而是做 constrained KL fitting：

```text
min_theta  L_fit(theta) = D_KL(pi_theta || pi_tar)
s.t.       D_KL(pi_theta || pi_old) <= beta
```

论文把约束写成 hinge penalty：

```text
L_total(theta)
= D_KL(pi_theta || pi_tar)
  + alpha max(D_KL(pi_theta || pi_old) - beta, 0)
```

直观含义：

1. 第一项让当前策略靠近目标分布 `pi_tar`。
2. 第二项限制当前策略不要离 `pi_old` 太远。
3. `delta_t` 控制每次只移动一点点概率质量。

### 8.3 KL fitting 的分布层推导

把某个 prompt 下的策略分布记为：

```text
p_i = pi_theta(a_i|s)
q_i = pi_tar(a_i|s)
```

forward KL 是：

```text
D_KL(p || q) = sum_i p_i log(p_i / q_i)
```

如果直接在概率 simplex 上优化，最小值显然在：

```text
p_i = q_i
```

因为：

```text
D_KL(p || q) >= 0
```

且等号当且仅当 `p=q`。

因此 Hard-Pair-GRPO 的目标就是把：

```text
pi_theta(a_p|s) 推向 pi_old(a_p|s)+delta_t
pi_theta(a_r|s) 推向 pi_old(a_r|s)-delta_t
```

其他 action 的目标概率和 old policy 相同。

### 8.4 一阶近似：为什么是局部概率转移

当 `delta_t` 很小时：

```text
q_p = p_old_p + delta
q_r = p_old_r - delta
q_c = p_old_c
```

对 preferred action：

```text
log(p_old_p / q_p)
= log(p_old_p / (p_old_p + delta))
= - log(1 + delta / p_old_p)
≈ - delta / p_old_p
```

对 rejected action：

```text
log(p_old_r / q_r)
= log(p_old_r / (p_old_r - delta))
= - log(1 - delta / p_old_r)
≈ + delta / p_old_r
```

所以在 `p = pi_old` 附近，KL fitting 的信号会让：

```text
preferred: 当前概率低于 target，需要上调
rejected:  当前概率高于 target，需要下调
```

这就是 Hard-Pair-GRPO 比 Soft 更“硬”的地方：不是告诉策略“preferred reward 高”，而是直接给策略一个局部分布目标。

### 8.5 关于“irrelevant responses 梯度为零”的谨慎解释

论文声称 Hard-Pair-GRPO 对无关 response 的梯度理论上为零。这个说法只在一种理想化解释下成立：如果优化器直接操作分布坐标，并且显式冻结 `c not in {a_p,a_r}` 的概率，那么无关 action 的目标不变。

但对真实神经网络策略：

```text
p_i = softmax(z_i)
```

即使 `q_c = p_old_c`，forward KL 对 logits 的梯度是：

```text
partial D_KL(p||q) / partial z_j
= p_j [ log(p_j/q_j) + 1 - E_p(log(p_i/q_i)+1) ]
```

在 `p = p_old`、`q` 只改变 pair 的情况下，对无关 `c`：

```text
log(p_c/q_c) = 0
```

但全局期望项包含 preferred/rejected 的 KL，因此无关 logits 的梯度不一定严格为零。更重要的是，神经网络参数共享会让更新 pair token/response 的同时影响其他 response。

所以更准确的说法是：

```text
Hard-Pair-GRPO 的 target distribution 只显式改变 pair 上的目标概率；
但参数化策略训练中，无关 response 是否完全不变，需要额外冻结、mask 或局部投影机制。
```

## 9. 理论推导 4：Hard-Pair-GRPO 的动态步长

论文使用：

```text
delta_t = delta_0 gamma^t,  gamma in (0,1)
```

显然：

```text
lim_{t->infty} delta_t = 0
```

因此随着训练推进：

```text
pi_tar -> pi_old
```

每一步分布移动越来越小。这有两个效果：

1. **稳定性**：后期不会继续大幅移动概率，减少振荡。
2. **信号衰减**：如果 `delta_t` 太快变小，学习信号也会变弱。

论文把它表述为“稳定收敛且不振荡”。这个方向合理，但严格收敛还需要：

1. 学习率条件。
2. KL penalty 足够强。
3. 目标分布每轮同步的具体机制。
4. preference pair 分布不剧烈变化。
5. 神经网络优化可达局部稳定点。

单靠 `delta_t -> 0` 不能证明算法整体收敛。

## 10. 关于单调改进保证

论文对 Soft 和 Hard 都声称：

```text
J(pi_theta) >= J(pi_old)
```

理由是 trust-region KL constraint + PPO/GRPO monotonic improvement framework。

这里需要非常谨慎。严格 TRPO 单调改进定理依赖：

1. 准确 advantage。
2. 明确 surrogate lower bound。
3. 受控的最大 KL 或 TV distance。
4. 优化问题足够精确地求解。

PPO clipping 和 KL penalty 本身并不自动给出每一步真实 return 的严格单调提升。论文 supplement 里写的 bound 类似：

```text
J(pi_new) >= J(pi_old) - const * beta
```

这只能说明 KL 小时性能下降有界，不等于保证：

```text
J(pi_new) >= J(pi_old)
```

除非还证明 surrogate improvement 大于 penalty 项。

因此我会把论文的单调改进理解为：

```text
在 trust-region 近似和偏好 reward 一致的假设下，更新有局部改进倾向；
不是无条件严格单调提升保证。
```

## 11. Pair-GRPO、DPO、GRPO 的关系

| 方法 | 是否用 reward model scalar | 是否直接用 preference pair | 是否做 RL-style policy update | 是否有 trust region |
| --- | --- | --- | --- | --- |
| GRPO | 是 | 间接 | 是 | 是 |
| DPO | 否 | 是 | 否，直接 preference classification / implicit policy objective | 通过 reference policy 隐式约束 |
| ORPO | 否 | 是 | 否，SFT + odds-ratio preference | 弱 |
| Soft-Pair-GRPO | 否或弱化 | 是，`+1/-1` | 是 | 是 |
| Hard-Pair-GRPO | 否或弱化 | 是，构造 target distribution | 更像 constrained distribution fitting | 是 |

我的理解：

1. Soft-Pair-GRPO 是“把 GRPO 的 scalar advantage 换成 pairwise advantage”。
2. Hard-Pair-GRPO 是“把 pairwise preference 变成一个局部 target policy，然后做 KL fitting”。
3. 它们都想保留 PPO/GRPO 的 trust-region 稳定性，同时减少 reward magnitude 噪声。
4. 它们和 DPO 的区别是：DPO 从最优 KL-regularized reward maximization 解出 preference loss；Pair-GRPO 仍站在 on-policy / trust-region policy update 的框架里。

## 12. 实验

论文做了两类实验。

### 12.1 LLM alignment

设置：

| 项 | 内容 |
| --- | --- |
| Base model | LLaMA-2-7B-Chat |
| Dataset | HH-RLHF 100K，UltraFeedback 200K |
| Baselines | Standard GRPO、DPO、ORPO |
| Metrics | automatic alignment scores、human evaluation、gradient variance、KL std |
| Hyperparameters | LR `1e-5`，`beta=0.01`，`delta_0=0.02`，`gamma=0.98`，`alpha=0.5` |

Automatic alignment metrics：

| Method | HH-RLHF Helpfulness | HH-RLHF Harmlessness | UltraFeedback Win Rate |
| --- | ---: | ---: | ---: |
| Standard GRPO | 78.2% | 81.5% | 76.3% |
| Soft-Pair-GRPO | 79.5% | 82.1% | 77.8% |
| DPO | 80.1% | 82.8% | 78.5% |
| ORPO | 80.5% | 83.2% | 79.1% |
| Hard-Pair-GRPO | 82.3% | 85.7% | 81.9% |

Human evaluation：

| Method | Coherence | Helpfulness | Harmlessness | Relevance | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard GRPO | 4.12 | 4.05 | 4.21 | 4.18 | 4.14 |
| Soft-Pair-GRPO | 4.20 | 4.15 | 4.28 | 4.23 | 4.22 |
| DPO | 4.25 | 4.22 | 4.33 | 4.29 | 4.27 |
| ORPO | 4.28 | 4.26 | 4.36 | 4.32 | 4.31 |
| Hard-Pair-GRPO | 4.42 | 4.40 | 4.51 | 4.45 | 4.45 |

Stability metrics：

| Method | Gradient-Norm Variance | KL-Divergence Std |
| --- | ---: | ---: |
| Standard GRPO | 0.087 | 0.023 |
| Soft-Pair-GRPO | 0.059 | 0.017 |
| DPO | 0.052 | 0.015 |
| ORPO | 0.048 | 0.013 |
| Hard-Pair-GRPO | 0.031 | 0.008 |

### 12.2 Ablation

UltraFeedback win rate：

| Configuration | Win Rate | Training Stability |
| --- | ---: | --- |
| Full Hard-Pair-GRPO | 81.9% | Stable |
| Fixed delta | 80.1% | Moderate oscillation |
| No trust-region constraint | 77.7% | Severe instability |
| Soft-Pair-GRPO | 79.4% | Moderate noise |

补充材料还给出 `delta_0` 和 `gamma` 的 ablation：

| Configuration | Win Rate | Stability |
| --- | ---: | --- |
| `delta_0=0.01, gamma=0.99` | 81.2% | Very Stable |
| `delta_0=0.02, gamma=0.98` | 81.9% | Stable |
| `delta_0=0.05, gamma=0.95` | 80.3% | Moderate Oscillation |

### 12.3 General RL: HalfCheetah-v4

论文还在 MuJoCo HalfCheetah-v4 上验证，说得到同样排序：

```text
PPO < GRPO < Soft-Pair-GRPO < Hard-Pair-GRPO
```

但正文只展示 reward curve 图，没有给完整数值表。因此这部分更像定性支持，不如 LLM alignment 表格可读。

## 13. 可复现性检查

我下载了 arXiv e-print 源码，里面包含：

```text
main.tex
supplementary.tex
references.bib
neurips_2026.sty
figures/loss_curve.png.jpeg
figures/general_rl_reward_curve.png.jpeg
00README.json
```

需要注意：

1. 论文正文说 “Full code is released in supplementary material”，但 e-print 源码里没有 `llm_trainer.py`、`rl_trainer.py`、`models.py`、`utils.py` 等代码文件。
2. `main.tex` 里引用的是 `figures/loss_curve.png` 和 `figures/general_rl_reward_curve.png`，源码实际文件名是 `.png.jpeg`，这可能导致直接编译找不到图片。
3. 实验缺少很多关键复现细节，例如 reward/evaluator 的具体实现、人评协议、随机种子、训练步数、显存配置、采样策略、prompt 模板。
4. 理论证明大多是 sketch，而不是完整严格证明。

因此，虽然报告中的实验数字来自论文，但我不会把这篇当成高可复现实验论文。更适合把它视为一个 Pair-GRPO 思路和理论草图。

## 14. 关键理论结论的可信度分级

| 结论 | 我的可信度 | 原因 |
| --- | --- | --- |
| Binary pair signal 与 scalar reward 在偏好方向上同向 | 高 | 对 `a_p ≻ a_r`，增加 preferred、降低 rejected 是自然方向 |
| Soft-Pair-GRPO 是 GRPO 的简单替换版本 | 高 | 算法定义清楚 |
| Soft 和 GRPO 梯度正比例 | 中 | 局部、未 clipping、pair-centered 近似下成立；正文比例方向有不一致 |
| Soft 方差严格低于 GRPO | 低到中 | 直觉可能成立，但证明不充分且协方差符号可疑 |
| Hard 构造 target distribution 更可控 | 中到高 | 分布目标清楚，确实更局部 |
| Hard 对 irrelevant responses 梯度为零 | 低 | 参数化 softmax 下通常不严格为零，除非显式 mask/freeze |
| 每步单调 policy improvement | 低 | 需要强 TRPO 条件，论文证明不足 |
| Hard > Soft > GRPO 实验排序 | 中 | 表格支持，但复现材料不足 |

## 15. 这篇论文真正有启发的点

### 15.1 偏好对齐中 reward magnitude 可能没那么重要

如果人类反馈主要是：

```text
a_p better than a_r
```

那么 reward model 的绝对分数：

```text
R(a_p)=7.3, R(a_r)=5.1
```

可能并不比：

```text
a_p -> +1
a_r -> -1
```

提供更多有效信息，反而带来 reward scale 和 reward noise。

这与 DPO/IPO/ORPO 的方向一致：偏好学习可以绕开或弱化显式 reward magnitude。

### 15.2 GRPO 可以有 pairwise 化版本

GRPO 通常和可验证任务里的 binary correctness reward 绑定在一起，例如数学题 final answer 对错。但这篇论文提醒我们，GRPO 的框架也可以直接吃 pairwise preference：

```text
same prompt
preferred response vs rejected response
```

这对 RLHF/RLAIF 数据可能有用。

### 15.3 Hard-Pair 的 target distribution 是有工程价值的

相比 reward-weighted gradient，target distribution 更像一个安全的局部编辑：

```text
只把 rejected 的一点概率挪给 preferred
每步挪多少由 delta_t 控制
总 KL 由 beta 控制
```

这可能适合那些非常怕 policy drift 的对齐场景。

## 16. 和现有后训练路线的对比

| 路线 | 核心监督 | 优点 | 风险 |
| --- | --- | --- | --- |
| PPO | scalar reward + value/advantage | 理论成熟、通用 | 复杂、成本高、reward hacking |
| GRPO | group-normalized reward | 无 critic、适合 LLM RL | reward scale/group statistics 仍影响更新 |
| DPO | preference pair + reference policy | 简洁稳定、不跑 RL loop | 偏 offline，受数据分布限制 |
| ORPO | SFT + odds ratio preference | 简洁 | trust-region 弱 |
| Soft-Pair-GRPO | preference pair 的 `+1/-1` | GRPO 最小改造、方向明确 | 仍可能全局漂移 |
| Hard-Pair-GRPO | pairwise target distribution | 更新更局部、KL 可控 | 实现复杂，理论保证不足 |

## 17. 如果要实现，应该怎么做

### 17.1 Soft-Pair-GRPO 伪代码

```python
for batch in preference_pairs:
    # batch: prompt, preferred, rejected
    logp_pref = policy.logprob(prompt, preferred)
    logp_rej = policy.logprob(prompt, rejected)
    old_logp_pref = old_policy.logprob(prompt, preferred)
    old_logp_rej = old_policy.logprob(prompt, rejected)

    rho_pref = exp(logp_pref - old_logp_pref)
    rho_rej = exp(logp_rej - old_logp_rej)

    obj_pref = clipped_surrogate(rho_pref, +1)
    obj_rej = clipped_surrogate(rho_rej, -1)

    kl = KL(policy, old_policy, prompt)
    loss = -(obj_pref + obj_rej) + beta * kl
    loss.backward()
```

这里的负号取决于是 maximize objective 还是 minimize loss。

### 17.2 Hard-Pair-GRPO 的实现难点

LLM 的 action space 是整段 response，不是小型离散 action 表。直接构造完整：

```text
pi_tar(.|s)
```

在 LLM 上几乎不可行，因为 response space 巨大。

可行近似可能是：

1. 只在 sampled responses 集合上构造局部 target distribution。
2. 把 preferred/rejected response 的 sequence logprob 当成 action probability proxy。
3. 对 batch 内候选响应做归一化。
4. 使用 pair-level KL / cross entropy，而不是 full vocabulary / full sequence distribution KL。
5. 对 token-level policy 保留普通 KL regularization。

这也是论文没有完全展开的关键工程缺口。

## 18. 我会如何改进论文算法

### 18.1 修正 Soft 梯度等价表述

更稳妥的 theorem 应该写成：

```text
Under local, unclipped, pair-centered approximation,
the reward-gradient part of GRPO and Soft-Pair-GRPO are positively colinear:

grad L_GRPO_reward ≈ C grad L_Soft_reward
C = Delta R / (2 sigma_R) > 0
```

并明确：

1. 这是 reward-gradient part，不包括 KL。
2. clipping 激活后不保证。
3. group 不等于 pair 时只是近似。

### 18.2 把方差定理改成条件命题

可以写成：

```text
If scalar reward contains zero-mean noise independent of preference ordering,
and pair labels have lower noise than reward magnitudes,
then binary pair gradients have lower variance in the reward-weight component.
```

这样比无条件写：

```text
Var_Soft < Var_GRPO
```

更合理。

### 18.3 Hard-Pair 增加 explicit mask

如果真的想保证 irrelevant responses 不动，需要：

```text
mask gradients outside sampled pair set
or project update onto local pair subspace
or use constrained optimization over candidate response set
```

否则 neural policy 的参数共享会让“局部 target”变成“全局参数更新”。

### 18.4 补充可复现实验

至少需要：

1. 开源训练代码。
2. 给出 exact prompts 和 evaluation scripts。
3. 多 seed 结果。
4. 完整 HalfCheetah 数值表。
5. 和最新 RLHF baselines 的公平设置。
6. 明确 Hard-Pair 在 LLM 巨大 action space 上如何近似 `pi_tar`。

## 19. 最终结论

这篇论文可以记成一句话：

```text
Pair-GRPO = 把 GRPO 的 reward-driven update 改成 preference-pair-driven update。
Soft 版本用 +1/-1 隐式表达偏好；
Hard 版本用 target distribution 显式移动概率质量。
```

理论上，最稳的结论是：

```text
在局部一阶近似下，pairwise binary signal 与 normalized scalar reward 对 preferred/rejected pair 给出同向梯度。
```

但不要过度相信文中的强声明：

```text
严格单调改进
严格方差层级
irrelevant response 梯度为零
```

这些都需要比论文当前更强的假设和更完整的证明。

我的整体评价：

1. **作为算法想法：有价值。** 尤其 Soft-Pair-GRPO 是 GRPO 的低成本 pairwise 改造。
2. **作为理论论文：不够严密。** 多个定理是 proof sketch，部分符号和结论存在可疑点。
3. **作为实验论文：复现材料不足。** arXiv 源码没有实际训练代码，部分图文件名也和 TeX 引用不一致。
4. **作为后训练路线启发：值得保留。** 它把 DPO 式 pairwise preference 和 GRPO 式 trust-region RL 放在同一个设计空间里，这个方向后续可能会有更严谨版本。
