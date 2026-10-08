# Pair-GRPO Family 精读

阅读日期：2026-06-01  
论文标题：A Unified Pair-GRPO Family: From Implicit to Explicit Preference Constraints for Stable and General RL Alignment  
作者：Hao Yu  
机构：Department of Automation, Tsinghua University  
arXiv：2605.06375v1，2026-05-07  
链接：[arXiv Abs](https://arxiv.org/abs/2605.06375)，[HTML](https://arxiv.org/html/2605.06375)，[PDF](https://arxiv.org/pdf/2605.06375)  
资料版本：arXiv PDF，SHA256 `afe7d39c9de89cbaaf78d6c399cb54b694370dfe665a71be005f9611df9b7386`  
源码版本：arXiv e-print，SHA256 `94ef19a29eaf9a1b6ecdc6dd6d34da9ee215fb2cdf841a33f6e689b3ec4ed3e1`

资料中的 PDF/e-print 哈希与源码清单为阅读时留存。下文按论文 v1 的定义展开，并分别讨论结论成立的条件与反例。

## 一句话结论

这篇论文提出了一个 **Pair-GRPO family**，试图把 GRPO 的 group-normalized scalar reward 改成 pairwise preference signal。它分成两个版本：

1. **Soft-Pair-GRPO**：把偏好对 `(a_p, a_r)` 直接编码成 `+1/-1`，仍然使用 GRPO/PPO 风格的 clipped surrogate 和 KL regularization。
2. **Hard-Pair-GRPO**：不再只给 reward，而是显式构造一个 target policy distribution，把概率质量从 rejected response 转给 preferred response，再做带 trust-region 的 KL fitting。

论文最值得读的地方是它把“偏好对齐中的相对偏好信号”和“GRPO trust-region 更新”连接起来。但我对它的理论严谨性比较谨慎：单个 pair 在中心化等条件下可得到梯度同向；一般 group 和跨 pair 平均不能直接套用；可是文中关于“严格方差降低”“单调改进保证”“irrelevant responses 梯度为零”等表述需要额外假设，正文和补充材料里也存在比例常数方向、方差协方差符号、代码可复现性等不够严密的地方。

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

然后用 reward model 或规则 verifier 给每个 response 打分，并在组内标准化：

$$
A_i=\frac{R(a_i)-\mu_R}{\sigma_R}
$$

这里 $R(a_i)$ 是第 $i$ 个回答的奖励，$\mu_R$、$\sigma_R$ 是同一问题下这组回答奖励的均值与标准差。$A_i$ 衡量“该回答比本组平均水平好多少”，并按组内波动幅度缩放；方差为零或很小时，实现需要相应的稳定处理。再用 PPO/GRPO clipped surrogate 更新策略。

论文认为这里有三个问题：

| 问题 | 解释 |
| --- | --- |
| 梯度估计受噪声影响 | reward 的相对差异、采样与组内统计量会影响更新；不能把任意整体平移或正比例缩放都视为噪声 |
| 连续 reward 信息冗余 | 人类偏好本质是 pairwise ordering，不一定需要绝对 reward magnitude |
| 全局 policy drift | reward-based objective 可能影响整个 action/response space，不只改变关键偏好对 |

因此论文提出：既然偏好数据本来就是 `a_p ≻ a_r`，就直接让算法围绕 pairwise preference 更新。

无数值稳定项时，对整组奖励做 $R_i\mapsto aR_i+b$（$a>0$）不改变标准化 advantage。加 $\epsilon$、改变组内结构或存在噪声后才需进一步分析。

## 2. 符号

| 符号 | 含义 |
| --- | --- |
| `s` | prompt / state |
| `a` | response / action |
| `pi_theta(a\|s)` | 当前策略 |
| `pi_old(a\|s)` | 产生当前 batch 的冻结旧策略；不同于通常长期固定的 KL reference |
| `(s, a_p, a_r)` | preference pair，`a_p` preferred，`a_r` rejected |
| `R(a)` | reward model 给 response 的 scalar reward |
| `rho(s,a)` | importance ratio，`pi_theta(a\|s) / pi_old(a\|s)` |
| `D_KL(P \|\| Q)` | KL divergence |
| `beta` | trust-region KL 阈值或 penalty 系数 |
| `epsilon` | PPO/GRPO clip range |
| `delta_t` | Hard-Pair-GRPO 中概率质量转移步长 |

本文按论文的简化 sequence-level 写法讨论；实践中的 GRPO 常用 token-level ratio，并另设长期冻结的 $\pi_{\rm ref}$ 做 KL 正则。不能把 $\pi_{\rm old}$ 与 $\pi_{\rm ref}$ 无条件混为一谈。

## 3. 标准 GRPO 的局部形式

论文采用的 GRPO objective 是：

$$
J_{\rm GRPO}(\theta)=\mathbb E\!\left[\min\!\left(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\right)\right]
$$

其中：

$$
\rho(s,a)=\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)},\qquad A_i=\frac{R(a_i)-\mu_R}{\sigma_R}
$$

为了看清理论主线，可以先忽略 clipping 和 KL penalty，只看局部一阶梯度。因为在 `theta = theta_old` 附近：

$$
\left.\rho\right|_{\theta=\theta_{\rm old}}=1,\qquad\nabla_\theta\rho=\rho\nabla_\theta\log\pi_\theta(a\mid s)
$$

所以局部 policy-gradient 形式是：

$$
\left.\nabla J_{\rm GRPO}\right|_{\theta=\theta_{\rm old}}=\mathbb E[A(a)\nabla\log\pi_\theta(a\mid s)]
$$

这是 reward-weighted policy gradient。单项 advantage 指定该样本的局部推动方向；合并多个样本并更新共享参数后，各自概率的实际变化还包含梯度相互作用。

## 4. Soft-Pair-GRPO

### 4.1 算法定义

给定偏好对：

$$
(s,a_p,a_r),\qquad a_p\succ a_r
$$

Soft-Pair-GRPO 定义 binary pairwise reward：

$$
r^{\rm Soft}(a_p)=+1,\qquad r^{\rm Soft}(a_r)=-1
$$

对应 objective：

$$
J_{\rm Soft}(\theta)=\mathbb E\!\left[\min\!\left(\rho r^{\rm Soft},\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)r^{\rm Soft}\right)\right]-\beta\,\mathbb E_s D_{\rm KL}(\pi_\theta\Vert\pi_{\rm old})
$$

这个目标分成两部分：前半用偏好奖励推动策略更新，后半通过 $-\beta D_{\rm KL}$ 惩罚与旧策略的偏离。因为这里做梯度上升，惩罚项要取负号；若代码改写成最小化 loss，则整体目标取负，KL 项也随之变正。

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

$$
g_{\rm Soft}=\nabla\log\pi_\theta(a_p\mid s)-\nabla\log\pi_\theta(a_r\mid s)\quad(\theta=\theta_{\rm old})
$$

在旧策略处、未 clipping 且暂不计 KL 时，沿此方向做足够小的梯度上升，会一阶增大 preferred/rejected 的对数概率比；不保证两个绝对概率分别升降，见第 6 节。

如果代码里把它写成 loss 并做梯度下降，则符号整体反过来，但更新含义不变。

## 5. 理论推导 1：Soft-Pair-GRPO 与 GRPO 的梯度关系

先从只有两个回答的组开始，再讨论一对回答嵌在更大组里的情况。两种情况下“组均值”的含义不同，这正是后续梯度能否同向的关键。

### 5.1 两响应 pair group 的最简单情形

假设同一 prompt 下只比较两个 response：

$$
\mathcal G=\{a_p,a_r\},\qquad\Delta R=R(a_p)-R(a_r)>0
$$

组内均值：

$$
\mu_R=\frac{R(a_p)+R(a_r)}2
$$

为了缩短式子，记 $R_p=R(a_p)$、$R_r=R(a_r)$。两点分别位于均值上方、下方 $\Delta R/2$，因此若使用二点总体标准差：

$$
\sigma_R=\sqrt{\frac{(R_p-\mu_R)^2+(R_r-\mu_R)^2}{2}}=\frac{|\Delta R|}{2}
$$

于是：

$$
A_p=\frac{\Delta R/2}{\Delta R/2}=+1,\qquad A_r=\frac{-\Delta R/2}{\Delta R/2}=-1
$$

因此在这个最简单情形下：

$$
g_{\rm GRPO}=g_{\rm Soft}
$$

也就是说，如果 GRPO 的 group 本身就是一个偏好 pair，组内标准化会自然把 reward 变成 `+1/-1`。这能解释为什么 binary preference signal 可能不会破坏方向。

### 5.2 更一般的 pair 嵌入 group 情形

现在一个 group 不止两个回答，从中取出偏好对 $a_p,a_r$。定义 $g_p=\nabla_\theta\log\pi_\theta(a_p\mid s)$、$g_r=\nabla_\theta\log\pi_\theta(a_r\mid s)$：二者都是相对于**同一组模型参数**的梯度向量，描述增大各自回答 log-prob 的局部方向。再记二者的差为 $h=g_p-g_r$，即 Soft 方法使用的方向。

$\mu_R,\sigma_R$ 仍是**整个 group** 的均值和标准差，不一定等于这一对回答自己的统计量。在旧策略处、无 clipping 时，这一对回答对 GRPO 梯度的贡献（不计共同平均系数）可以分为两部分：

$$
g_{\rm GRPO,pair}
=\frac{R_p-R_r}{2\sigma_R}(g_p-g_r)
+\frac{(R_p+R_r)/2-\mu_R}{\sigma_R}(g_p+g_r).
$$

**把共同项是如何出现的也展开。** 先记 pair 自身均值 $\bar R=(R_p+R_r)/2$、差值 $\Delta R=R_p-R_r$，则：

$$
R_p-\mu_R=(\bar R-\mu_R)+\Delta R/2,\qquad
R_r-\mu_R=(\bar R-\mu_R)-\Delta R/2.
$$

分别代回原始梯度贡献：

$$
\begin{aligned}
g_{\rm GRPO,pair}
&=\frac{R_p-\mu_R}{\sigma_R}g_p+\frac{R_r-\mu_R}{\sigma_R}g_r\\
&=\frac{\bar R-\mu_R+\Delta R/2}{\sigma_R}g_p
+\frac{\bar R-\mu_R-\Delta R/2}{\sigma_R}g_r\\
&=\frac{\Delta R}{2\sigma_R}(g_p-g_r)
+\frac{\bar R-\mu_R}{\sigma_R}(g_p+g_r).
\end{aligned}
$$

只有最后一项消失，才能把整个 pair 贡献直接写成 Soft 方向的一个倍数。

第一项比较这一对回答谁更好，沿 $g_p-g_r$ 推动；第二项表示这一对回答作为整体比全组均值高还是低，沿 $g_p+g_r$ 共同推动。仅当 $\mu_R=(R_p+R_r)/2$，或共同项确实消失时，才有：

$$
g_{\rm GRPO,pair}=C h,\qquad
C=\frac{R_p-R_r}{2\sigma_R}>0.
$$

若 group 恰好只有这两个响应，采用总体标准差、无额外 $\epsilon$，则 $C=1$。如果用样本标准差或数值稳定项，尺度会变。

### 5.3 跨 pair 的期望不能随意拆开

即使每一对都满足 $g_{\rm GRPO,pair}=C h$，一般仍有：

$$
\mathbb E[C h]\ne\mathbb E[C]\,\mathbb E[h].
$$

不同 pair 的方向与正系数相互关联，平均后不一定同向。论文正文与补充材料的比例方向也不一致，但问题不只在倒数：还必须说明共同项何时消失，以及怎样处理期望。

因此可接受的结论是**单个 pair、旧策略处、未 clipping、pair-centered、仅 reward 梯度**的正比例关系。加入 KL 后，除非连 KL 系数也相应缩放，否则完整梯度不再满足相同的比例；远离旧策略时 $\rho_p,\rho_r$ 不同，也会改变关系。

## 6. 理论推导 2：确定的是局部概率比方向

在上述局部条件下，令 $\theta'=\theta+\eta h$，则：

$$
\log\frac{\pi_{\theta'}(a_p\mid s)}{\pi_{\theta'}(a_r\mid s)}
-\log\frac{\pi_\theta(a_p\mid s)}{\pi_\theta(a_r\mid s)}
=\eta\lVert h\rVert^2+O(\eta^2).
$$

但两个对数概率分别的变化为：

$$
\Delta\log\pi(a_p\mid s)\approx\eta\,g_p^\top h,\qquad
\Delta\log\pi(a_r\mid s)\approx\eta\,g_r^\top h.
$$

它们不必一正一负。一个共享参数反例：三个 action 的 logits 为 $z(\theta)=(\theta,2\theta,-3\theta)$，初始 $\theta=0$，preferred 为第一个，rejected 为第二个。此时 $g_p=1,g_r=2,h=-1$。取 $\eta=0.001$，则 $\theta'=-0.001$：

| 概率 | 更新前 | 更新后 |
| --- | ---: | ---: |
| preferred | 0.3333333 | 0.3329994 |
| rejected | 0.3333333 | 0.3326666 |

两个概率都下降，但 preferred/rejected 比值增大。原文“preferred 必涨、rejected 必跌”不能作为一般神经网络策略的保证。独立 logits 等特殊参数化下可能成立，不能推广到共享参数。

## 7. 理论推导 3：Soft-Pair-GRPO 的方差讨论

论文声称（这里以单个梯度分量的方差说明；向量情形需用协方差矩阵或其迹）：

$$
\operatorname{Var}_{\rm Soft}<\operatorname{Var}_{\rm GRPO}
$$

直觉上，Soft-Pair 的优势是去掉了 reward magnitude 的噪声：

```text
GRPO: advantage depends on relative rewards and group statistics
Soft: advantage fixed as +1/-1
```

如果 reward model 的绝对值很 noisy，而 preference ordering 较可靠，那么 Soft 的确可能降低梯度噪声。

但论文给出的严格证明并不充分。补充材料写：

$$
g=g_p-g_r,\qquad\operatorname{Var}(g)=\operatorname{Var}(g_p)+\operatorname{Var}(g_r)-2\operatorname{Cov}(g_p,g_r)
$$

然后说 Soft-Pair 中 `Cov(g_p, g_r) < 0`，所以方差降低。

这里符号上有问题：如果公式是 `Var(g_p - g_r)`，那么当 `Cov(g_p,g_r) < 0` 时：

$$
-2\operatorname{Cov}(g_p,g_r)>0
$$

方差反而变大。要让 `g_p - g_r` 的方差降低，需要 `Cov(g_p,g_r) > 0`。如果把 rejected response 的负号已经吸收到 `g_r` 里，公式又应该写成 `Var(g_p + g_r)`，此时协方差解释也要重新定义。

所以我会这样理解这条理论：

1. **经验直觉合理**：binary pair signal 可能比 noisy scalar reward 稳。
2. **严格不等式不成立为一般定理**：必须假设 reward noise、pair sampling、gradient covariance 的特定结构。
3. **论文证明有符号问题**：不能直接作为严格证明使用。

## 8. Hard-Pair-GRPO

### 8.1 目标分布构造

Hard-Pair-GRPO 想进一步解决 Soft 的问题：Soft 仍然是 reward-weighted policy gradient，可能导致全局 policy drift。

它对每个 preference pair 构造一个 target distribution：

$$
\pi_{\rm tar}(a\mid s)=\begin{cases}\pi_{\rm old}(a_p\mid s)+\delta_t&a=a_p\\\pi_{\rm old}(a_r\mid s)-\delta_t&a=a_r\\\pi_{\rm old}(a\mid s)&a\notin\{a_p,a_r\}\end{cases}
$$

其中：

$$
\delta_t=\delta_0\gamma^t,\qquad\delta_0>0,\quad0<\gamma<1
$$

也就是显式把一小块概率质量从 rejected response 转给 preferred response。

为了保证这是合法分布，需要：

$$
0\le\delta_t\le\pi_{\rm old}(a_r\mid s),\qquad\pi_{\rm old}(a_p\mid s)+\delta_t\le1
$$

实践中需要自适应缩小 $\delta_t$。对 $D_{\rm KL}(p\Vert q)$，若当前 $p_r>0$ 却令目标 $q_r=0$，KL 会发散，因此通常还应要求 $0<\delta_t<p_{\rm old,r}$，而非允许直接减到零。

### 8.2 KL fitting objective

Hard-Pair-GRPO 不再用 `+1/-1` surrogate，而是做 constrained KL fitting：

$$
\min_\theta D_{\rm KL}(\pi_\theta\Vert\pi_{\rm tar})\quad\text{subject to }D_{\rm KL}(\pi_\theta\Vert\pi_{\rm old})\le\beta
$$

论文把约束写成 hinge penalty：

$$
\mathcal L_{\rm total}=D_{\rm KL}(\pi_\theta\Vert\pi_{\rm tar})+\alpha\max\!\left(D_{\rm KL}(\pi_\theta\Vert\pi_{\rm old})-\beta,0\right)
$$

直观含义：

1. 第一项让当前策略靠近目标分布 `pi_tar`。
2. 第二项限制当前策略不要离 `pi_old` 太远。
3. `delta_t` 控制目标移动的概率质量。有限的 hinge 系数 $\alpha$ 只是软惩罚，并不自动保证每步满足硬 KL 约束。

### 8.3 KL fitting 的分布层推导

把某个 prompt 下的策略分布记为：

$$
p_i=\pi_\theta(a_i\mid s),\qquad q_i=\pi_{\rm tar}(a_i\mid s)
$$

明确按“当前策略到目标策略”的方向写：

$$
D_{\rm KL}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}
$$

不带 trust-region 约束且能直接优化整个概率 simplex 时，最小值在：

$$
p_i=q_i\quad\forall i
$$

因为：

$$
D_{\rm KL}(p\Vert q)\ge0
$$

且等号当且仅当 `p=q`。带回约束后，只有目标本身满足 $D_{\rm KL}(q\Vert\pi_{\rm old})\le\beta$ 且模型能够表示它，才能直接取 $p=q$；否则必须求受约束最优解。

因此 Hard-Pair-GRPO 的目标就是把：

```text
pi_theta(a_p|s) 推向 pi_old(a_p|s)+delta_t
pi_theta(a_r|s) 推向 pi_old(a_r|s)-delta_t
```

其他 action 的目标概率和 old policy 相同。

### 8.4 一阶近似：为什么是局部概率转移

当 `delta_t` 很小时：

$$
q_p=p_{{\rm old},p}+\delta,\quad q_r=p_{{\rm old},r}-\delta,\quad q_c=p_{{\rm old},c}
$$

对 preferred action：

$$
\log\frac{p_{{\rm old},p}}{q_p}=-\log\!\left(1+\frac{\delta}{p_{{\rm old},p}}\right)\approx-\frac{\delta}{p_{{\rm old},p}}
$$

对 rejected action：

$$
\log\frac{p_{{\rm old},r}}{q_r}=-\log\!\left(1-\frac{\delta}{p_{{\rm old},r}}\right)\approx\frac{\delta}{p_{{\rm old},r}}
$$

所以在 `p = pi_old` 附近，KL fitting 的信号会让：

```text
preferred: 当前概率低于 target，需要上调
rejected:  当前概率高于 target，需要下调
```

这就是 Hard-Pair-GRPO 比 Soft 更“硬”的地方：不是告诉策略“preferred reward 高”，而是直接给策略一个局部分布目标。

### 8.5 关于“irrelevant responses 梯度为零”的谨慎解释

论文声称 Hard-Pair-GRPO 对无关 response 的梯度理论上为零。这个说法只在一种理想化解释下成立：如果优化器直接操作分布坐标，并且显式冻结 `c not in {a_p,a_r}` 的概率，那么无关 action 的目标不变。

但对真实神经网络策略：

$$
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
$$

即使 `q_c = p_old_c`，此方向的 KL 对 logits 的梯度是：

$$
\frac{\partial D_{\rm KL}(p\Vert q)}{\partial z_j}=p_j\left(\log\frac{p_j}{q_j}-\sum_i p_i\log\frac{p_i}{q_i}\right)
$$

在 `p = p_old`、`q` 只改变 pair 的情况下，对无关 `c`：

$$
\log(p_c/q_c)=0
$$

但全局期望项包含 preferred/rejected 的 KL，因此无关 logits 的梯度不一定严格为零。更重要的是，神经网络参数共享会让更新 pair token/response 的同时影响其他 response。

所以更准确的说法是：

```text
Hard-Pair-GRPO 的 target distribution 只显式改变 pair 上的目标概率；
但参数化策略训练中，无关 response 是否完全不变，需要约束输出分布或使用满足相应 Jacobian 约束的投影；仅屏蔽样本损失不足以保证。
```

## 9. 理论推导 4：Hard-Pair-GRPO 的动态步长

论文使用：

$$
\delta_t=\delta_0\gamma^t,\qquad\gamma\in(0,1)
$$

显然：

$$
\lim_{t\to\infty}\delta_t=0
$$

因此随着训练推进：

$$
\lVert\pi_{{\rm tar},t}-\pi_{{\rm old},t}\rVert_1=2\delta_t\longrightarrow0
$$

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

$$
J(\pi_\theta)\ge J(\pi_{\rm old})
$$

理由是 trust-region KL constraint + PPO/GRPO monotonic improvement framework。

这里需要非常谨慎。严格 TRPO 单调改进定理依赖：

1. 准确 advantage。
2. 明确 surrogate lower bound。
3. 受控的最大 KL 或 TV distance。
4. 优化问题足够精确地求解。

PPO clipping 和 KL penalty 本身并不自动给出每一步真实 return 的严格单调提升。论文 supplement 里写的 bound 类似：

$$
J(\pi_{\rm new})\ge J(\pi_{\rm old})-C\beta
$$

这只能说明 KL 小时性能下降有界，不等于保证：

$$
J(\pi_{\rm new})\ge J(\pi_{\rm old})
$$

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

**原论文内部差异：** 主表 Soft-Pair 的 UltraFeedback win rate 为 77.8%，此消融表为 79.4%；缺少足够说明来确定是否同一设置。保留两个原值，不自行合并或改数。

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

## 13. 历史可复现性检查记录

阅读时留存的 arXiv e-print 存档文件清单如下，其 SHA256 见文首：

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
| Soft 的局部 pair log-ratio 上升 | 高，有条件 | 旧策略处、忽略 clipping/KL、足够小步长；不保证两个绝对概率分别升降 |
| Soft-Pair-GRPO 是 GRPO 的简单替换版本 | 高 | 算法定义清楚 |
| Soft 和 GRPO 梯度正比例 | 仅条件成立 | 单个 pair 且共同项消失；跨 pair 期望不能直接提出变系数 |
| Soft 方差严格低于 GRPO | 低到中 | 直觉可能成立，但证明不充分且协方差符号可疑 |
| Hard 构造 target distribution 更可控 | 中到高 | 分布目标清楚，确实更局部 |
| Hard 对 irrelevant responses 梯度为零 | 低 | 参数化 softmax 下通常不严格为零，除非对输出分布施加额外约束 |
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

在目标分布定义上，它只编辑一对响应；这不代表参数更新本身具有相同局部性：

```text
只把 rejected 的一点概率挪给 preferred
每步挪多少由 delta_t 控制
目标偏移由 delta_t 控制；实际 KL 还需验证约束是否满足
```

这可能适合那些非常怕 policy drift 的对齐场景。

## 16. 和现有后训练路线的对比

| 路线 | 核心监督 | 优点 | 风险 |
| --- | --- | --- | --- |
| PPO | scalar reward + value/advantage | 理论成熟、通用 | 复杂、成本高、reward hacking |
| GRPO | group-normalized reward | 无 critic、适合 LLM RL | 组内采样、奖励噪声和归一化退化影响更新 |
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

这里是最小化 loss，surrogate 前为负、KL 前为正。它只是算法示意，缺少 optimizer step、梯度清零等实际训练步骤；还需明确 pair 的采样/筛选分布、旧策略冻结、token mask、长度归一化及重要性权重，不能把任意离线偏好 batch 直接当作未经选择的 on-policy 样本。

### 17.2 Hard-Pair-GRPO 的实现难点

LLM 的 action space 是整段 response，不是小型离散 action 表。直接构造完整：

$$
\pi_{\rm tar}(\cdot\mid s)
$$

在 LLM 上几乎不可行，因为 response space 巨大。

可行近似可能是：

1. 只在 sampled responses 集合上构造局部 target distribution。
2. 用完整 sequence logprob 计算候选权重；logprob 本身不是概率，需做指数与稳定归一化。
3. 对 batch 内候选响应做归一化。
4. 使用 pair-level KL / cross entropy，而不是 full vocabulary / full sequence distribution KL。
5. 对 token-level policy 保留普通 KL regularization。

这也是论文没有完全展开的关键工程缺口。

## 18. 我会如何改进论文算法

### 18.1 修正 Soft 梯度等价表述

应直接使用第 5 节含共同项的分解。仅在单个 pair、共同项消失、旧策略处、未 clipping 且不含 KL 时，才能写 $g_{\rm GRPO,pair}=C g_{\rm Soft,pair}$。一般 group 不是简单“近似一下即可”；误差项要显式估计，跨 pair 还要处理 $C$ 与 $h$ 的相关性。

### 18.2 将方差直觉写成可检验的条件

一个更窄、可证明的示例是固定方向 $h$，权重受到独立零均值噪声 $\varepsilon$：

$$
\widehat g=(a+\varepsilon)h,\quad
\mathbb E[\varepsilon]=0,\quad
\operatorname{tr}\operatorname{Cov}(\widehat g)
=\operatorname{Var}(\varepsilon)\lVert h\rVert^2.
$$

去掉这一噪声项可降低该条件模型的方差。但 GRPO 的组内均值、标准差与 pair 筛选会让权重和方向相关，偏好标签本身也可能有噪声，因此这不是 Soft-Pair 严格优于 GRPO 的普遍定理。

### 18.3 局部目标需要输出分布约束

只 mask 掉非 pair 的 loss，甚至将非 pair logits 的显式梯度置零，也不保证那些响应的概率不动：softmax 归一化和共享参数仍会改变它们。

若真要保证局部性，需要直接在概率 simplex 上固定其他坐标，或对参数步长施加 $J_c\Delta\theta=0$ 等输出 Jacobian 约束（这通常仅保证一阶不变），并检查实际更新后的分布。只在候选集合中归一化也不等于冻结候选集合之外的概率。

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
在旧策略处、未 clipping、pair-centered 的单个 pair 上，两者的 reward 梯度正比例；一般 group 或跨 pair 平均不保证。Soft 直接保证的是局部 pair log-ratio 的推动方向。
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
