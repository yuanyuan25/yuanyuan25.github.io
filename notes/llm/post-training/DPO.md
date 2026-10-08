# DPO：偏好优化、隐式奖励与 PPO

这篇笔记沿着一个问题展开：已有“回答 A 比回答 B 更好”的偏好数据，怎样让语言模型更愿意生成好的回答？先看 RLHF 如何先学打分、再优化生成策略，再推导 DPO 怎样把这两步合成一个训练目标。

DPO（Direct Preference Optimization）直接用偏好对训练策略，标准离线 DPO 不需要另外训练显式 RM，也不需要在优化循环中运行 PPO。DPO 得到的是生成模型；它与固定 reference 一起可以定义一个隐式奖励函数。

## 1. 模型角色与符号

对于同一个问题 $x$，语言模型可能生成许多不同回答。我们用策略 $\pi(y\mid x)$ 表示“看到 $x$ 后生成完整回答 $y$ 的概率”。策略由语言模型实现；改变模型参数，就是改变这些回答的概率分布。

奖励函数 $r(x,y)$ 则给一个已经确定的“问题—回答”组合打分。它输出一个实数，分数高表示模型判断这个回答更值得鼓励。**生成概率回答“会不会说”，奖励分数回答“有多值得鼓励”**，两者承担不同角色。

| 符号 | 含义 |
| --- | --- |
| $x$ | prompt |
| $y_w,y_l$ | 同一 prompt 下的 preferred / rejected 回答 |
| $r_\phi(x,y)$ | 显式奖励模型输出的标量 |
| $\pi_{\rm ref}$ | DPO 的固定参考策略，通常为训练开始前的 SFT 模型 |
| $\pi_\theta$ | DPO 正在优化的策略 |
| $\pi^*$ | 给定奖励下，理想 KL 正则化问题的最优策略 |
| $\beta>0$ | 奖励与偏离 reference 之间的权衡系数 |

通常从同一个 SFT checkpoint 复制两份：一份冻结为 reference，另一份接受 DPO 更新。reference 不是 PPO 中随训练迭代同步的 old policy。

## 2. Bradley–Terry 偏好模型

对于一个问题 $x$，假设我们已有一个打分函数 $r$，并拿到两个候选回答。偏好数据把更受偏爱的回答记作 $y_w$（winner），另一个记作 $y_l$（loser）；符号 $y_w\succ y_l$ 表示比较者更偏爱前者。这里先不要求知道奖励的具体值，而是规定：奖励差越大，比较者选择高分回答的概率越大。

Bradley–Terry 模型把这个关系写成下面的偏好概率。$\sigma(z)=1/(1+e^{-z})$ 是 sigmoid，它把任意实数差值转换到 0 与 1 之间：

$$
\begin{aligned}
P(y_w\succ y_l\mid x) &=\frac{e^{r(x,y_w)}}{e^{r(x,y_w)}+e^{r(x,y_l)}}\\
&=\frac{1}{1+e^{r(x,y_l)-r(x,y_w)}}\\
&=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr).
\end{aligned}
$$

当两个回答得分相等时，模型给出各一半的偏好概率；当 $r(x,y_w)>r(x,y_l)$ 时，选择 $y_w$ 的概率大于一半。公式第一步给两个奖励取指数作为正权重，第二步将分子、分母同时除以 $e^{r(x,y_w)}$，于是只剩下奖励差。

这里输出的是两个回答之间的**偏好概率**，不是语言模型生成某个回答的概率。


<details>
<summary>公式配图（可选）</summary>

![公式或实现配图](<pic/Bradley-Terry.png>)

</details>

## 3. 显式 RM 的极大似然训练

现在让奖励函数由一个参数为 $\phi$ 的可训练模型 $r_\phi$ 来实现，这就是显式 RM。训练数据集 $\mathcal D=\{(x_i,y_{w,i},y_{l,i})\}_{i=1}^N$ 一共有 $N$ 条比较记录；下标 $i$ 表示第几条记录，$\phi$ 是要调整的参数。

对于第 $i$ 条记录，我们定义奖励差 $\Delta r_i=r_\phi(x_i,y_{w,i})-r_\phi(x_i,y_{l,i})$。接下来要做的，是让 RM 对这些已观察到的偏好给出尽可能高的概率。

### 3.1 从单条偏好概率到数据集似然

单条样本被标为 chosen 胜出的概率为 $p_i=\sigma(\Delta r_i)$。在样本条件独立的建模假设下，观察到整组偏好的概率是各项之积：

$$
\begin{aligned}
\mathcal L_{\rm likelihood}(\phi) &=\prod_{i=1}^N P_\phi(y_{w,i}\succ y_{l,i}\mid x_i)\\
&=\prod_{i=1}^N\sigma(\Delta r_i).
\end{aligned}
$$

### 3.2 取对数：乘积变求和

似然把整组已知偏好出现的可能性，看成参数 $\phi$ 的函数。很多小于 1 的概率相乘，数值容易非常小；改用对数后可以逐条累加，也更便于求梯度。

使用 $\log\prod_i p_i=\sum_i\log p_i$，再代入 BT 模型：

$$
\begin{aligned}
\log\mathcal L_{\rm likelihood}(\phi)
&=\log\prod_{i=1}^N\sigma(\Delta r_i)\\
&=\sum_{i=1}^N\log\sigma(\Delta r_i)\\
&=\sum_{i=1}^N\log\sigma\!\left(r_\phi(x_i,y_{w,i})-r_\phi(x_i,y_{l,i})\right).
\end{aligned}
$$

因为 log 严格递增，取对数不改变最大化解：

$$
\begin{aligned}
\phi^* &=\arg\max_\phi\mathcal L_{\rm likelihood}(\phi)\\
&=\arg\max_\phi\log\mathcal L_{\rm likelihood}(\phi).
\end{aligned}
$$

### 3.3 极大似然变成最小化负对数损失

训练代码通常以“最小化 loss”为统一接口，所以把希望增大的对数似然取负号。$\mathcal L_{\rm RM,sum}$ 表示全部样本的损失之和，$\arg\min$ 表示“使它最小的参数”，不是损失的数值。

给整个目标乘 $-1$ 后，最大化改成最小化：

$$
\begin{aligned}
\mathcal L_{\rm RM,sum}(\phi) &=-\log\mathcal L_{\rm likelihood}(\phi)\\
&=-\sum_{i=1}^N\log\sigma(\Delta r_i),
\end{aligned}
$$

$$
\phi^*=\arg\min_\phi\mathcal L_{\rm RM,sum}(\phi).
$$

### 3.4 从样本和到均值、经验期望

令 $\widehat{\mathcal D}$ 表示对这 $N$ 条记录均匀抽样的经验分布；除以与参数无关的正常数 $N$：

$$
\begin{aligned}
\mathcal L_{\rm RM}(\phi)
&=\frac{1}{N}\mathcal L_{\rm RM,sum}(\phi)\\
&=-\frac1N\sum_{i=1}^N\log\sigma(\Delta r_i)\\
&=-\mathbb E_{(x,y_w,y_l)\sim\widehat{\mathcal D}}
\log\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
\end{aligned}
$$

这三个写法依次是：总损失除以样本数、每条损失的平均值、从数据集中均匀抽一条记录时的平均损失。最后的期望符号只是在简写同一个平均过程。求和与均值的最优解相同，但固定学习率时梯度尺度不同。下文用 $\mathcal D$ 简写训练数据分布。

<details>
<summary>公式配图（可选）</summary>

![公式或实现配图](<pic/DPO_RM_Loss.png>)

![公式或实现配图](<pic/RM_L.png>)

![公式或实现配图](<pic/RM_L_log.png>)

![公式或实现配图](<pic/RM_argmax.png>)

![公式或实现配图](<pic/RM_Loss.png>)

![公式或实现配图](<pic/RM_Loss_e.png>)

</details>

## 4. RLHF 的 KL 正则化目标

RLHF 的第二步才开始训练生成模型：先固定已经学好的奖励模型，让待训练策略 $\pi$ 生成回答，再根据 RM 给出的分数调整策略。为了不让模型只顾迎合 RM、过度偏离原本的语言能力，还要保留一个冻结的参考策略 $\pi_{\rm ref}$。

我们定义 $\mathcal D_x$ 为训练时抽取问题的分布，$J_r(\pi)$ 为希望**最大化**的策略目标。外层先从 $\mathcal D_x$ 抽问题 $x$，内层再让当前策略生成回答 $y$：

$$
J_r(\pi)=\mathbb E_{x\sim\mathcal D_x}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{\rm KL}\bigl(\pi(\cdot\mid x)\|\pi_{\rm ref}(\cdot\mid x)\bigr)
\right].
$$

第一项是当前策略平均能拿到的奖励，鼓励它多生成高分回答；第二项是相对 reference 的分布偏离程度，前面的负号表示偏离会被扣分。$\beta>0$ 决定这种惩罚有多强。两个目标要一起看：单纯把某条回答的奖励推高，还要支付改变生成分布的代价。PPO 是近似求解这类目标的一种方法。


<details>
<summary>公式配图（可选）</summary>

![公式或实现配图](<pic/DPO_RL_Loss.png>)

</details>

## 5. KL 散度与期望

### 5.1 先定义离散 KL

先暂时离开语言模型，考虑同一个取值集合 $\mathcal Z$ 上的两个离散概率分布 $P$ 和 $Q$。$z$ 表示其中一个可能的结果，$P(z)$、$Q(z)$ 表示两个分布分别给这个结果分配的概率。我们定义从 $P$ 到 $Q$ 的 KL 散度为：

$$
D_{\rm KL}(P\Vert Q)=\sum_{z\in\mathcal Z}P(z)\log\frac{P(z)}{Q(z)}.
$$

这个式子本质上是**加权求和**：对于每个结果 $z$，先算对数概率比 $\log[P(z)/Q(z)]$，再乘以结果在 $P$ 下出现的概率 $P(z)$，最后遍历所有结果相加。因此，比较的数值来自两个分布，权重则始终来自 $P$。

### 5.2 把对数概率比视为随机变量

为了把这个加权和写成期望，我们引入随机变量 $Z$，让它服从分布 $P$，记作 $Z\sim P$。大写 $Z$ 表示还没有抽样的随机结果，小写 $z$ 表示它的一种具体取值。

接着定义函数 $g(z)=\log[P(z)/Q(z)]$：输入一个结果 $z$，输出两个分布在该结果上的对数概率比。因为输入 $Z$ 是随机的，$U=g(Z)$ 也成为一个随机变量。它不再表示“抽中了哪个结果”，而表示“抽中这个结果后得到多大的对数比”。

例如，若 $P$ 给两个结果的概率为 0.8 和 0.2，而 $Q$ 都给 0.5，那么 $U$ 分别以 0.8、0.2 的概率取值 $\log(0.8/0.5)$、$\log(0.2/0.5)$。它的期望就是这两个数按出现概率加权的平均值。一般地，按照离散随机变量的期望定义：

$$
\begin{aligned}
\mathbb E_{Z\sim P}[U] &=\sum_{z\in\mathcal Z}P(z)g(z)\\
&=\sum_{z\in\mathcal Z}P(z)\log\frac{P(z)}{Q(z)}.
\end{aligned}
$$

第一步枚举 $Z$ 的每种取值：取到 $z$ 的概率是 $P(z)$，这时 $U$ 的数值是 $g(z)$，所以两者相乘后求和。第二步只是把 $g(z)$ 换回刚才定义的对数比，没有新增假设。

### 5.3 识别为同一个加权和

现在比较第 5.1 节与刚得到的式子，会发现两边是完全相同的加权和。因此可以把 KL 散度写成下面的期望形式：

$$
D_{\rm KL}(P\Vert Q)
=\mathbb E_{Z\sim P}\left[\log\frac{P(Z)}{Q(Z)}\right].
$$

用于语言模型时，固定 prompt $x$，把随机变量换成回答 $Y\sim\pi(\cdot\mid x)$：

$$
D_{\rm KL}(\pi\Vert\pi_{\rm ref})
=\mathbb E_{Y\sim\pi(\cdot\mid x)}
\log\frac{\pi(Y\mid x)}{\pi_{\rm ref}(Y\mid x)}.
$$

在这个应用里，“可能的结果”就是一条完整回答，$P$ 换成当前策略 $\pi$，$Q$ 换成参考策略 $\pi_{\rm ref}$。因此可以理解为：让当前模型生成回答，计算这条回答在两个模型下的对数概率比，再对当前模型可能生成的所有回答取平均。抽样权重来自 $\pi$；直接改成从 reference 抽样，会得到另一个平均值。

若 $P(z)>0$ 而 $Q(z)=0$，相应 KL 为无穷；以下推导假设 reference 在相关响应上有正概率且配分函数有限。对数使用自然对数。


<details>
<summary>公式配图（可选）</summary>

![公式或实现配图](<pic/KL_SUM.png>)

![公式或实现配图](<pic/KL_Y.png>)

![公式或实现配图](<pic/KL_E.png>)

![公式或实现配图](<pic/KL_to_E.png>)

</details>

## 6. 从 RL 目标得到最优策略

接下来求解第 4 节的目标。先固定一个问题 $x$，把奖励函数 $r$ 和参考策略 $\pi_{\rm ref}$ 都视为给定；唯一要寻找的是回答分布 $\pi(\cdot\mid x)$。我们把这个问题上的目标记作 $J_r(\pi\mid x)$，省去外层问题平均，先回答“对于这一个问题，怎样分配回答概率最好”。

推导的思路是把目标整理成 KL 散度。只要得到“一个常数减去 KL”的形式，就可以利用 KL 非负来判断哪种回答分布最优。

### 6.1 展开 KL，并合并期望

利用第 5 节的等式：

$$
\begin{aligned}
J_r(\pi\mid x)
&=\mathbb E_{y\sim\pi}r(x,y)-\beta\mathbb E_{y\sim\pi}\log\frac{\pi(y\mid x)}{\pi_{\rm ref}(y\mid x)}\\
&=\mathbb E_{y\sim\pi}\left[r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\rm ref}(y\mid x)}\right].
\end{aligned}
$$

两项都对同一策略采样的回答取期望，所以可以合并。

### 6.2 最大化变成最小化，将奖励写进分母

为了把式子整理成 KL 的形状，我们定义一个辅助目标 $F(\pi\mid x)=-J_r(\pi\mid x)/\beta$。$F$ 只是同一优化问题的另一种写法，并不是新模型或新奖励。由于 $\beta>0$，给目标乘负数会翻转大小关系，因此：

$$
\arg\max_\pi J_r(\pi\mid x)=\arg\min_\pi F(\pi\mid x).
$$

用 $r/\beta=\log e^{r/\beta}$ 和 $\log a-\log b=\log(a/b)$，逐行得到：

$$
\begin{aligned}
F(\pi\mid x)
&=\mathbb E_{y\sim\pi}\left[\log\frac{\pi(y\mid x)}{\pi_{\rm ref}(y\mid x)}-\frac{r(x,y)}{\beta}\right]\\
&=\mathbb E_{y\sim\pi}\left[\log\frac{\pi(y\mid x)}{\pi_{\rm ref}(y\mid x)}-\log e^{r(x,y)/\beta}\right]\\
&=\mathbb E_{y\sim\pi}\log\frac{\pi(y\mid x)}{\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta}}.
\end{aligned}
$$

这里比较的是最优解，不是宣称 $\max J$ 与 $\min F$ 的数值相等。

### 6.3 分母尚未归一化，引入 Z

观察刚得到的分母：对于固定问题 $x$，它给每条回答 $y$ 一个正权重。我们将这个权重定义为 $w_x(y)=\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta}$。其中 reference 提供初始概率，指数项按奖励调整权重：奖励较高的回答得到更大的乘数。

这些权重相加未必等于 1，所以还不能直接叫概率。为了把它们构造成分布，我们定义**配分函数** $Z(x)$，也就是这个问题下所有回答权重的总和：

$$
\begin{aligned}
Z(x) &=\sum_y w_x(y)\\
&=\sum_y\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta}.
\end{aligned}
$$

这里的 $Z(x)$ 是依赖问题 $x$ 的归一化常数，与上一节用大写 $Z$ 表示的随机变量是不同对象；后面都按函数 $Z(x)$ 使用。在 $0<Z(x)<\infty$ 的条件下，将每个权重除以同一个总和，定义新的回答分布 $\pi^*$：

$$
\begin{aligned}
\pi^*(y\mid x) &=\frac{w_x(y)}{Z(x)},\\[4pt]
\sum_y\pi^*(y\mid x) &=\frac{\sum_yw_x(y)}{Z(x)}\\
&=1.
\end{aligned}
$$

式子右侧的求和验证了新分布的概率之和确实为 1。此时星号只是给这个候选分布起名；下一步把它代回目标，才证明它为什么最优。

### 6.4 代回目标，拆出与策略无关的常数

因为 $w_x(y)=Z(x)\pi^*(y\mid x)$，所以：

$$
\begin{aligned}
F(\pi\mid x)
&=\mathbb E_{y\sim\pi}\log\frac{\pi(y\mid x)}{Z(x)\pi^*(y\mid x)}\\
&=\mathbb E_{y\sim\pi}\left[\log\frac{\pi(y\mid x)}{\pi^*(y\mid x)}-\log Z(x)\right]\\
&=\mathbb E_{y\sim\pi}\log\frac{\pi(y\mid x)}{\pi^*(y\mid x)}-\log Z(x)\\
&=D_{\rm KL}(\pi\Vert\pi^*)-\log Z(x).
\end{aligned}
$$

第一行用 $Z(x)\pi^*(y\mid x)$ 替换 $w_x(y)$；第二行用对数的商法则，把归一化常数拆出来。对于同一个问题，所有回答对应同一个 $Z(x)$，所以第三行可以把 $\log Z(x)$ 提到回答期望之外。最后一行认出剩余期望正是 $D_{\rm KL}(\pi\Vert\pi^*)$。

还要注意：这里 $r$ 与 reference 都固定，因此 $Z(x)$ 也不依赖正在优化的 $\pi$。这才允许我们在寻找最优策略时将它视作常数。

### 6.5 用 KL 非负性得到最优策略

还原 $J=-\beta F$，得到严格的目标值恒等式：

$$
J_r(\pi\mid x)=\beta\log Z(x)-\beta D_{\rm KL}(\pi\Vert\pi^*).
$$

KL 非负，且在两个分布相同时为零，因此：

$$
\begin{aligned}
\arg\max_\pi J_r(\pi\mid x) &=\arg\min_\pi D_{\rm KL}(\pi\Vert\pi^*)\\
&=\pi^*,\\[4pt]
\max_\pi J_r(\pi\mid x) &=\beta\log Z(x).
\end{aligned}
$$

再对固定 prompt 分布取期望，只把常数换成 $\beta\mathbb E_x\log Z(x)$；逐 prompt 的最优分布不变。这是分布空间中的理想解，有限模型、有限数据与优化误差仍可能使实际训练结果偏离它。



<details>
<summary>公式配图（可选）</summary>

配图中的目标变换应按“最优解相同”理解；缩放目标或删除常数后，目标数值一般不相等。

![公式或实现配图](<pic/DPO_RL_trans_1.png>)

![公式或实现配图](<pic/DPO_Z.png>)

![公式或实现配图](<pic/DPO_pai*.png>)

![公式或实现配图](<pic/DPO_RL_trans_2.png>)

</details>

## 7. 从最优策略重参数化奖励

前面是在“已知奖励”的条件下求最优生成分布。DPO 接着反过来使用这个关系：既然奖励能决定最优策略，能否用策略概率来表示奖励，再直接训练策略去解释偏好数据？这样，第 3 节本来用来训练 RM 的偏好目标，就能改写为训练语言模型的目标。

### 7.1 从最优策略反解奖励

将第 6 节的式子两边除以 reference，再取对数：

$$
\begin{aligned}
\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}&=\frac{e^{r(x,y)/\beta}}{Z(x)},\\
\log\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}&=\frac{r(x,y)}{\beta}-\log Z(x),\\
r(x,y)&=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\log Z(x).
\end{aligned}
$$

### 7.2 对同一个 prompt 作奖励差，常数相消

对于同一个问题的两条回答，我们分别定义 $a_w=\log[\pi^*(y_w\mid x)/\pi_{\rm ref}(y_w\mid x)]$ 和 $a_l=\log[\pi^*(y_l\mid x)/\pi_{\rm ref}(y_l\mid x)]$。这两个记号只为缩短接下来的式子，分别表示优选回答、非优选回答相对 reference 的对数概率比。它们不是 PPO 的 advantage。

$$
\begin{aligned}
r(x,y_w)-r(x,y_l)
&=[\beta a_w+\beta\log Z(x)]-[\beta a_l+\beta\log Z(x)]\\
&=\beta a_w-\beta a_l.
\end{aligned}
$$

两条回答必须对应同一 $x$，才能消掉同一个 $Z(x)$；这一步没有把两个回答的生成概率直接作差。

### 7.3 先代入偏好概率，再代入 RM 似然

先在 Bradley–Terry 中替换奖励差：

$$
P(y_w\succ y_l\mid x)
=\sigma\!\left(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}\right).
$$

现在偏好概率只需要策略与 reference 的概率，就能计算出来。把它放回第 3 节的最大化对数似然，优化变量便从显式奖励函数转向了表示它的策略分布：

$$
\max_{\pi^*}\ \mathbb E_{\mathcal D}\log\sigma(\beta a_w-\beta a_l).
$$

### 7.4 用可训练策略参数化，改写成最小化 loss

把理论分布参数化为 $\pi_\theta$，再给整个最大化目标取负号：

$$
\mathcal L_{\rm DPO}(\theta)
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}
\right).
$$

实际训练时，我们用参数为 $\theta$ 的语言模型 $\pi_\theta$ 表示这个分布，reference 始终冻结。每次取一条偏好样本，让两个模型分别计算 $y_w$、$y_l$ 的序列 log-prob，形成两个对数比之差，再经过 sigmoid 和负对数得到 loss，最后只对 $\theta$ 反向传播。

这样，BT 模型、偏好数据和极大似然的含义都保留下来，而训练对象变成了生成模型。这不是假设每次更新中的 $\pi_\theta$ 都已等于理想最优策略。



<details>
<summary>公式配图（可选）</summary>

![公式或实现配图](<pic/DPO_trans_R.png>)

![公式或实现配图](<pic/DPO_trans_RM.png>)

![公式或实现配图](<pic/DPO_trans_RM_pai.png>)

</details>

## 8. 隐式奖励到底是什么

把训练完成的策略记作 $\pi_{\rm DPO}$。现在给定任意问题 $x$ 和一条确定的回答 $y$，冻结 DPO 模型与 reference，让二者对**同一串回答 token** 计算概率，而不是各自再生成一条新回答。由这两个概率，我们定义隐式奖励：

$$
\hat r(x,y)=\beta\left[\log\pi_{\rm DPO}(y\mid x)-\log\pi_{\rm ref}(y\mid x)\right],
$$

$$
\log\pi(y\mid x)=\sum_{t=1}^{|y|}\log\pi(y_t\mid x,y_{<t}).
$$

其中 $y_t$ 是回答的第 $t$ 个 token，$y_{<t}$ 是此前的回答前缀；序列概率按自回归分解为每步条件概率的乘积，取对数后就是上面的逐 token 求和。计算时通常直接累加 log-prob，避免完整序列概率相乘后过小。

这是**标量奖励分数**，可以为负，也可以大于 1，不是概率。标准序列 log-prob 使用回答 token 的求和；若改为长度平均，就改变了奖励定义。打分需统一 tokenizer、prompt 模板、回答边界与 EOS 约定。

| 概率比 | 隐式奖励（$\beta>0$） | 含义 |
| --- | --- | --- |
| $\pi_{\rm DPO}/\pi_{\rm ref}>1$ | 正 | 相对 reference，该回答的概率被增强 |
| $\pi_{\rm DPO}/\pi_{\rm ref}=1$ | 零 | 概率不变 |
| $\pi_{\rm DPO}/\pi_{\rm ref}<1$ | 负 | 相对 reference，该回答被抑制 |

例如取 $\beta=1$、两个模型给同一回答的概率分别为 $0.04$ 和 $0.01$，则奖励为 $\log4\approx1.386$。这反映模型学到的偏好，不保证回答客观正确。单独使用 DPO 模型的 log-prob 是另一种打分，并非上述隐式奖励。

## 9. 能否将它作为 PPO 的奖励

可以。缺少只依赖 prompt 的常数项不是障碍。在理想推导中令 $c(x)=\beta\log Z(x)$、$\hat r=r-c(x)$，则固定 prompt 分布下：

$$
J_{\hat r}(\pi)=J_r(\pi)-\mathbb E_x c(x),
\qquad
\nabla J_{\hat r}(\pi)=\nabla J_r(\pi).
$$

传统 Bradley–Terry RM 也仅通过奖励差学习偏好，不能仅靠这类数据识别唯一的奖励零点。PPO 需要数值奖励，不要求恢复这个零点。

进一步定义价值基线 $V(x)$ 为策略在问题 $x$ 上的预期回报，优势 $A(x,y)=r(x,y)-V(x)$ 表示“这条回答比通常预期好多少”。给所有回答的奖励都减去同一个 $c(x)$，准确的价值基线也会同步减去它。因此，把整段回答视为一个动作时：

$$
\begin{aligned}
\hat A(x,y) &=[r(x,y)-c(x)]-[V(x)-c(x)]\\
&=r(x,y)-V(x)\\
&=A(x,y).
\end{aligned}
$$

实际 token 级 PPO 的 GAE、价值估计误差、奖励预处理等会影响训练轨迹；不能据此声称两种奖励下每一步更新都数值相同。上述常数不变性针对固定 prompt、未额外引入长度相关折扣的响应级目标。

接入过程：

1. PPO 策略生成回答；冻结的 DPO 与 reference 对该回答打分。
2. 由奖励与价值估计构造优势；正奖励也可能低于预期，从而产生负优势。
3. PPO 用当前策略与采样旧策略的 token 概率比执行 clipped 更新：

$$
\begin{aligned}
\rho_t(\psi) &=\frac{\pi_\psi(y_t\mid x,y_{<t})}{\pi_{\rm old}(y_t\mid x,y_{<t})},\\[4pt]
J_{\rm clip} &=\mathbb E_t\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right).
\end{aligned}
$$

这里 $\psi$ 是正在训练的 PPO 策略参数，$\pi_{\rm old}$ 是采样这批回答时的旧策略，$A_t$ 是 token 位置 $t$ 的优势估计，$\epsilon$ 是裁剪范围。$\rho_t$ 衡量“同一个已采样 token 在更新前后变得多容易出现”；clipping 限制一次更新偏离旧策略过远。

这里有两种概率比：DPO/reference 用来定义奖励；PPO 当前策略/old policy 用于更新。PPO 另行使用的 KL 参考策略也必须明确配置，不应与这两个角色混用。直接最小化当前策略与 DPO 策略的 KL，更接近策略蒸馏。

“能够使用”不等于“值得再训练”：若使用同一个 reference、相同 KL 系数，并且从精确的 DPO 策略开始，则它已是这个隐式奖励目标的理想最优策略；再次优化该目标没有额外的理想收益。实际使用价值要看奖励质量、模型差异和训练设置。

## 10. 来源与阅读边界

- [DPO 原论文：§4–5、附录 A](https://arxiv.org/html/2305.18290)。最优策略与奖励等价类来自该论文；概率/奖励的数值例子是本文解释。
- [PPO 原论文](https://arxiv.org/abs/1707.06347)。这里只给出策略 surrogate，完整实现还包含价值拟合等组件。

