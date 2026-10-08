# 修订与公式核对

修订日期：2026-10-08。范围为 `后训练` 与 `Infra` 的 16 篇笔记：12 篇有正文，4 篇原本为空或只有标题。

## 1. 文件保留与分类

| 位置 | 用途 |
| --- | --- |
| `archive/post-training/`、`archive/infra/` | 修订前逐文件复制，含原 Markdown、图片与其他依赖 |
| `post-training/`、`infra/` | 修订后的 Markdown，以及由其生成的同名 HTML |
| [阅读入口](index.html) | 统一导航、全文搜索与分类 |
| `_reader/original-manifest.json` | 原始 113 个文件的大小与 SHA256 清单 |
| `_reader/formula-image-audit.json` | 原图片与修订文字的对应清单 |

原图片不重绘、不覆盖。图内有误时在相邻正文标明；保留原图不代表认可其每个公式。Markdown 是编辑源文件，HTML 是可重新生成的阅读版本。

Infra 保留原名称，范围涵盖训练、推理与基础设施；现按模型基础、训练与并行、推理与性能分组，后续可以加入集群与调度。

## 2. 核心修正

| 笔记 | 主要修正 |
| --- | --- |
| [DPO](post-training/DPO.md) | 区分概率、奖励与优势；补齐最优策略推导；固定 DPO/reference 可以给 PPO 提供标量奖励 |
| [MTP](post-training/MTP.md) | 区分 Meta 并行头与 DeepSeek 顺序模块；补输入位移与训练目标；推测解码区分 greedy 与保分布采样 |
| [SD-Zero](post-training/SD-Zero.md) | 限定 RLVR、SDFT/SDPO 的描述；纠正 62.1、49.7/49.5 的解读；采样预算不等于等 FLOPs |
| [Pair-GRPO](post-training/pair-grpo-family.md) | KL 符号、遗漏的梯度共同项、跨 pair 期望；两个绝对概率可同时下降的反例；支持集、软约束与共享参数 |
| [Transformer](infra/transformer结构.md) | RoPE 基数与角频率、split-half 切片、相对位置恒等式；普通 MLP 与 SwiGLU |
| [激活函数](infra/激活函数.md) | SiLU 非全域单调、负半轴导数可为零；自门控与可训练投影的区别 |
| [混合精度](infra/混合精度.md) | dtype、AMP、优化器状态、loss scaling 与 FP32 主权重的不同作用 |
| [显存分析](infra/显存分析.md) | Qwen2.5-7B 约 7.616B 参数；GQA/MHA KV；GB/GiB；静态状态/峰值；MFU/HFU |
| [FlashAttention](infra/Flash_attention.md) | 二次计算仍在；补输出累积；区分 FA2/FA3 的优化 |
| [数据并行](infra/分布式训练-数据并行.md) | 通信语义、N−1 步、bit/byte 换算；ZeRO 64 倍口径；FSDP 聚合当前单元参数 |
| [模型并行](infra/分布式训练-模型并行.md) | 行/列 TP 梯度；稳定词表 softmax；SP/CP、EP 的定义；非专家层处理不同数据 |
| [流水线并行](infra/分布式训练-流水线并行.md) | 阶段输出传递、micro-batch 与 DP；气泡假设；激活显存补序列长度、dtype 与层内系数 |

## 3. 公式与原图差异

完整图片引用列表见核对清单。下面列出需要特别注意的部分。

| 原图 | 核对结论 |
| --- | --- |
| `DPO_RL_trans_1.png`、`DPO_RL_trans_2.png` | 删常数或缩放后的目标值不能直接相等；正文给恒等式与 argmax/argmin 等价关系 |
| DPO 的 RM/DPO loss 图 | 图用最大化 log-likelihood，正文负号写作最小化 loss；BT、KL、Z 与奖励重参数化保持相同数学含义 |
| MTP 三张架构图 | 与预测深度、shifted-token 输入对应；训练模块不要求推理时全启用 |
| FlashAttention 分块 softmax 与算法图 | 统一到共同 max；图维护归一化 O，正文维护未归一化 u 后除以分母，两者等价 |
| `decoder.png` | 含 learned position embedding、Top Query Layer 的具体架构，不是所有 LLaMA/Qwen 模型的结构 |
| `image-2.png` | 图采用列向量，正文行向量，权重相应转置后等价；普通 MLP 不等于 SwiGLU |
| RoPE 两张代码图 | split-half 从 d//2 开始；截图额外 scaling 与基础公式区分 |
| `image-6.png` | 简化 ACT2FN 表不能证明所有 GELU 变体相同；Swish/SiLU 等价有参数条件 |
| 模型配置与模块截图 | H=3584、I=18944、L=28、GQA 4 KV 头，与参数量逐项对照 |
| `image.png`（混合精度流程） | 图中加梯度，最小化 loss 时应减梯度 |
| Attention 展开图 | 正文补缩放、mask、归一化与 GQA 假设 |
| `gpu_memory.png` | gradients(fp15) 应为 FP16；12P 状态包含 FP32 主权重 |
| `ZeRO-1-cost.png` | 3T 对应特定通信流程，优化后可不同，不能作为阶段定义 |
| `tp_col_split_gradient_1.png` | 各输出分片必须使用各自 Gi，正确式为各 Gi Wiᵀ 求和 |
| `tp_split_output_layer_softmax.png` | 图省略全局最大值归约，正文补稳定计算 |
| `tp_whole.png` | 注意力末端应记 Wo；普通 MLP 图不是完整 SwiGLU |
| `ep+dp+tp+pp.png` | 非专家层复制参数但处理不同 batch，不是重复计算同一数据 |
| 流水线四张图 | 对应均衡 stage 的理想时间表，不能认为气泡已完全消除 |
| `sd-zero-algorithm.png` | teacher 条件中的 r 为对错提示简写，KL 方向一致；底部方法比较仅限定为论文设置 |
| 训练日志、性能截图 | 保留历史证据，缺少完整环境记录，未冒充本次实测 benchmark |

## 4. 推导完整性补审

用户指出原图包含连续推导后，重新对照了原 Markdown 和成组公式图。结论是：**图片没有丢，但第一版文字化确实压缩了若干中间步骤。** 图片完整、公式能渲染、数值例子通过，均不能单独证明推导已逐步转写；本次补审专门检查这个差别。

| 推导链 | 第一版被压缩的部分 | 当前阅读位置与补全内容 |
| --- | --- | --- |
| DPO 的 RM 训练 | 负对数求和到均值/期望；乘积取对数的展开 | [DPO §3](post-training/DPO.md#3-显式-rm-的极大似然训练)：单条概率 → 联合似然 → 对数求和 → argmax → 负号 → 经验均值 |
| DPO 的 KL 期望 | 原图先定义随机变量，再套期望定义 | DPO §5：逐一保留随机变量、加权求和、识别为 KL 的步骤 |
| DPO 的最优策略 | 奖励并入 log、未归一化分母、Z 的作用、常数拆分 | DPO §6：五个小节逐步展开，并区分目标值相等和最优解等价 |
| DPO 代回偏好目标 | 反解奖励、两条回答的常数相消、概率到似然 | DPO §7：四个小节展开，说明每步变换与变量含义 |
| FlashAttention | 局部概率、重缩放后的分子、分母求和、输出换基准 | [FlashAttention §2–4](infra/Flash_attention.md)：逐行展开指数与求和，并推导图中归一化 O 与正文 u 的等价关系 |
| RoPE | 共轭、指数合并、复数四项展开、拼回 split-half | [Transformer §4–5](infra/transformer结构.md)：保留展开过程，并解释复数旋转如何变成张量操作 |
| 张量并行 | 输入/权重链式法则、词表交叉熵、embedding 梯度累加 | [模型并行 §2–3](infra/分布式训练-模型并行.md)：矩阵微分展开；保留不同输出分片的梯度，纠正原图错误 |
| 参数/显存/FLOPs | 各个矩阵与激活项怎样形成合计系数 | [显存分析](infra/显存分析.md)：恢复形状表、11/5/19/4/34 分项、KV 矩阵到最后一行、各投影 FLOPs、6PT/8PT 来历 |
| ZeRO 与流水线 | 状态分项除以 N、总容量与忙碌量、边界激活相加 | [数据并行](infra/分布式训练-数据并行.md)、[流水线并行](infra/分布式训练-流水线并行.md)：逐项推导合计公式 |
| Pair-GRPO | 修正后的共同项分解上一版只直接给等式 | [Pair-GRPO §5.2](post-training/pair-grpo-family.md)：先拆均值与差值，再代回并收集两类梯度项 |

本次重点补齐原稿已有的推导链，以及理解修正公式所需的中间步骤；不是为所有基础数学结论另写完整教材式证明。MTP 原稿以架构图为主，SD-Zero 以目标定义/算法流程为主，未发现需要像 DPO 一样恢复的多图代数链；四篇待补文件仍不算完整笔记。原图与原稿继续保持原样。

2026-10-08 完整复制至 `yuan-notes/notes/llm/`，当前正文、图片、原稿和生成工具均在本仓库内。原目录保留，两份独立，不建立跨目录链接或同步。后续主要维护本仓库的 Markdown；`archive/` 保留原稿。原始 SHA256 清单中的历史路径名只用于本仓库内部的校验映射，不访问原知识库。

随后按阅读体验重新整理正文：恢复“问题、对象定义、推导理由、公式含义”的解释顺序；删除正文中按图片文件名逐项映射的编辑过程描述；必要的图内笔误放入配图说明。本页保留修订历史，文章正文以概念与推导为主。

## 5. 待补笔记

- [PPO](post-training/PPO.md)、[GRPO](post-training/GRPO.md)：原文件为空，只标状态和相关链接。
- [GXPO](post-training/GXPO.md)：原文件为空，缩写出处未明确，未擅自解释成其他算法。
- [MLA](infra/MLA.md)：原文件只有标题，不算作已完成的技术笔记。

## 6. 验证边界

验证包括原稿与图片 SHA256、公式编译、关键数值/数学恒等式、本地链接及浏览器功能；结果由脚本生成至 `_reader/verification.json` 与 `_reader/browser-verification.json`。

技术判断依据各笔记所列原论文与官方文档。论文实验数字属于作者报告，并非本次复现；GPU/NPU 分布式代码未做硬件运行测试。长论文原稿中的 PDF/e-print 哈希及源码清单是历史阅读记录，本次未重新下载对应存档核验。

## 7. 重新生成

在 `yuan-notes` 根目录执行：

```sh
cd tools/notes-reader
npm ci --ignore-scripts --no-audit --no-fund
npm run build
npm run verify
```

直接打开 `notes/llm/index.html` 即可离线阅读，公式字体与搜索索引都在本地。修改当前 Markdown 后重新生成 HTML；`archive/` 始终保留原稿。
