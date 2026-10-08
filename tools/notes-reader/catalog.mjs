// Navigation groups and article labels. Paths are relative to notes/llm/.
export const sections=[
 {id:'post-training',title:'后训练',description:'理解偏好如何成为学习信号，以及策略优化、预测与自蒸馏的目标和训练过程。',groups:[
  ['偏好与策略优化',['post-training/DPO.md','post-training/PPO.md','post-training/GRPO.md','post-training/GXPO.md','post-training/pair-grpo-family.md']],
  ['预测与自蒸馏',['post-training/MTP.md','post-training/SD-Zero.md']],
 ]},
 {id:'infra',title:'Infra',description:'从模型的计算结构出发，梳理精度、显存、并行训练与推理性能，逐步扩展到集群和调度。',groups:[
  ['模型基础',['infra/transformer结构.md','infra/激活函数.md','infra/MLA.md']],
  ['训练与并行',['infra/混合精度.md','infra/分布式训练-数据并行.md','infra/分布式训练-模型并行.md','infra/分布式训练-流水线并行.md']],
  ['推理与性能',['infra/显存分析.md','infra/Flash_attention.md']],
 ]},
];
export const pending=new Set(['PPO','GRPO','GXPO','MLA']);
export const titles={DPO:'DPO：偏好、奖励与 PPO',PPO:'PPO',GRPO:'GRPO',GXPO:'GXPO','pair-grpo-family':'Pair-GRPO 精读',MTP:'MTP 多 token 预测','SD-Zero':'SD-Zero 精读',transformer结构:'Transformer 与 RoPE',激活函数:'激活函数与 SwiGLU',MLA:'MLA',混合精度:'混合精度','分布式训练-数据并行':'数据并行、ZeRO 与 FSDP','分布式训练-模型并行':'TP、EP、SP 与 CP','分布式训练-流水线并行':'流水线并行',显存分析:'参数量、显存与计算量',Flash_attention:'FlashAttention','revision-notes':'修订与公式核对'};
// Short discovery text for cards; the complete Markdown remains in each article.
export const summaries={
 DPO:'从偏好比较和 KL 约束推导 DPO，理解隐式奖励、参考模型，以及它们与 PPO 的关系。',
 PPO:'待补充：策略概率比、裁剪目标、价值函数与 GAE，以及完整训练流程。',
 GRPO:'待补充：组内相对优势、策略优化目标与训练流程。',
 GXPO:'方法名称与来源待确认，后续补充定义、优化目标与推导。',
 'pair-grpo-family':'梳理 Pair-GRPO 的成对比较、组内信号与优化目标，理解各个量在训练中的作用。',
 MTP:'一次预测多个未来 token：训练目标、计算结构，以及与常规自回归预测的关系。',
 'SD-Zero':'从教师与学生的条件分布出发，理解自蒸馏目标、训练信号和方法的适用范围。',
 transformer结构:'沿着 Transformer 的计算过程理解注意力、张量形状与 RoPE 旋转位置编码。',
 激活函数:'理解非线性激活的作用，并从 SiLU 的定义推导 SwiGLU 的门控结构。',
 MLA:'待补充：注意力结构、KV Cache 压缩与推理时的计算过程。',
 混合精度:'区分计算精度与存储精度，理解 FP16、BF16、主权重和损失缩放。',
 '分布式训练-数据并行':'从梯度同步到状态切分，梳理 DDP、ZeRO 与 FSDP 的显存和通信开销。',
 '分布式训练-模型并行':'结合矩阵运算理解张量、专家、序列和上下文并行的切分方式与通信。',
 '分布式训练-流水线并行':'将模型分段执行，理解微批次、流水线气泡、调度与激活显存之间的关系。',
 显存分析:'根据模型结构估算参数量、训练状态、激活与 KV Cache，并区分计算和带宽瓶颈。',
 Flash_attention:'从稳定 Softmax 推导分块计算与在线合并，理解 FlashAttention 如何减少显存读写。',
};
