---
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
# backgroundImage: url('https://marp.app/assets/hero-background.svg')
marp: true
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
a[href='red'] {
    color: red;
    pointer-events: none;
    cursor: default;
    text-decoration: none;
}
</style>

<style>
img[alt~="right"] {
  display: block;
  margin:auto;
}
a[href='red'] {
    color: red;
    pointer-events: none;
    cursor: default;
    text-decoration: none;
}
</style>


![bg left:45% 80%](../images/course.webp)

# **LLM智能应用开发**

第11讲: 大语言模型解析 VIII
模型推理二三事

<!-- https://marp.app/ -->

---

# LLM结构的学习路径

* LLM结构解析(开源LlaMA)
* 训练流程：数据、损失函数、训练流程**
* **推理与部署流程**

---

# LLM推理过程二三事

* LLM二阶段推理
* KV-caching机制

---

#  LLM的输入输出

![w:1000 center](../images/2025/l11/pipeline.png)

---

#  LLM推理过程中实际涉及的步骤

![w:1000 center](../images/2025/l11/pipeline_with_tokens.png)

* LLM的一次推理输出logits，并非token
* 要得到token，还需通过Decoding strategy对logits进行解码

---


#  LLM推理过程中实际涉及的步骤

回顾一下LlaMA model的源码:
1. LlaMAModel获得最后一层DecoderLayer的输出
2. LM_head获得logits
3. Decoding strategy解码logits得到token
* 常用的Decoding strategy有：
  * Greedy decoding
  * Sampling
  * Beam search

---

# LLM的解码(decoding)策略

* 如果我们把logits(通过softmax转换为token的概率分布)作为输入，通常有如下解码策略：
  * 贪婪解码(Greedy decoding)：每次直接选择概率最高的token，简单高效，但并非全局最优
  * 采样(Sampling)：按一定的采样策略选择一个单词，增加生成过程的多样性，但可能会导致生成的文本不连贯
  * Beam search：通过维护一个长度为k的候选序列集，每一步(单token推理)从每个候选序列的概率分布中选择概率最高的k个token，再考虑序列概率，保留最高的k个候选序列

---

## Greedy decoding

![w:600 center](https://huggingface.co/blog/assets/02_how-to-generate/greedy_search.png)

---

## Beam search

![w:600 center](https://huggingface.co/blog/assets/02_how-to-generate/beam_search.png)


---


# 采样策略

* 一切从概率出发，叠加控制
  * 按条件概率随机采样
  * Top-k采样
  * Top-p采样 (核采样，Nucleus sampling)
  * Top-k+Top-p采样


---

# 采样策略: top-k采样

输入：南京大学计算机学院的课程有
概率分布: {算法:0.4, 操作系统:0.3, 计算机:0.2, 数据:0.05, ...}
* top-k采样，每次从概率最高的k个单词中进行随机采样
* 例如k=2，有可能生成的输出有
  * 南京大学计算机学院的课程有算法
  * 南京大学计算机学院的课程有操作系统
* 贪婪解码本质是一种top-k采样(k=1)

---

# 采样策略: top-p采样

* top-p采样，源自[The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751)
* 核心思路，重构采样集合
  * 给定token分布$P(x_i\mid x_{1:i-1})$，top-p集合$V^{(p)}\subset V$，使得$\sum_{x\in V^{(p)}}P(x\mid x_{1:i-1})\geq p$
  * 和top-k很像，区别在于在何处对分布进行截断

---

# HF关于采样策略的实现

* 参考:[top_k_top_p_filtering](https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py#L903) (老版本)
* 参考:
  * src/transformers/generation/logits_process.py
    * TopPLogitsWarper
    * TopKLogitsWarper
  * src/transformers/generation/utils.py
    * _get_logits_processor
      * 先topk，再topp

---

# 为什么要引入Temperature?

* logits本质是未归一化的偏好；模型常常过度自信→输出缺少多样性
* 只靠top-k/top-p截断无法改变分布“尖锐程度”，难以在“稳定 vs 创造力”之间细调
* Temperature提供一个连续控制杆：既可降低幻觉/重复，也可放开想象力
* 因此在实践中常把temperature视为“首个要调”的超参数

---

# Temperature详解

* 数学形式：$\tilde{p}_i = \text{softmax}(z_i / T)$，$T>0$
  * $T<1$：放大logits差异，分布更“尖锐”，输出更确定
  * $T>1$：压平logits，概率更平均，输出更随机
* 实践经验
  * 0.1-0.5：摘要/QA等需要确定性的任务
  * 0.7-1.3：创作/头脑风暴
  * $T\rightarrow 0$: 接近贪婪解码；$T\rightarrow \infty$: 接近均匀采样
* 注意：温度变化与top-k/top-p存在耦合，先锁定一个，再微调另一个，否则难以定位问题

---

# 为什么要引入Penalty?

* 纯靠temperature/top-k/top-p仍可能出现短循环、口头禅、提示词泄露等模式崩溃
* 不同任务对“重复”和“长度”容忍度不同，需要有针对性的约束手段
* Penalty机制通过修改logits或得分，打破模型对高频token的偏好，提升可控性
* 有的为“软惩罚”(repetition/presence)，有的为“硬约束”(no_repeat_ngram)，可组合使用
  
---

# 常见Penalty机制

* repetition penalty (HF实现)
  * 对生成过的token乘以$1/\text{penalty}$或$\text{penalty}$，惩罚重复；>1.0时抑制循环
* presence / frequency penalty (OpenAI)
  * presence：是否出现过→每次出现扣常数
  * frequency：出现次数越多扣得越多→抑制关键词刷屏
* length penalty (Beam Search)
  * 调整对长序列的偏好，$\text{score}/((5+|y|)^\alpha / (5+1)^\alpha)$

---

# Penalty调参Tips

* 问答/代码：从`repetition penalty=1.05`或`frequency penalty=0.5`起步
* 创作/诗歌：降低重复惩罚，改用`presence penalty=0.3`确保主题多样
* 长对话：设`no_repeat_ngram=3-4`，再酌情增加repetition penalty
* Beam search翻译：`length_penalty`在0.6-1.0之间调节句长
* 调参顺序：temperature → top-k/p → penalty → 其他限制（stop words, max_tokens）

---

# LLM推理之运行时两大阶段

* 基于LLM自回归生成(autoregressive generation)的特点
  * 逐token生成，生成的token依赖于前面的token
  * 一次只能生成一个token，无法同时生成多个token
* LLM生成过程分为两个阶段
  * Prefill phase
  * Decoding phase

---

# LLM推理第一阶段: Prefill

输入token序列，输出下一个token

![w:900 center](../images/2025/l11/prefill.jpg)

---

# LLM推理第二阶段: Decoding

![w:700 center](../images/2025/l11/decoding1.jpg)
![w:700 center](../images/2025/l11/decoding2.jpg)

---

# LLM推理第二阶段: Decoding

![w:700 center](../images/2025/l11/decoding2.jpg)
![w:700 center](../images/2025/l11/decoding4.jpg)

---

# LLM完成推理后，解码

将生成的token序列解码成文本

![w:700 center](../images/2025/l11/decodingAll.jpg)

---

# LLM二阶段推理解析

* 将LLM当作函数，输入是token序列，输出是下一个token
* LLM通过自回归(autoregressive generation)不断生成"下一个token"
* 脑补下当LLM接收到输入的token序列后如何进行下一个token的推理

<div style="display:contents;" data-marpit-fragment>

![w:1000 center](../images/2025/l11/pipeline_with_tokens.png)

</div>

---

# LLM推理过程会产生一些中间变量

第一个"下一个token"生成: 输入token序列"经过"(调用forward方法)N层Decoder layer后，的到结果
细看其中一层Decoder layer,frward方法会返回若干中间输出，被称之为激活(activation)
![w:700 center](../images/2025/l11/pipeline.png)


---

# Prefill phase

* 第一个"下一个token"生成过程被称之为Prefill阶段
* 为何特殊对待？
  * 计算开销大
* 简单推导一下一次LLM的推理过程的计算开销

![w:700 center](https://cdn-uploads.huggingface.co/production/uploads/65263bfb3177c2a794997821/BGKtYLqM1X9o72oc9NW8Y.png)

---

# 计算开销

* 符号约定
  * b: batch size
  * s: sequence length
  * h: hidden size/dimension
  * nh: number of heads
  * hd: head dimension

---

# 计算开销


* 给定矩阵$A\in R^{1\times n}$和矩阵$B\in R^{n\times 1}$，计算$AB$需要$n$次乘法操作和$n$次加法操作，总计算开销为$2n$ (Floating-Point Operations (FLOPs))
  * FLOPs: floating point operations
* 给定矩阵$A\in R^{m\times n}$和矩阵$B\in R^{n\times p}$，计算$AB$中的一个元素需要$n$次乘法操作和$n$次加法操作，一共有$mp$个元素，总计算开销为$2mnp$ 

---

# Self-attn模块

* 第一步计算: $Q=xW_q$, $K=xW_k$, $V=xW_v$
  * 输入x的shape: $(b,s,h)$，weight的shape: $(h,h)$
  * Shape视角下的计算过程: $(b,s,h)(h,h)\rightarrow(b,s,h)$
    * 如果在此进行多头拆分(reshape/view/einops)，shape变为$(b,s,nh,hd)$，其中$h=bh*hd$
  * 计算开销: $3\times 2bsh^2\rightarrow 6bsh^2$

---

# Self-attn模块

* 第二步计算: $x_{\text{out}}=\text{softmax}(\frac{QK^T}{\sqrt{h}})VW_o+x$
  * $QK^T$计算: $(b,nh,s,hd)(b,nh,hd,s)\rightarrow (b,nh,s,s)$
    * 计算开销: $2bs^2h$ (为理解方便，暂且忽略softmax的计算开销)
  * $\text{softmax}(\frac{QK^T}{\sqrt{h}})V$计算: $(b,nh,s,s)(b,bh,s,hd)\rightarrow(b,nh,s,hd)$
    * 计算开销: $2bs^2h$
* 第三步$W_o$计算: $(b,s,h)(h,h)\rightarrow(b,s,h)$
  * 计算开销: $2bsh^2$
* Self-attn模块总计算开销: $8bsh^2+4bs^2h$
---

# MLP模块

$$x=f_\text{activation}(x_{\text{out}}W_{\text{up}})W_{\text{down}}+x_{\text{out}}$$
* 第一步计算，假设上采样到4倍
  * Shape变化:$(b,s,h)(h,4h)\rightarrow(b,s,4h)$
  * 计算开销: $8bsh^2$
* 第二步计算，假设下采样回1倍
  * Shape变化:$(b,s,4h)(4h,h)\rightarrow(b,s,h)$
  * 计算开销: $8bsh^2$
* MLP模块总计算开销: $16bsh^2$


---

# Decoder layer模块计算开销

* Self-attn模块计算开销: $8bsh^2+4bs^2h$
* MLP模块计算开销: $16bsh^2$
* Decoder layer模块计算开销: $24bsh^2+4bs^2h$

* 以上为一次推理的计算开销，开销为sequence的平方级别

---

## Decoding phase

* 当第一个"下一个token"生成完毕后，LLM开始"自回归推理"生成
* 第二个"下一个token"
  * 输入x的shape: $(b,s+1,h)$，继续以上推理过程，计算开销$O(s^2)$
* 第三个"下一个token"
  * 输入x的shape: $(b,s+2,h)$，继续以上推理过程，计算开销$O(s^2)$
* 第n个"下一个token"
  * 输入x的shape: $(b,s+n-1,h)$，继续以上推理过程，计算开销$O(s^2)$

---

## Decoding phase

* 每次自回归推理过程，都需要平方级别的开销？
  * 且包含了计算开销和内存开销


---

# 回顾Self-attn中$QK^T$的计算过程

* 第一个"下一个token"
  * $QK^T$计算: $(b,nh,s,hd)(b,nh,hd,s)\rightarrow (b,nh,s,s)$
* 第二个"下一个token"
  * $QK^T$计算: $(b,nh,s+1,hd)(b,nh,hd,s+1)\rightarrow (b,nh,s+1,s+1)$
* 考虑自回归特性，$(s,s)$和$(s+1,s+1)$为下三角阵
  * $(s+1,s+1)$的前$s$行$s$列就是$(s,s)$
* 考虑复用$(s,s)$？

---

# LLM自回归过程中的kv复用

* 要复用什么，还得从需求出发
* 需求: 生成"下一个token" 
* Decoder layers之后的lm_head计算
  * shape视角: $(b,s,h)(h,h)\rightarrow (b,s,h)$
* 生成第二个"下一个token"
  * shape视角: $(b,s+1,h)(h,h)\rightarrow (b,s+1,h)$
  * 第二个"下一个token"的logits在$(b,s+1,h)$中第二个维度$index_{s+1}$处，该logits只受$(b,s+1,h)$中第二个维度$index_{s+1}$处的$h$值和对应的$w$影响

---

# LLM自回归过程中的kv复用

* 所以，真正自回归计算的部分是$(b,s+1,h)$中的第二个维度$index_{s+1}$的部分，复用的是用于计算$(b,s+1,h)$中第二维度$index_{s+1}$的数值
  * shape的视角: $(b,s+1,h)\rightarrow (b,1,h)$
* 整个self-attn计算过程中，只有$QK^T$中的$K$和$\text{softmax}(\frac{QK^T}{\sqrt(h)})V$中的$V$需要复用
  * 为K和V构建缓存: 即KVCache
  * Q依赖当前token的Embedding，必须实时计算；Attn输出和MLP输出也会被LayerNorm/残差更新，无法直接重用

---


![w:1000 center](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/key-value-caching_.png)
