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

第9讲: Sparse Attention

---

## 背景与动机

### 回顾 Self-Attention 机制

$$
A = Softmax(\frac{QK^T}{\sqrt{d}})V
$$

其中 $Q$ shape: [bs, nh, q_len, hd], $KV$ shape: [bs, nh, kv_len, hd]，

若记 $N$ 为序列长度，则 $q\_len \in O(N), kv\_len \in O(N)$，

因此 Attention 的计算复杂度是 $O(N^2)$。

---

### 长文本场景

在输入序列较长的应用场景中，例如：
* 生成长文档总结
* 大规模代码分析
* 多文档问答
* 多轮对话
  
平方级的复杂度导致 Self-Attention 计算速度显著变慢，限制了大模型在这些场景中的广泛应用。

---

### 思考

本质上，计算 Self-Attention 的过程就是让所有 token “注意” 其他所有 token 的过程。

联想：平时我们在阅读文章时，并不会一个字一个字地理解，往往是通读一遍抓住关键词，就能掌握大致意思。

🔍 **问题：真的需要每个 token 都“注意”所有其他 token 吗？**

---

### 观察

在 llama3 模型上输入 "A is B. C is D. A is" 并设置 max_new_tokens=1 (prefill) 得到以下 Attention Heatmap（部分），其中颜色由暗到亮（由紫到黄）表示激活值从小到大（0-1）。

<img src="../images/l9/sparsity.png" width="100%" />

---

### 稀疏性（Sparsity）

在 Self-Attention 计算过程中:
* **现象**：注意力矩阵中，大部分权重接近 0;
* **启发**：可以只保留最关键的连接参与计算；
* **目标**：减少无用计算，提高效率。

---

## Sparse Attention

Sparse Attention 可以如下表述：

$$
\hat{A} = Softmax(\frac{QK_s^T}{\sqrt{d}})V_s
$$

其中 $K_s, V_s$ 均为从完整 $KV$ 中筛选出来的部分 $KV$，且 $s \ll N$。

例如，对于 $N=8k$ 的输入，可以取 $s=2k$，稀疏比例达到 $\frac{1}{4}$，由此将计算量降低到 $\frac{1}{16}$。

---

### Sparse Attention 分类

1. 根据如何筛选关键 token：
   * Static pattern
   * Dynamic pattern
2. 根据是否需要训练：
   * Training-free
   * Training-based

由于训练需要消耗大量资源，学术界主要聚焦在 Training-free 或只要简单训练的方法上；而部分大模型厂商（如 DeepSeek）正在尝试 Training-based Sparse Attention。

---

### Static pattern 方法简介

1. Sliding windows: 维护一个固定大小的窗口，保留最近的 tokens 参与计算，其余全部丢弃。

   * 优点：实现简单，计算复杂度降低到 $O(N)$；
   * 缺点：精度损失较大，尤其是在长度超过预训练长度后大幅下降。

![sliding_window center](../images/l9/sliding_window.png)

---

### Static pattern 方法简介

2. Attention sinks: [StreamingLLM](https://arxiv.org/abs/2309.17453) 发现注意力权重往往会集中在首 token 上，将这一现象称为 attention sinks。基于该发现，StreamingLLM 在 sliding window 的基础上进一步保留 attention sinks，降低了长文本场景下稀疏导致的精度损失。

![streamingllm center](../images/l9/streamingllm.png)

---

### Static pattern 方法简介

总体上看，这些固定的模式往往不能适应文本生成中变化的关键 token，或多或少会有较显著的精度损失。

---

### Dynamic pattern 方法简介

1. [MInference](https://arxiv.org/abs/2407.02490) 通过观察注意力矩阵，总结出三种常见模式，根据输入动态选择最合适的模式，从而加速 prefill 阶段：

<img src="../images/l9/minference.png" width="100%"/>

---

### Dynamic pattern 方法简介

2. [Quest](https://arxiv.org/abs/2406.10774) 采用分页设计，估计每个 KV page 与当前 Q 的相似度，动态选择最相似（激活值最高）的 pages 参与计算：
![quest height:450 center](../images/l9/quest.png)

---

### Dynamic pattern 方法简介

* 优点：相较于 static pattern，dynamic pattern 类的方法精度更高；

* 缺点：由于计算最合适的 tokens 会引入一定 overhead，综合下来会比简单的 static pattern 方法慢（但是相比 dense attention 还是有加速效果）;同时，如何设计选择算法也依赖经验（启发式）。

---

### Training-based 方法简介

1. [NSA](https://arxiv.org/pdf/2502.11089) 通过门控机制融合了粗粒度 token 压缩、细粒度的 token 选择和滑动窗口这三个模块的输出，从而达到稀疏效果。

![NSA height:300 center](../images/l9/nsa.png)

---

### Training-based 方法简介

2. [DSA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) 主要在 [MLA](https://arxiv.org/abs/2412.19437) 的基础上加上了稀疏模块（绿色部分）。本质上，Lightning Indexer 利用量化后（FP8）的 qk 计算 attention，根据这一轻量计算选择与 Q 最相似的 K。
![DSA height:400 center](../images/l9/dsa.png)

---

### Training-based 方法简介

总体而言，Training-based 方法由于其成本高，当前大模型厂商少有投入。但从 DeepSeek 公布的效果来看(NSA)，训练后的原生 Sparse Attention 精度几乎无损甚至能反超 Dense Attention，推理速度也更快。
![nsa_performance height:350 center](../images/l9/nsa_performance.png)