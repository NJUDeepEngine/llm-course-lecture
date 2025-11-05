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

第10讲: 大语言模型解析 VII  
模型训练二三事

<!-- https://marp.app/ -->

---

## LLM结构的学习路径

* LLM结构解析(开源LlaMA)
* **训练流程：数据、损失函数、训练流程**
* 推理与部署流程

---

## LLM训练的三方面

* 输入侧
  * 数据准备，批处理(batch)策略
* 输出侧
  * 目标函数
* 训练执行
  * 训练框架，评估与保存

---

## LLM训练输入侧

* 数据准备
   * 原始语料整理 → 数据集划分 → Tokenizer对齐
* 批处理(batch)策略
   * Dataset / DataLoader / collate_fn


---




## LLM训练输出侧

* 目标函数
   * 语言模型损失、指令微调损失、对齐/奖励建模


--- 

## LLM训练执行

* 训练流程
   * Trainer快速上手 vs. 手写训练脚本
* 评估与保存
   * 指标、日志、Checkpoint管理

---

## 数据准备：从语料到可训练样本

* 数据来源：开源语料、业务日志、合成数据
* 质量控制：清洗噪声、去重复、敏感信息脱敏
* 样本结构：明确字段（instruction/input/output/messages）
* 划分策略：train/validation/test避免数据泄漏
* Tokenizer对齐：确保训练与推理共享词表及预处理

---

## 数据准备：流程概览

1. 从公开数据集中载入（示例：Alpaca）
2. 规则过滤并清洗文本
3. 统一字段/模板便于拼接
4. 拆分训练与验证集
5. Tokenizer对齐

---

## Step1: 载入Alpaca数据

```python
from datasets import load_dataset

raw_ds = load_dataset("tatsu-lab/alpaca")
train_raw = raw_ds["train"]
```

* 直接使用HF镜像，避免手动下载/解析
* `train_raw`为`Dataset`对象，可继续链式操作

---

## Step2: 规则过滤

```python
def keep_example(example):
    answer = example["output"].strip()
    return len(answer) > 5

filtered = train_raw.filter(keep_example)
```

* `filter`会自动并行处理，返回新的`Dataset`
* 可叠加敏感词、长度、语言检测等逻辑

---

## Step3: 统一字段模板

```python
def build_messages(example):
    user_prompt = example["instruction"].strip()
    if example["input"]:
        user_prompt += "\n" + example["input"].strip()
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["output"].strip()},
        ]
    }
structured = filtered.map(build_messages)
```
* `messages`结构兼容LLaMA Factory等框架
* 也可保持`instruction/input/output`三字段

---

## Step4: 划分训练/验证

```python
split_data = structured.train_test_split(test_size=0.02, seed=42)
train_data = split_data["train"]
val_data = split_data["test"]
```

* `train_test_split`基于随机种子保证可复现
* 按任务标签分层：`train_test_split(..., stratify_by_column="category")`
  
---

## Step4: 划分训练/验证

* 按长度分桶，先增加分桶字段，再使用`stratify_by_column`保持长短样本分布一致
```python
def add_length_bucket(example):
    length = len(example["messages"][0]["content"].split())
    bucket = min(length // 200, 4)  # 0-4 共5档
    example["len_bucket"] = bucket
    return example
bucketed = structured.map(add_length_bucket)
split_bucketed = bucketed.train_test_split(
    test_size=0.02,
    seed=42,
    stratify_by_column="len_bucket",
)
```
---

## Step5: Tokenizer对齐（初始化）

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b")
tokenizer.pad_token = tokenizer.eos_token
```

* 统一tokenizer，避免训练/推理词表不一致
* 某些模型需手动指定`pad_token`

---

## Step5: Tokenizer对齐（构造模板）

```python
def build_prompt(example, max_length=2048):
    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]
    return (
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{assistant}{tokenizer.eos_token}"
    )

print(build_prompt(train_data[0]).splitlines()[:4])
```

* 显式构造模板，便于检查角色标记
* 可换成自定义系统提示/多轮对话

---

## Step5: Tokenizer对齐（编码+标签）

```python
def tokenize(example, max_length=2048):
    text = build_prompt(example, max_length=max_length)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
```

* 自回归任务直接复制`input_ids`为`labels`
* 若需mask部分prompt，可将对应的label置为`-100`

---

## Step5: Tokenizer对齐（批量处理）

```python
tokenized_train = train_data.map(
    tokenize,
    remove_columns=train_data.column_names,
)

tokenized_val = val_data.map(
    tokenize,
    remove_columns=val_data.column_names,
)
```

* `map`自动并行，适合大规模数据
* `remove_columns`保留纯tensor字段，便于DataLoader/Trainer载入
* LLaMA Factory可通过`preprocess_func`复用同样逻辑

---

## 从预处理到训练数据流

* 预处理阶段：`map`逐样本构造prompt、标签、mask
* 训练阶段：DataLoader按batch读取，需要统一长度
* Collate函数承担批次内的“最后一公里”
  * padding使张量可堆叠
  * 生成`attention_mask`与`labels`中的忽略位
  * 可附加sample权重、position ids等训练信号、为定制loss构建对应label等
* 预处理 ≈ 单条样本准备；Collate ≈ 批次打包，两者衔接保证模型输入“稳定”

---

## Collate函数设计意图

* 解决变长序列在同一batch中堆叠的问题
* 统一补齐长度（padding），避免模型错判真实token与填充token
* 构造`attention_mask`与`labels`的忽略区域（`-100`）
* 可注入额外信息：loss mask、position ids、sample权重
* 与DataLoader协同，保证batch维度张量结构稳定

---

## Collate函数示例：Causal LM

```python
from torch.nn.utils.rnn import pad_sequence
import torch
def causal_lm_collate(batch, pad_id, label_pad_id=-100):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    attention = [torch.tensor(item["attention_mask"]) for item in batch]

    padded_input = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=label_pad_id)
    padded_attention = pad_sequence(attention, batch_first=True, padding_value=0)
    return {
        "input_ids": padded_input,
        "labels": padded_labels,
        "attention_mask": padded_attention,
    }
```

* `label_pad_id=-100`确保交叉熵在填充区域不计loss

---

## Collate函数挂载至DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    tokenized_train,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda batch: causal_lm_collate(
        batch,
        pad_id=tokenizer.pad_token_id,
    ),
)
```

* 通过lambda传入pad token
* 若使用Trainer，可传递给`data_collator`参数

---

## Collate函数与Trainer集成

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="checkpoints/demo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=lambda batch: causal_lm_collate(
        batch,
        pad_id=tokenizer.pad_token_id,
    ),
)
```

---

## 目标函数：定义模型要学的能力

* 预训练与微调常用**自回归交叉熵**(Cross Entropy)
  * 目标：预测下一个token
  * Label通常是右移一位的输入序列
* 变体
  * Label smoothing：缓解过拟合
  * 限定loss区域：只对Assistant段落计loss
  * 多任务：对不同字段加权求和
* 指令微调数据：输入+输出拼接，再使用mask忽略Prompt部分


---

## 为什么labels是input_ids右移一位？

* 自回归语言模型训练目标：在看到前`t`个token后预测第`t+1`个token
* 模型前向输出`logits`与下一个输入对齐，即位置`t`的logit对应`t+1`的输入token
* 为了计算预测`x_{t+1}`的loss，需要让`labels[t] = input_ids[t+1]`


---

## 为什么labels是input_ids右移一位？

```python
labels = tokenized["input_ids"].copy()
labels[:-1] = labels[1:]      # 右移
labels[-1] = -100             # 最后一个位置无法预测下一个token
labels[0] = -100              # 忽略起始位置的loss
tokenized["labels"] = labels
```
* 等价写法：直接复制`input_ids`并在loss计算前将prompt区域mask为`-100`
* 本质：让模型学会“给定上下文，预测下一个token”的条件概率

---

## 训练执行：梯度如何被计算与更新

* 高阶封装：`Trainer`/LLaMA Factory 屏蔽样板代码
* 手写循环：自定义前向、反向、梯度累积
* 混合精度：AMP/BF16 降低显存，提高吞吐
* 优化器与调度器：AdamW、Adafactor + Warmup/Decay
* 资源优化：gradient checkpointing、ZeRO、LoRA/QLoRA

---

## 评估与保存：闭环保障可复现

* 在线监控：训练loss、学习率、梯度范数、显存
* 验证集：PPL、任务指标、人工抽检输出
* 日志系统：TensorBoard、W&B(wandb.ai)
* Checkpoint：定期保存模型/优化器/配置
* 元信息：记录数据版本、commit hash、命令行参数


---

## 用LLaMA Factory微调

* 环境
  * Python 3.10+，`pip install llamafactory[torch]`
* 基座模型
  * HF格式(`--model_name_or_path`)或本地目录
* 数据
  * SFT: json、jsonl或HF dataset
  * 对话字段需包含`instruction/input/output`或`messages`
* 配置
  * `train`阶段yaml/cli参数：LoRA、batch、优化器等

---

## 数据示例（jsonl）

```json
{"instruction": "写一段自我介绍", "input": "", "output": "大家好，我是南京大学的学生..."}
{"instruction": "翻译句子", "input": "LLM可以做什么？", "output": "What can an LLM do?"}
```

* 或使用`messages`列表

```json
{
  "messages": [
    {"role": "user", "content": "介绍一下课程安排"},
    {"role": "assistant", "content": "本课程包含建模、训练、推理三个模块"}
  ]
}
```

* 文件路径在配置中通过`train_file`引用

---

## 典型训练配置（lora_qlora.yaml）

```yaml
stage: sft
model_name_or_path: /data/model/Meta-Llama-3-8B-Instruct
finetuning_type: lora
template: llama3
train_file: data/alpaca.jsonl
val_file: data/val.jsonl
output_dir: output/lora-sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 3
lora_target: q_proj,v_proj
bnb_4bit_quant_type: nf4
save_total_limit: 3
logging_steps: 10
```

* 可通过`template`指定prompt格式
* `bnb_4bit_*`参数用于QLoRA量化

---

## 启动训练命令

```bash
llamafactory train \
  --config configs/lora_qlora.yaml \
  --flash_attn \
  --gradient_checkpointing
```

* 常用参数
  * `--resume_from_checkpoint`断点续训
  * `--quantization_bit 4`快速设定QLoRA
  * `--deepspeed configs/ds_z3.json`分布式
* 训练后`output_dir`包含LoRA权重、配置与日志

---

## Trainer：快速落地训练

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
tokenizer.pad_token = tokenizer.eos_token
args = TrainingArguments(
    output_dir="checkpoints/l10",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=lambda batch: causal_lm_collator(batch, tokenizer.pad_token_id),
)
trainer.train()
```

* 优点：封装优化器、调度器、分布式、日志
* 通过`compute_metrics`/`callbacks`扩展自定义逻辑

---

## 手写训练循环：关键步骤

```python
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

optimizer = AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler(enabled=True)
model.train()
for step, batch in enumerate(dataloader, start=1):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    scaler.scale(loss).backward()
    if step % grad_accumulation == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

# 训练过程中的度量与监控

* 在线指标
  * 训练loss、学习率、梯度范数
* 验证指标
  * Perplexity、BLEU/ROUGE(指令)、自定义任务指标
* 工具
  * Trainer内置TensorBoard、Wandb配置
  * 自定义Callback记录采样输出
* 及时发现过拟合、梯度异常、Nan

---

# 训练完成后的动作

* 保存
  * `save_pretrained`存储模型+Tokenizer
  * `peft_state_dict`保存LoRA权重
* 转换
  * 转onnx、TensorRT、GGUF等部署格式
* 校验
  * 小样本推理验证
  * 版本化：记录数据版本、参数、代码哈希

