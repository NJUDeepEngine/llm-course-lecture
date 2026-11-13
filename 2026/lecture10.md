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

[下载配套 Notebook](notebook/lecture10.ipynb)

<!-- https://marp.app/ -->

---

## LLM结构的学习路径

上一讲完成了文本编码、模型加载和 LoRA 权重的保存。本讲把这些步骤串成一次训练。

1. 数据准备与批处理：从样本到 `input_ids`、`attention_mask`、`labels`
2. 训练目标：用当前位置的 logits 预测下一个 token
3. 训练执行：梯度更新、评估与保存

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

* `filter` 返回新的 `Dataset`，可用 `num_proc` 指定多进程
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
bucketed = bucketed.class_encode_column("len_bucket")
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

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

* 统一tokenizer，避免训练/推理词表不一致
* 某些模型需手动指定`pad_token`

---

## Step5: Tokenizer对齐（构造模板）

```python
def build_prompt(example):
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
    text = build_prompt(example)
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

* `map` 支持 `batched=True` 或通过 `num_proc` 使用多进程
* `remove_columns` 移除原始文本字段，保留预处理生成的数值字段

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
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(model_id)
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

* 自回归语言模型用当前位置的 logits 预测下一个 token
* Hugging Face 的 `LlamaForCausalLM` 接收和 `input_ids` 同位置的 `labels`，计算 loss 时在内部对齐
* 指令微调可把 prompt 的标签设为 `-100`，只对回答和结束符计 loss
* Padding 的标签也设为 `-100`，其 `attention_mask` 为 0

---

## 手算一次 next-token 对齐

| 位置 | 0 | 1 | 2 | 3 | 4 |
|:--|:--|:--|:--|:--|:--|
| `input_ids` | 用户 | 开始回答 | 南京 | 大学 | EOS |
| `labels` | -100 | -100 | 南京 | 大学 | EOS |
| 该位置 logits 的目标 | 忽略 | 南京 | 大学 | EOS | 无 |

回答的第一个 token「南京」由它前一个位置的 logits 预测。

---

## 手写 loss，并和模型内置 loss 对照

```python
import torch.nn.functional as F

outputs = model(**batch)
logits = outputs.logits[:, :-1, :].float()
targets = batch["labels"][:, 1:]

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=-100,
)
torch.testing.assert_close(loss, outputs.loss)
```

传给 `model(..., labels=...)` 的标签保留原位置。Notebook 会逐步验证这两种计算。

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
  * 安装方式见 [LLaMA Factory 官方仓库](https://github.com/hiyouga/LlamaFactory)
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

* 自定义数据先在 `data/dataset_info.json` 注册文件与字段映射，再用 `dataset` 指定名称

---

## 典型训练配置（lora_sft.yaml）

```yaml
stage: sft
do_train: true
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
finetuning_type: lora
template: llama3
dataset: alpaca_en_demo
output_dir: output/lora-sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 3
lora_target: q_proj,v_proj
save_total_limit: 3
logging_steps: 10
```

* 可通过`template`指定prompt格式
* 使用 bitsandbytes 做 QLoRA 时，在支持的 GPU 环境加 `quantization_bit: 4`

---

## 启动训练命令

```bash
llamafactory-cli train configs/lora_sft.yaml
```

* 常用参数
  * `resume_from_checkpoint` 断点续训
  * `quantization_bit: 4` 配置 QLoRA
  * `deepspeed` 指定分布式配置
* 训练后`output_dir`包含LoRA权重、配置与日志

---

## Trainer：快速落地训练

沿用前面已创建的 `trainer`、训练集与 `data_collator`。

```python
trainer.train()
print(trainer.evaluate())
trainer.save_model("checkpoints/l10/final")
tokenizer.save_pretrained("checkpoints/l10/final")
```

* `Trainer` 管理优化器、学习率调度、梯度累积与日志
* 可通过 callbacks 扩展日志与检查点行为

---

## 手写训练循环：关键步骤

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for batch in train_loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

这里每批更新一次参数，方便观察前向、反向和参数更新。上面的 `Trainer` 示例每 8 个 micro-batch 累积更新一次。

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
  * 模型和 tokenizer 分别调用 `save_pretrained`
  * 对 PEFT 模型调用 `save_pretrained` 保存 LoRA adapter
* 转换
  * 转onnx、TensorRT、GGUF等部署格式
* 校验
  * 小样本推理验证
  * 版本化：记录数据版本、参数、代码哈希
