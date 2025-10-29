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

第9讲: 大语言模型解析 VI
基于HF LlaMA实现的讲解

<!-- https://marp.app/ -->

---

# LLM结构的学习路径

* LLM结构解析(开源LlaMA)
* 自定义数据集构造
* 自定义损失函数和模型训练/微调

---

## 如何让LLM“动”起来

<!-- ![bg right:40% 100%](images/l4/transformer.png) -->

<!-- ![bg right:30% 100%](images/l4/llama_arch_rope.png) -->

* 训练
  * 预训练 (pretraining)
    * 继续预训练(Continuous PreTraining, CPT)
  * 指令微调 (INstruction fine-tuning)
    * 监督微调 (Supervised Finetuning, SFT)
    * RLHF (带人类反馈(Human feedback)的强化学习(RL))
* 推理

---

## 数据集

* 预备 ```pip install datasets```
* 人类视角下的数据集 v.s. LLM视角下的数据集
  * 转换工具: tokenizer
<div style="display:contents;" data-marpit-fragment>

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

</div>

* 通过tokenizer将原始文本编码(encode)为token序列
<div style="display:contents;" data-marpit-fragment>


```python
encoded_input = tokenizer("Tell me a story about Nanjing University.")
```

</div>

---

## token序列 <---> 文本

* 字典结构
  * input_ids: token id
  * attention_mask
* 操作
  * encode
  * decode
  * padding
  * truncation


---

## 字典结构

基本元素：input_ids 和 attention_mask
```python
encoded_input = tokenizer("Tell me a story about Nanjing University.")
```
通过tokenizer编码后的token序列
```python
{
  'input_ids': [41551, 757, 264, 3446, 922, 33242, 99268, 3907, 13], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```


---

## 编码和解码

* 编码(encode)
  * tokenizer(input): 得到{'input_ids','attention_mask'}字典结构
  * tokenizer.tokenize(input): 得到tokens
  * tokenizer.encode(input): 得到tokens的ids
* 解码(decode)
  * tokenizer.decode(input): 得到文本
    * input为ids的List


---

## 如何批处理

* 多段文本组成的batch
<div style="display:contents;" data-marpit-fragment>

```python
batch_sentences = [
    "Tell me a story about Nanjing University.",
    "耿鬼能超进化么？",
    "大语言模型课程怎么考试？",
]
encoded_inputs = tokenizer(batch_sentences)
```

</div>

---

## 如何批处理

输出结果

```python
{'input_ids': 
  [[41551, 757, 264, 3446, 922, 33242, 99268, 3907, 13], 
  [20551, 123, 111188, 27327, 72404, 42399, 33208, 82696, 11571], 
  [27384, 120074, 123123, 123440, 104237, 118660, 11571]], 
'attention_mask': 
  [[1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1]]
}
```


---

## 批处理内容不一样长

```python
batch_sentences = [
    "Tell me a story about Nanjing University.",
    "耿鬼能超进化么？",
    "大语言模型课程怎么考试？",
]
```

添加padding

```python
encoded_input = tokenizer(batch_sentences, padding=True)
```

---

## Padding

```python
{'input_ids': 
  [[41551, 757, 264, 3446, 922, 33242, 99268, 3907, 13], 
  [20551, 123, 111188, 27327, 72404, 42399, 33208, 82696, 11571], 
  [27384, 120074, 123123, 123440, 104237, 118660, 11571, 128009, 128009]], 
'attention_mask': 
  [[1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1, 1, 1], 
  [1, 1, 1, 1, 1, 1, 1, 0, 0]]
}
```

---

## Padding


* 指定长度进行padding

<div style="display:contents;" data-marpit-fragment>
  
```python
encoded_input = tokenizer(batch_sentences, padding="max_length", max_length=20, truncation=True)
```

</div>

* 控制padding方向: padding_side
  * tokenizer.padding_side: left or right
<div style="display:contents;" data-marpit-fragment>

```python
tokenizer.padding_side = 'left'
encoded_input = tokenizer(batch_sentences, padding="max_length", max_length=20, truncation=True)
```

</div>

---

## 其他

* 句子太长，LLM无法处理
  * 指定长度进行truncation
    * 调用tokenizer时配置参数```truncation=True```
* 将token序列转化为tensor格式
  * 调用tokenizer时配置参数```return_tensors="pt"```


---

## 加载数据集

```python
from datasets import load_dataset

ds = load_dataset("yahma/alpaca-cleaned")
```

* 数据集有其自身格式，一般地，包含'train', 'validation', 'test'部分
  * 调用```load_dataset()```方法后获得数据集字典
    * 获取训练集```ds['train']```
    * 看看数据集构成...


---

## 加载数据集

* 需实现数据集的预处理方法，并交由Datasets的map方法调用
  * 预处理方法
<div style="display:contents;" data-marpit-fragment>

```python
def tokenize_function(dataset):
  ...
  return ...
  ```

</div>

* 调用预处理方法
<div style="display:contents;" data-marpit-fragment>

```python
ds = load_dataset("yahma/alpaca-cleaned", split='train[:100]')
ds = ds.map(tokenize_function, batched=True)
```
</div>


---

## 微调模型

加载模型
```python
model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)
```
设置训练参数
```python
from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="test_trainer")
```

---

## 阅读HF LlaMA实现

移步vscode

---

## LLM封装和参数装载(load)

* One basic PyTorch model
* LLM base model
* LoRA adapters

---

##  One basic PyTorch model

```python
import torch

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        return x
```

---

## 一个model实例的初始

当一个model被创建
```python
model = MyNetwork()
```
* 伴随而“建”的有什么？
* MyNetwork继承了```torch.nn.Module```
  * 回想```init```函数做了些什么？
    * 定义了每个基础模块
      * 每个模块亦继承了```torch.nn.Module```
      * 通常所说的参数存放在基础模块中


---

## nn.Linear: LLM的核心基本基础模块


nn.Linear的[实现](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)

```python
class Linear(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
```

---

## nn.Linear的init方法
```python
def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
```

---

## nn.Linear的reset_parameters方法

```python
def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
```


---

## nn.Linear的forward方法

```python
def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```

* 其中```F```是```torch.nn.functional```
  * ```from torch.nn import functional as F```



---

## nn.Linear中weight的定义和初始化
weight定义
```python
self.weight = Parameter(
    torch.empty((out_features, in_features), **factory_kwargs)
)
self.reset_parameters()
```
weight初始化，详见[torch.nn.init](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_)
```python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
```
```Parameter()```初始化会自动注册到```model.parameters()```中

---

## model如何存储和装载

* model保存，核心为保存参数
* PyTorch提供的保存方法
  * ```torch.save```
* model里都有什么, 可以用```print(model)```查看
  


<div style="display:contents;" data-marpit-fragment>

```python
MyNetwork(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
)
```

</div>

---

## model.state_dict()

* model参数存储在内部的字典结构```model.state_dict()```中
  *  ```print(model.state_dict().keys())```

<div style="display:contents;" data-marpit-fragment>


```python
odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias'])
```

</div>

<div style="display:contents;" data-marpit-fragment>

可通过```torch.save```存储模型至磁盘
```python
torch.save(model.sate_dict(), "model_weights.pt")
```

</div>

---

## model加载

* ```torch.save```存储的是一个模型的```state_dict```，那么加载的话
  * 创建model
  * 调用

<div style="display:contents;" data-marpit-fragment>


```python
model.load_state_dict(torch.load('model_weights.pt', weights_only=True))
```

</div>

* 存储/装载state_dict针对模型参数，也可直接存储/装载模型结构+模型参数
  * ```torch.save(model, 'model.pt')```
  * ```model = torch.load('model.pt', weights_only=False)```


---

## 基于PyTorch的参数装载过程

* torch.save
* torch.load
* torch.nn.Module.load_state_dict
* torch.nn.Module.state_dict


---


## HuggingFace对model的封装

* tensor的存储结构, [safetensors](https://github.com/huggingface/safetensors)
  * Storing tensors safely (as opposed to pickle) and that is still fast (zero-copy). 
* ```from_pretrained```和```save_pretrained```

<div style="display:contents;" data-marpit-fragment>


```python
import transformers
model_id = '/Users/jingweixu/Downloads/Meta-Llama-3.1-8B-Instruct'
llama = transformers.LlamaForCausalLM.from_pretrained(model_id)
llama.save_pretrained('/Users/jingweixu/Downloads/llama_test', from_pt=True)
```

</div>



---

## safetensors的其他存储/加载方式

```python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}
save_file(tensors, "model.safetensors")

tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)
```



---

## HuggingFace中的LoRA

* PEFT库提供LoRA实现
* LoRA是建立在一个已有的base model之上
* LoRA中的参数是base model的参数的一部分
  * 先加载base model
  * 再加载/创建对应的LoRA adapters

---

## HF加载LoRA的过程

```python
import transformers

model_id = '/Users/jingweixu/Downloads/Meta-Llama-3.1-8B-Instruct'
llama = transformers.LlamaForCausalLM.from_pretrained(model_id)
```

<div style="display:contents;" data-marpit-fragment>


```python
from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

peft_model = get_peft_model(llama, peft_config)

```

</div>

---

## 原始的LlamaForCausalLM结构

```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```


---

## PEFT的PeftModelForCausalLM结构

```python
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 4096)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaSdpaAttention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=4096, bias=False)
                )
```


---

## 读懂PEFT加载LoRA的过程

* 入口: ```get_peft_model```方法
  * ```mapping_func.py```中的方法
  
<div style="display:contents;" data-marpit-fragment>


```python
self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
``` 
```class BaseTuner(nn.Module, ABC):```中的```inject_adapter```方法和```_create_and_replace```方法（LoRA.model.py中实现）

</div>

* 入口: ```peft_model.py```中的```PeftModel.from_pretrained```方法