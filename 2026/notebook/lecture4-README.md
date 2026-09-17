# Lecture 4 课堂实验

按顺序运行 lecture4.ipynb。依赖 torch、transformers、tokenizers、ipykernel。
使用兼容的一组版本，例如 torch 2.8.0、transformers 4.56.1。
实验 1 使用 Qwen/Qwen3-0.6B tokenizer。实验 2.1 离线加载完整模型权重，在 CPU 上展示真实输入 embedding，需要数 GB 可用内存。
模型默认位于 ~/.cache/llm-course/qwen3-tokenizer；可通过 LECTURE4_MODEL_PATH 指定其他完整模型目录。缺少权重时会明确报错，不自动下载。
可设置 LECTURE4_TOKENIZER_PATH 指向本地 tokenizer；也会识别 ~/.cache/llm-course/qwen3-tokenizer 缓存。

| 讲义章节 | Notebook |
|---|---|
| 中英文分词、批量输入 | 实验 1 |
| 查表、A2 Task 2 接口 | 实验 2 |
| 从小词表到真实模型 | 实验 2.1：加载、形状、查表验证、重复 ID 与 padding |
| one-hot 与查表等价 | 实验 3 |
| 词表分片、A2 Task 3 | 实验 4 |
| 二维 RoPE | 实验 5 |
| Tensor 操作补充 | 附录 |

课堂类演示主要接口，不替代作业官方模板。分片种子为 base_seed + rank，验证使用局部参数拼接的同一完整表。
RoPE 实验验证二维旋转与相对位置；未实现 A2 optional Task 4 的 NTK 缩放、高维配对和动态缓存。
