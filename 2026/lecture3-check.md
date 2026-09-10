# Lecture 3 检查记录

更新：2026-09-11。

- 2026 讲义基于 2025 版整理，保留 2025 原文件。
- 第 14–18 页围绕同一示例说明 .grad、.grad_fn、叶子张量、retain_grad 和 AccumulateGrad；区分梯度计算、累加与参数更新。
- inference_mode 介绍和对比表已注释，当前渲染共 32 页。
- 最后四页只讲 A0 的前向、上游梯度、输入梯度与权重梯度的 shape；不讲提交流程。
- 明确 A0 的 W[D,E] 与 nn.Linear 存储的 W[E,D] 布局不同。
- Notebook 同步 .grad/.grad_fn 示例；A0 打印各阶段 shape，并比较逐位置外积求和、展平矩阵计算与 Autograd；检查 mean 相对 sum 的梯度缩放。
- Notebook 中高阶求导、图可视化、inference_mode 和 profiler 作为扩展内容保留。
- 19 个代码单元在独立命名空间中按顺序执行通过（PyTorch 2.8.0，CPU）。有意触发的错误由示例捕获。未验证 GPU。
- Marp 构建成功；浏览器复查 32 页无尺寸溢出，重点公式页已目视检查。
- 正文图片位于 images/2026/l3，并同步至 docs/images/2026/l3；公共封面复用 images/course.webp。

参考：[A0 Demo](https://njudeepengine.github.io/LLM-Blog/2025/06/10/A0-onboarding/)。
