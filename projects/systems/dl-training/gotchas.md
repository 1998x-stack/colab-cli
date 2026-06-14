# dl-training gotchas

## 1. N(0, 1.0) 初始化是无声的模型杀手

**发现**: 用 `N(0, 1.0)` 初始化的 MLP，初始损失从预期的 2.30 跳到 2051（88994% 误差），梯度范数 662（正常应为 1-2）。

**为什么**: 标准正态初始化的权重过大，导致 logits 分布极宽，softmax 概率几乎为 0 或 1，cross-entropy 爆炸。

**检测**: 训练前总是检查初始损失。K=10 分类应接近 2.30。偏离超过 10% 说明初始化或损失函数有问题。

## 2. Colab 首次 session 对 DL 训练实验几乎无用

**发现**: CIFAR-10 下载 (~170MB) + CUDA JIT 编译 = 7-10 min 开销。加上 Colab ~10 min GPU 窗口，首次 session 在完成任何有意义的训练前就会死亡。

**修复**: 
1. 先用 CPU session 或短 exec 下载数据到 `/content/data/`
2. 等数据缓存后再用 GPU session 训练
3. 对多实验批量部署，考虑 Kaggle（30h/week GPU，无 WebSocket 断开问题）

## 3. view() 在 transformer 中特别危险

**发现**: Transformer 代码中频繁使用 `.permute()` 和 `.transpose()`，导致 tensor non-contiguous。后续 `.view()` 会崩溃。

**修复**: 全局使用 `.reshape()` 替代 `.view()`（Karpathy 的 6 大常见错误之一）。
