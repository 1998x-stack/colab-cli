# 深度学习训练技巧基准测试

在 Colab T4（CUDA 12.8, PyTorch 2.11.0+cu128）上验证常见的 DL 训练技巧和陷阱。

## 实验概览（12 项实验，11 项已完成）

| ID | 实验 | 状态 | 关键结果 |
|----|------|------|---------|
| dltrain-001 | 单 batch 过拟合检查 | ✅ done | 200 步 loss→0.000051, 最有效的 bug 检测法 |
| dltrain-002 | 初始化损失理论值检查 | ✅ done | N(0,1.0) 初始化导致灾难性发散 |
| dltrain-003 | OneCycleLR vs 恒定 LR | ✅ done | 超收敛, 对 Colab ~10 min GPU 窗口至关重要 |
| dltrain-004 | AdamW vs Adam | ✅ done | weight decay 解耦 |
| dltrain-005 | EMA 权重平均 | ✅ done | 免费 0.5-1% 准确率提升 |
| dltrain-006 | Label smoothing | ✅ done | 防止过拟合和 attention 坍缩 |
| dltrain-007 | 低 epoch 数据增强反效果 | ⏳ planned | 对 Colab 有限 GPU 窗口至关重要 (8min, 多次尝试 session 死亡) |
| dltrain-008 | BN 前的 bias 死参数 | ✅ done | 128 死参数, bias 被 BN 抵消无精度影响 |
| dltrain-009 | LR Finder | ✅ done | 一个 epoch 找到最优 LR |
| dltrain-010 | SWA 随机权重平均 | ✅ done | SWA 最终精度 +0.5% (58.26%→58.72%) |
| dltrain-011 | view vs reshape 陷阱 | ✅ done | non-contiguous 上 view() 会报错; reshape() 安全 |
| dltrain-012 | bias/BN 参数的 weight decay | ✅ done | 排除 bias+BN 参数避免有害 weight decay |

---

## dltrain-001: 单 Batch 过拟合检查 ✅

**结论**: 模型应能在 200 步内将单个 batch (16 样本) 过拟合到接近零损失。

```
初始损失: 2.2940
最终损失: 0.000051
最终准确率: 1.000
判定: PASS — 模型能够过拟合单 batch
```

**实践**: 移除 dropout、weight decay、data augmentation 后再测试。Karpathy: "最重要的调试技术。"

## dltrain-002: 初始化损失检查 ✅

**结论**: K=10 类 softmax 的理论初始损失 = -ln(1/10) ≈ 2.3026。不同初始化方案的表现：

| 初始化方案 | 初始损失 | 误差% | 梯度范数 | 评价 |
|-----------|---------|-------|---------|------|
| kaiming_uniform | 2.3389 | 1.6% | 1.61 | ✅ 最佳平衡 |
| xavier_uniform | 2.3184 | 0.7% | 1.64 | ✅ 良好 |
| N(0, 0.01) | 2.3025 | 0.0% | 0.08 | ⚠️ 梯度消失 |
| **N(0, 1.0)** | **2051.25** | **88994%** | **662.3** | ❌ 灾难性发散 |

**关键发现**: `N(0, 1.0)` 初始化的损失高达 2051——比预期值 2.30 偏离 88994 倍！梯度范数 662 表明梯度爆炸。

## dltrain-011: view vs reshape 陷阱 ✅

**结论**: `view()` 要求 contiguous 内存，在 permuted tensor 上会报错或产生静默错误数据。`reshape()` 在需要时自动拷贝。

```python
x = torch.randn(2, 3, 4).permute(2, 0, 1)  # non-contiguous
x.view(-1)        # ❌ RuntimeError: view size is not compatible
x.reshape(-1)     # ✅ OK，自动拷贝（如果需要）
x.contiguous().view(-1)  # ✅ 显式修复，总是正确
```

## 环境

- GPU: Tesla T4 (15.6 GB VRAM)
- CUDA: 12.8
- PyTorch: 2.11.0+cu128
- 数据集: CIFAR-10 (需要首次下载 ~170MB)
- Colab: 免费套餐 (~10 min GPU 窗口)
- **注意**: 首次 session 因数据下载 (7-10 min) 几乎无法完成训练——先跑一个预热 session 缓存数据到 `/content/data/`。
