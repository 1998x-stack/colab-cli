# Neural Network Quantization: INT4 & INT8 — 技术指南 & Colab 实战分析

2026-06-13 | 综合研究 (理论 + 开源生态 + Colab 可行性)

---

## 1. 量化基础

### 1.1 什么是量化

将模型权重/激活值从浮点 (FP32/FP16) 映射到低位整数 (INT8/INT4) 的过程。三大收益：

| 收益 | 机制 | 幅度 |
|------|------|------|
| **内存缩减** | 每参数 4字节→1字节 (INT8) 或 0.5字节 (INT4) | 4-8x |
| **计算加速** | INT8 Tensor Core 吞吐 = FP16 的 2x | 1.3-2x (实际) |
| **带宽缓解** | 从 DRAM 搬运更少字节 → 推理瓶颈通常是带宽而非计算 | 3-4x |

### 1.2 核心数学

**非对称量化 (Affine):**

```
scale = (max_val - min_val) / (q_max - q_min)
zero_point = round(-min_val / scale) + q_min      # 对齐 FP 零点和整数零点

量化:  q = clamp(round(x / scale + zero_point), q_min, q_max)
反量化: x̂ = (q - zero_point) * scale
```

**对称量化 (Scale-only, 权重默认):**

```
scale = max(|min|, |max|) / max(|q_min|, |q_max|)

量化:  q = clamp(round(x / scale), q_min, q_max)
反量化: x̂ = q * scale
```

**两种误差来源:**
- **舍入误差:** 真值与最近量化能表示值之差, 上限 `±0.5 * scale`
- **截断误差:** 超出 [min, max] 的值被 clamp, 可能远大于舍入误差

### 1.3 粒度对比

| 粒度 | 共享 scale/zp 范围 | 开销 | 精度 |
|------|-------------------|------|------|
| Per-tensor | 整个张量 | 2 个值 | 最低 |
| Per-channel | 每个输出通道 | out_features 个 | INT8 标准 |
| Per-group (group=128) | 每 128 个连续权重 | params/128 个 | INT4 必须用 |

INT4 强制要求 per-group — per-tensor/per-channel INT4 精度不可接受。

---

## 2. INT8 细节

### 2.1 表示范围

| 模式 | 范围 | 值个数 | 用途 |
|------|------|--------|------|
| Signed INT8 | [-128, 127] | 256 | 权重 (零均值分布) |
| Unsigned UINT8 | [0, 255] | 256 | ReLU 激活值 (非负) |

### 2.2 校准方法 (静态量化)

| 方法 | 复杂度 | 抗离群 | 质量 |
|------|--------|--------|------|
| MinMax | O(n) | 否 | 低 (离群值主导) |
| Percentile (99.9%) | O(n log n) | 是 | 好 |
| MSE grid search | O(n × candidates) | 是 | 很好 |
| KL Divergence | O(n × candidates) | 是 | 最优 |

**实践建议:** 99.9% percentile 是 LLM 的务实起点; KL divergence 是精度敏感场景的最佳选择。

### 2.3 INT8 矩阵乘法流程

```
Step 1: 量化输入
  X_int8 = round(X_fp32 / scale_x)
  W_int8 = round(W_fp32 / scale_w)

Step 2: INT32 累加的整数矩阵乘 (Tensor Core)
  Y_int32 = X_int8 @ W_int8^T    # INT8×INT8 → 累加到 INT32

Step 3: 反量化输出
  Y_fp32 = Y_int32 * (scale_x * scale_w)
```

INT32 累加是必须的 — K=1024 的维度下最大值可达 `127 × 127 × 1024 ≈ 16.5M`，INT16 会溢出。

---

## 3. INT4 细节

### 3.1 为什么 INT4 比 INT8 难得多

INT4 只有 **16 个可表示值**，INT8 有 256 个 — 粒度相差 16 倍。

核心问题:
1. **离群值劫持 scale:** 一个 outlier=8.0 当典型范围 [-1,1] → scale 被拉伸覆盖 [-8,8] → 15 个正常值只映射到 ~2 个量化级别
2. **必须 per-group:** group_size=128 是标准 (GPTQ/AWQ/bitsandbytes 默认)
3. **NF4 非均匀编码:** 最优 INT4 格式使用查表反量化而非简单算术

### 3.2 Group Size 权衡 (7B 模型, INT4)

| Group | 有效 bpw | 总内存 | 精度 vs FP16 |
|-------|----------|--------|-------------|
| 32 | 4.13 | 3.61 GB | <1% loss |
| 64 | 4.06 | 3.56 GB | 0.5-1.5% loss |
| 128 | 4.03 | 3.53 GB | 1-2% loss (默认推荐) |
| 256 | 4.02 | 3.51 GB | 明显退化 |

Group=128 是精度/开销的最佳折中。

---

## 4. 核心技术

### 4.1 GPTQ (Frantar et al., 2023)

**思路:** 逐层二阶误差最小化。逐列量化权重，每量化一列后用 Hessian 逆矩阵补偿剩余列的误差。

```
对每个 Linear 层 (权重 W, 校准激活 X):
  1. H = 2 * X^T X + λI           # 每层独立 Hessian
  2. 对每列 j (块大小 B=128):
     a. qⱼ = quantize(wⱼ)
     b. err = wⱼ - qⱼ
     c. 用 H⁻¹ 将 err 补偿到剩余未量化列
```

**特点:** INT4 PTQ 精度最高; 校准时间 30-120min (7B); 需 GPU 内存存 Hessian。

### 4.2 AWQ (Lin et al., MLSys 2024 Best Paper)

**核心洞察:** 仅 ~1% 权重 (对应大激活幅值的通道) 是"显著"的。只需保护这些通道。

**数学上无损的缩放变换:**

```
y = xW = (x · diag(s)⁻¹) · (diag(s) · W)

其中 sⱼ = |Xⱼ|^α   (α ≈ 0.5, 网格搜索)
```

- 显著通道的权重变大 → 跨越更多量化 bin → 更高有效精度
- 对应的激活值变小 → 整体输出不变
- 缩放因子融合到前一层的 LayerNorm 中 → 零运行时开销

**对比 GPTQ:** 校准快 3-5x (10-30min), 推理吞吐高 20-50%, 精度略低 (差距 <0.5%)。

### 4.3 bitsandbytes NF4 (Dettmers et al., QLoRA 2023)

**思想:** 标准均匀 4-bit 量化对正态分布权重次优。NF4 对 N(0,1) 是信息论最优的。

**构造方法:**
- 将标准正态分布 PDF 按等面积分为 16 个 bin
- 每 bin 的 NF4 值 = bin 内 N(0,1) 的期望值, 归一化到 [-1, 1]
- 零点附近的值得到细粒度量化, 尾部值得到粗粒度 → 非均匀间距

**Double Quantization:** 对 per-block scale 因子自身再做 INT8 量化 → scale 开销从 3% 降到 <0.5%。

**QLoRA 配置 (标准):**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

### 4.4 SmoothQuant (Xiao et al., 2023)

**解决的问题:** LLM 激活值在特定通道有极端离群值 → W8A8 激活量化极难。

**解法:** 与 AWQ 同类的数学无损缩放:

```
sⱼ = max(|Xⱼ|)^α / max(|Wⱼ|)^(1-α)

y = (x · diag(s)⁻¹) · (diag(s) · W) = x̂ · Ŵ
```

α=0.5 是通用默认; α=0.75-0.9 适用于激活离群值极端的模型 (LLaMA, GLM-130B)。

缩放因子融合进前一层 LayerNorm → 零额外开销。对 175B 模型维持接近无损的 W8A8。

### 4.5 GGUF / K-quant 体系

llama.cpp 生态定义了最全面的量化格式家族。关键创新是**两级 scale 层级**:

| 格式 | Block | bpw | 说明 |
|------|-------|-----|------|
| Q8_0 | 32 | 8.5 | 接近无损, 1×f16 scale |
| Q4_0 | 32 | 4.5 | 旧版 4-bit, 避免使用 |
| Q4_K_M | 256→32 | ~4.5 | **标准推荐**, 两级 scale, 混合精度 |
| Q5_K_M | 256→32 | ~5.5 | 质量敏感场景 |
| Q2_K | 256 | 2.56 | 极端压缩 |

**K-quant 层级:** Super-block (256权重, 共享 d/m) → Sub-block (32权重, 量化 6-bit scale) → 量化权重。

**M 变体:** 对关键 tensor (attention.wv, feed_forward.w2) 用 Q6_K, 其余 Q4_K → 更好的质量/大小比。

---

## 5. VRAM 分析

### 5.1 模型权重 (纯推理)

| 模型 | FP32 | FP16 | INT8 | INT4/NF4 |
|------|------|------|------|----------|
| 350M | 1.4 GB | 0.7 GB | 0.35 GB | 0.18 GB |
| 1.7B (SmolLM2) | 6.8 GB | 3.4 GB | 1.7 GB | 0.85 GB |
| 3B | 12 GB | 6 GB | 3 GB | 1.5 GB |
| 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B | 280 GB | 140 GB | 70 GB | 35 GB |

### 5.2 KV Cache (常被忽略)

```
per_token = 2(K+V) × 2 bytes(FP16) × n_layers × d_kv × n_heads
```

| 模型 | 1024 tokens | 2048 tokens |
|------|-------------|-------------|
| SmolLM2-1.7B | 2.6 GB | 5.2 GB |
| Qwen2.5-3B | 6 GB | 12 GB |
| Llama-3-8B | 12 GB | 24 GB |

**T4 (15.1 GB) 推理上限:**
- FP16: 7B 模型仅剩 1.1 GB — 太紧
- INT8: 7B + 8 GB KV cache — 舒适
- INT4: 13B + 8.6 GB KV cache — 舒适
- INT4 + 2048 context: SmolLM2-1.7B (6.05 GB total) — 可行; 3B (13.5 GB) — 勉强

### 5.3 QLoRA 训练 VRAM

7B NF4 模型 + LoRA rank=8 + batch=1:

| 组件 | 内存 |
|------|------|
| NF4 权重 | 3.5 GB |
| LoRA 适配器 (FP16) | ~16 MB |
| Optimizer states (Adam) | ~400 MB |
| Activations (seq=512) | ~1-3 GB |
| **Total** | **~5-8 GB** |

**结论:** QLoRA 7B 拟合 T4 VRAM, 但训练时间 >>10 min → Colab free tier 不可行 (见 §7)。

---

## 6. 开源生态地图

### 6.1 核心库

| 库 | 定位 | Stars | License | T4 兼容 |
|----|------|-------|---------|---------|
| **bitsandbytes** | NF4/INT8 推理 + QLoRA | ~13k | MIT | 完全 |
| **AutoGPTQ** | 基于 Hessian 的 PTQ | ~4.7k | MIT | 完全 |
| **AutoAWQ** | 激活感知 PTQ | ~2.5k | Apache 2.0 | 完全 |
| **llama.cpp** | CPU/GPU GGUF 推理 | ~100k | MIT | 完全 |
| **HQQ** | 无校准即时量化 | ~940 | Apache 2.0 | 完全 |
| **AQLM** | 加性码本极端压缩 (2-bit) | ~1.3k | Apache 2.0 | 可行 |
| **QuIP#** | Hadamard + 格码本 2-bit | ~590 | GPL v3 | 可行 |
| **torchao** | PyTorch 官方量化 | 内置 | BSD | 完全 |
| **TensorRT-LLM** | NVIDIA 推理优化 | — | Apache 2.0 | 部分 (4-bit 需 Ampere+) |
| **ONNX Runtime** | 跨平台部署量化 | — | MIT | 完全 |

### 6.2 终端用户项目

| 项目 | 量化使用 | T4 |
|------|---------|-----|
| **Ollama** (~174k ★) | GGUF Q4_K_M 自动拉取 | 可行 (≤13B Q4) |
| **vLLM** (~82k ★) | AWQ/GPTQ/NF4 原生支持 | 可行 (≤3B, v0.10.2) |
| **ExLlamaV2** (~4.3k ★) | 最快 GPTQ 推理内核 | T4 上最快 |
| **TGI** (~10.9k ★) | 维护模式, 迁移到 vLLM | 不推荐 |

### 6.3 前沿研究 (2024-2026)

| 论文 | 年份 | 核心思想 | 位宽 |
|------|------|---------|------|
| SpinQuant | ICLR 2025 | 学习旋转消除离群值 | 4-bit |
| QuaRot | NeurIPS 2024 | 旋转消除离群值 | 4-bit |
| QTIP | NeurIPS 2024 | 格码 + 非相干处理 | 2-4 bit |
| SvdQuant | 2025 | SVD 低秩分量吸收离群 | 4-bit |
| ParetoQ | 2025 | 极低位缩放定律 | 2-3 bit |
| BitNet b1.58 | ACL 2025 | 三元权重 {-1, 0, +1} | 1.58 bit |
| FlatQuant | 2025 | 平坦性感知量化 | 4-bit |

---

## 7. Colab 实现可行性 (Free Tier)

### 7.1 硬约束

| 约束 | 值 | 影响 |
|------|-----|------|
| GPU | T4, 15.1 GB VRAM, CUDA 12.8 | 模型上限 7B INT4 |
| GPU 窗口 | **~10 min** (实测) | 核心约束 — 不是 VRAM |
| 首 session 预热 | 7-10 min (下载 + pip + CUDA JIT) | 首 session 几乎不会有产出 |
| RAM | ~12 GB | pip 编译时可能 OOM |
| Disk | ~78 GB | 足够缓存模型 |

### 7.2 三级可行性

**TIER 1 — 完全可行 (≤10 min GPU, 含预热 ≤2 sessions)**

| 项目 | 模型 | VRAM | GPU 时间 |
|------|------|------|----------|
| INT8 从零实现 (对称/非对称量化 Linear) | 自定义 MLP | <1 GB | 2-3 min |
| CNN 量化对比 (FP32 vs FP16 vs INT8 vs INT4) | ResNet-18 / MobileNetV2 | <1 GB | 2-3 min |
| PTQ on ImageNette (torch.ao) | ResNet-18 | ~200 MB | 1-2 min |
| SmolLM2 NF4 推理 + 困惑度评测 | 1.7B NF4 | ~3 GB | 3-5 min |
| GGUF 推理基准测试 (llama.cpp) | Qwen2.5-1.5B Q4_K_M | ~1 GB + CPU | CPU 推理 |
| 量化误差逐层分析 | SmolLM2-360M | <1 GB | 3-5 min |

**TIER 2 — 有条件可行 (需多 session 或接近 10 min 上限)**

| 项目 | 模型 | VRAM | GPU 时间 | 注意事项 |
|------|------|------|----------|---------|
| GPTQ 量化 (group=128) | SmolLM2-1.7B / Qwen2.5-3B | ~10 GB | 5-10 min | 校准数据需预下载 |
| AWQ 量化 | SmolLM2-1.7B | ~5 GB | 3-5 min | CUDA kernel 编译 ~3 min (首 session) |
| QLoRA 微调 (300 样本, 2 epoch) | SmolLM2-1.7B NF4 | ~3 GB | 5-8 min | 勉强可在单 session 完成 |
| NF4 vs GPTQ vs AWQ 对比 | SmolLM2-360M | ~2 GB | 8-12 min | 拆分为多个 session |
| vLLM 0.10.2 量化服务 | 1.7B GPTQ | ~6 GB | 5 min | VLLM_USE_V1=0 + transformers 5.x patch |

**TIER 3 — 不可行 (超过 10 min 或 VRAM)**

| 项目 | 失败原因 | 替代方案 |
|------|---------|----------|
| QLoRA 7B 全量微调 | 训练时间 30-60+ min | Kaggle P100 |
| GPTQ 量化 7B | 15-30 min | 拆分为多 session 或 Kaggle |
| BitNet 从零训练 | 数小时 | Kaggle |
| QAT 1B+ 模型 | 完整训练循环, 数小时 | Kaggle |
| FP16 推理 7B + 长上下文 | 14 GB 权重 + 12+ GB KV cache = OOM | INT4 量化 |

### 7.3 Colab 特定注意事项

**bitsandbytes 安装** — Colab CUDA 12.8, 需 `bitsandbytes>=0.45`。避免源码编译 (耗 3-8 min), 用 `--prefer-binary`。

**CUDA Kernel 编译** — AutoGPTQ/AutoAWQ/vLLM 首次 import 时编译 CUDA kernels (2-5 min)。策略: 首 session 做 warmup import → session 死掉 → 次 session 从缓存加载。

**vLLM 0.10.2 on T4:**
```python
import os
os.environ["VLLM_USE_V1"] = "0"
# + transformers 5.x monkey-patch (见 CLAUDE.md)
```

**GPTQ on T4** — `use_triton=False` 必须设置 (Triton 不支持 Turing)。

**torch.ao 兼容性** — Colab 默认 PyTorch 2.4-2.5, `quantize_dynamic()` 稳定; 新版 `quantize_pt2e()` 可能需要 2.6+。

### 7.4 推荐 Session 架构

```
Session 1 (预热 — 预计在 ~10 min 时死掉):
  pip install + HF login + 模型下载 + CUDA kernel 编译

Session 2 (正式运行):
  量化 + 评测 + 绘图 (全部 ≤8 min GPU)

Session 3+ (可选, 如需对比多种方法):
  下载 artifacts, 切方法, 重复
```

---

## 8. 推荐学习路径

### 8.1 入门 (1-2 天)

1. **从零实现 INT8:** [kaushikacharya/Quantization_in_Depth](https://github.com/kaushikacharya/Quantization_in_Depth) — 手写对称/非对称量化, per-tensor/per-channel/per-group
2. **可视化学习:** [OscarSavolainen/Quantization-Tutorials](https://github.com/OscarSavolainen/Quantization-Tutorials) + YouTube — ResNet 量化的完整流程
3. **最小化实践:** [crazy-JiangDongHua/pytorch-quantization-demo](https://github.com/crazy-JiangDongHua/pytorch-quantization-demo) — 最小独立 PTQ/QAT 实现

### 8.2 LLM 量化 (2-3 天)

4. **bitsandbytes 实战:** QLoRA 论文 + HF `BitsAndBytesConfig` — NF4 加载 + 推理
5. **GPTQ vs AWQ 对比:** 对 SmolLM2-1.7B 分别用 AutoGPTQ 和 AutoAWQ 量化, 对比 perplexity/速度/VRAM
6. **GGUF 转换:** HF → GGUF via llama.cpp `convert_hf_to_gguf.py` → Q4_K_M 量化 → `llama-perplexity` 评测

### 8.3 前沿 (3-5 天)

7. **阅读论文:** [Awesome-Model-Quantization](https://github.com/Kai-Liu001/Awesome-Model-Quantization) — SpinQuant, QuaRot, QTIP
8. **极端压缩:** AQLM 或 QuIP# 2-bit 量化 → 对比 4-bit 精度差距
9. **三元化前沿:** BitNet b1.58 2B — 体验无浮点乘法的推理

---

## 9. 在 Colab 上可以实现的具体项目

### 项目 1: Quantization Playground (TIER 1, 2-3h)

交互式对比 FP32→FP16→INT8→INT4 对 CNN 精度/大小/速度的影响。

- 模型: ResNet-18 on ImageNette
- 方法: torch.ao dynamic quantization + 自定义 INT4 Linear
- 输出: 多面板图 (精度曲线 + 模型大小 + 推理延迟)
- Session: 1 次 (5-7 min GPU)

### 项目 2: INT8 From Scratch (TIER 1, 2h)

纯 PyTorch 实现对称/非对称 INT8 量化, 不含任何库依赖。

- 实现: `quantize()`, `dequantize()`, `QuantizedLinear` 类
- 验证: MNIST MLP — 对比 FP32 baseline 精度
- 延伸: per-channel vs per-tensor, 校准方法对比 (MinMax vs MSE vs Percentile)
- 输出: 教程级 notebook + 精度对比表

### 项目 3: LLM Quant Explorer (TIER 1-2, 4h)

对 SmolLM2-1.7B 应用多种量化方法, 全面对比。

- 格式: FP16 baseline → NF4 (bitsandbytes) → GPTQ 4-bit → AWQ 4-bit → GGUF Q4_K_M
- 评测: WikiText-2 perplexity + 推理吞吐 (tok/s) + VRAM
- Session: 4 次 (每方法一次), 各 5-8 min
- 输出: 量化方法对比表 + 推荐决策树

### 项目 4: QLoRA 微调 (TIER 2, 3-4h)

SmolLM2-1.7B NF4 + LoRA rank=8 微调 300 条指令数据。

- 数据集: Alpaca-52k 子集 (预下载)
- 训练: 2 epochs, batch=1, gradient accumulation
- 时间: ~5 min GPU
- 输出: 微调后 LoRA 权重 + 训练前后对比

---

## 参考文献

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (Frantar et al., 2023)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (Lin et al., MLSys 2024 Best Paper)
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438) (Xiao et al., 2023)
- [QuIP#: 2-bit Quantization with Lattice Codebooks](https://arxiv.org/abs/2402.04396) (Tseng et al., 2024)
- [HQQ: Half-Quadratic Quantization of Large Machine Learning Models](https://mobiusml.github.io/hqq_blog/) (2024)
- [Keras Quantization Guide](https://keras.io/guides/quantization_overview/)
- [Awesome Model Quantization](https://github.com/Kai-Liu001/Awesome-Model-Quantization)
