# CUDA 暗角基准测试

在 Colab T4（CUDA 12.8, PyTorch 2.11.0+cu128）上系统性地验证 PyTorch/CUDA 性能陷阱——哪些仍然存在，哪些已被修复。

## 实验概览（16 项实验，6 个子类别）

| 子类别 | 实验数 | 完成 | 核心发现 |
|--------|--------|------|---------|
| [Kernel Launch](#kernel-launch) | 2 | 2/2 | GPU vs CPU 交叉点: matmul ~128×128, 逐元素 ~100K |
| [Hidden Sync](#hidden-synchronization) | 3 | 3/3 | `.item()` 隐藏 sync 代价 5-29%; `torch.where` 快 2-3.8× |
| [Tensor Layout](#tensor-layout) | 3 | 3/3 | **CE(permute) 慢 115×** — 最严重陷阱；CUDA 12.8 修复了隐式 copy |
| [Memory Allocator](#memory-allocator) | 1 | 1/1 | CUDA 12.8 allocator 处理碎片化良好 (0% waste) |
| [Mixed Precision](#mixed-precision) | 2 | 2/2 | FP16 eps NaN **未复现** (GradScaler 有效)；Tensor Core 利用率 37% |
| [torch.compile](#torch-compile) | 1 | 1/1 | compile **消除**了 eager 模式的 layout 敏感性(4.0×→1.0×) |

---

## Kernel Launch

### launch-001: GPU vs CPU 交叉点

**结论**: T4 上的 matmul 交叉点为 ~128×128；逐元素操作交叉点为 ~100K 元素。

| 尺寸 | CPU | GPU | 胜者 |
|------|-----|-----|------|
| 64×64 matmul | 0.02 ms (9 GFLOPS) | 0.05 ms (11 GFLOPS) | CPU |
| 128×128 matmul | 0.07 ms (63 GFLOPS) | 0.04 ms (96 GFLOPS) | **GPU 1.5×** ← 交叉点 |
| 2048×2048 matmul | 165 ms (104 GFLOPS) | 4.75 ms (3619 GFLOPS) | GPU 34.8× |
| 100K elem relu | 0.021 ms | 0.022 ms | ~持平 ← 交叉点 |
| 5M elem relu | 2.735 ms | 0.190 ms | GPU 14.4× |

### launch-002: CUDA 首次调用税

**结论**: `torch.cuda.init()` 几乎无效。首次 CUDA 操作仍需 389ms（vs 后续 300µs）。

```
首次 .to('cuda'):  389 ms
后续 .to('cuda'):  0.3 ms  ← 1296× 差异
首次 matmul:      102 ms
后续 matmul:      0.2 ms  ← 491× 差异
```

---

## Hidden Synchronization

### sync-001: `.item()` 隐藏同步代价

**结论**: 每个 `.item()` 隐式调用 `cudaStreamSynchronize()`。用 `.tolist()` 批量提取可减少 5-29% 开销。

| 每步指标数 | per-item (ms/step) | tolist (ms/step) | 开销 |
|-----------|-------------------|------------------|------|
| 1 | 1.16 | 0.90 | **+29%** |
| 4 | 0.99 | 0.94 | +5% |
| 10 | 1.01 | 0.96 | +5% |

### sync-002: `torch.where` vs boolean masking

**结论**: boolean masking 触发 CPU sync 以确定输出形状。`torch.where` 产生固定尺寸输出，快 1.2-3.8×。

### sync-003: 无 synchronize() 的 CUDA 计时

**结论**: 不调用 `torch.cuda.synchronize()` 的计时只测量 CPU 提交延迟 (~15-70µs)，而非 GPU 执行时间。误差随矩阵增大而增大。

| 矩阵 | 无 sync (µs) | 有 sync (µs) | 实际时间 (CUDA events) | **误差** |
|------|-------------|-------------|---------------------|---------|
| 64×64 | 27 | 130 | 83 | 4.8× |
| 256×256 | 27 | 285 | 207 | **10.6×** |
| 1024×1024 | 71 | 1044 | 525 | **14.8×** |

---

## Tensor Layout

### layout-001: 隐式 `.contiguous()` 拷贝 — NULL RESULT

**结论**: CUDA 12.8 上，5 层 op chain 在 non-contiguous tensor 上仅触发 **1 次** `aten::copy_`。性能损失 1.0-1.1×。CUDA 12.8 的 stride-aware kernels 已经大幅减少了隐式拷贝。

**旧预期**: 5-15 次隐式拷贝，2-10× 减速。**已被 PyTorch 2.11 修复。**

### layout-002: LLM logits 的 cross_entropy layout 陷阱 — 严重确认

**结论**: `F.cross_entropy(logits.permute(0, 2, 1), targets)` 对 LLM 形状的 logits 慢 **17-118×**。使用 `log_softmax + gather` 避免 permute。

| 形状 (B×S×V) | CE+permute (ms) | log_softmax+gather (ms) | **加速比** |
|---------------|-----------------|------------------------|-----------|
| 1×128×50257 | 51.55 | 0.45 | **114.7×** |
| 2×512×50257 | 56.27 | 3.43 | **16.4×** |
| 1×128×32000 | 32.86 | 0.28 | **118.0×** |

**注意**: `.contiguous()` 在 permute 后无帮助——问题不在 contiguity，而在 permute 本身的 kernel 调度。

### layout-003: index_select vs 普通索引 — NULL RESULT

**结论**: CUDA 12.8 上 `index_select` 和普通索引 (`x[:, idx]`) 性能几乎相同（1.0-1.2×）。预期 2-6× 差异未观察到。

---

## Memory Allocator

### memory-001: 分配器碎片化

**结论**: CUDA 12.8 caching allocator 处理 small→large 和 large→small 两种分配顺序均为 **0% waste**。极端情况（800×15MB 填充后申请 3GB）仍会 OOM。

---

## Mixed Precision

### precision-001: FP16 eps=1e-8 NaN 陷阱 — NULL RESULT

**结论**: PyTorch 2.11 的 AMP `GradScaler` 有效防止了 FP16 eps 下溢。所有 eps 值（1e-8 到 1e-3）在 500 步内均未产生 NaN。

**旧预期**: eps=1e-8 在 50-200 步内产生 NaN。**AMP GradScaler 已修复。**

### precision-002: Tensor Core 利用率

**结论**: T4 Tensor Cores 在 ~384×384 开始激活，峰值利用率 37%（23.8 TFLOPS / 65 TFLOPS 理论值）。

| 指标 | 值 |
|------|-----|
| FP32 峰值 | 4.1 TFLOPS @ 1536×1536 |
| FP16 峰值 | 23.8 TFLOPS @ 3072×3072 |
| FP16 加速比 | **6.4×** @ 8192×8192 |
| Tensor Core 利用率 | 37% @ 1536×1536 |

---

## torch.compile

### compile-002: Non-contiguous max() 在 compile 下 — NULL RESULT 反转

**结论**: Eager 模式下 `torch.max()` 在 non-contiguous tensor 上慢 1.6-5.5×。但 **`torch.compile` 消除了这一差异（1.0×）**。

| 尺寸 | Eager 连续 | Eager 非连续 | Comp 连续 | Comp 非连续 | Eager 减速 | **Comp 减速** |
|------|-----------|------------|----------|-----------|----------|-------------|
| 256×256 | 23.5 µs | 38.2 µs | 86.1 µs | 85.6 µs | 1.6× | **1.0×** |
| 4096×4096 | 269.8 µs | 1308.7 µs | 426.0 µs | 418.5 µs | 4.8× | **1.0×** |

**旧预期**: compile 下 non-contiguous max() 慢 8×。**PT 2.11 inductor 已修复。**

---

## 环境

- GPU: Tesla T4 (SM 7.5, 15.6 GB VRAM)
- CUDA: 12.8
- PyTorch: 2.11.0+cu128
- Colab: 免费套餐 (~10 min GPU 窗口)
