# PyTorch CPU-GPU 传输基准测试

在 Colab T4（CUDA 12.8, PyTorch 2.11.0+cu128）上完成的 3 项 CPU-GPU 数据传输实验。

## 实验概览

| ID | 实验 | 关键结果 | 影响 |
|----|------|---------|------|
| transfer-001 | GPU→CPU pinned vs pageable | **9.2× 加速** (1.4→13.3 GB/s) | 所有训练循环 |
| transfer-002 | `.to(device, dtype)` 合并调用陷阱 | **4-12× 加速** (两步法 vs 合并) | 图像数据加载 |
| transfer-003 | 临时 `.pin_memory()` 反模式 | **0.6-1.2×** (影响小於预期) | DataLoader 设计 |

## transfer-001: Pinned Memory 传输不对称性

**结论**: GPU→CPU 方向 pinned memory 加速 9.2×。但 CUDA 12.8 上新发现：`.to('cpu', non_blocking=True)` 无显式 pinned memory 也能达到 13.3 GB/s。

```
2 GB tensor (float16):
  GPU→CPU default:  1.4 GB/s
  GPU→CPU pinned:  13.3 GB/s  ← 9.2× 加速
  CPU→GPU default: 10.1 GB/s  ← 无需 pinned 已接近 PCIe 上限
```

## transfer-002: dtype 转换顺序陷阱

**结论**: 对 uint8 数据调用 `.to(device='cuda', dtype=float32)` 会在 CPU 端先做类型转换（数据膨胀 4×），再传输到 GPU。两步法（先传 uint8，再在 GPU 上 cast）快 4-12×。

| 图像尺寸 | 合并调用 (ms) | 两步法 (ms) | 加速比 |
|----------|-------------|-----------|--------|
| 256×256 RGB | 0.54 | 0.13 | **4.0×** |
| 512×512 RGB | 1.86 | 0.46 | **4.0×** |
| 1024×1024 RGB | 4.50 | 0.96 | **4.7×** |
| 2048×2048 RGB | 54.52 | 5.00 | **10.9×** |
| 4096×4096 RGB | 252.11 | 20.50 | **12.3×** |

**正确做法**:
```python
# ❌ 慢：CPU 端先 cast float32（数据膨胀 4×），再传 GPU
x_gpu = x_uint8.to(device='cuda', dtype=torch.float32)

# ✅ 快：先传 uint8（数据小），GPU 端再 cast
x_gpu = x_uint8.to('cuda').to(torch.float32)
```

## transfer-003: 临时 pin_memory() 反模式

**结论**: 预期 `tensor.pin_memory().to('cuda', non_blocking=True)` 比直接 `.to('cuda')` 慢 1.5-2×。但 **CUDA 12.8 上未观察到** — 临时 pin 的开销仅 0.6-1.2×，多数情况下甚至略快。

这是一个**NULL RESULT** — 现代 CUDA 的 pinned staging buffer 机制已经优化了这个问题。

## 环境

- GPU: Tesla T4 (15.6 GB VRAM, PCIe 3.0 x16)
- CUDA: 12.8
- PyTorch: 2.11.0+cu128
- Colab: 免费套餐
