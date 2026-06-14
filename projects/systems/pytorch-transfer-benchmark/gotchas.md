# pytorch-transfer-benchmark gotchas

## 1. uint8→float32 转换必须在 GPU 端做

**发现**: 任何图像 pipeline 加载 uint8 数据后，务必分两步：先用 `.to('cuda')` 传 uint8，再用 `.to(torch.float32)` 在 GPU 端转换。

**为什么**: PyTorch 的 `.to(device, dtype)` 合并调用在 CPU 端先 cast（uint8→float32 膨胀 4×），再传输。两步法的传输量只有 1/4。

**影响**: 4096×4096 RGB 图像传输慢 12.3×（252ms vs 20.5ms）。

**适用范围**: 所有从磁盘加载 uint8 图像的项目。

## 2. CUDA 12.8 上临时 pin_memory() 影响不大

**发现**: `tensor.pin_memory().to('cuda', non_blocking=True)` 相比直接 `.to('cuda')` 的额外开销仅 0.6-1.2×。CUDA 12.8 的 caching allocator 内部已经创建了 pinned staging buffer，所以用户的显式 pin 不会造成双重分配。

**与旧文档的差异**: 旧版 CUDA (<11.x) 上没有内部 staging buffer，所以显式 pin 会造成双重 pinned 分配——但 CUDA 12.8 的行为已经改变。

## 3. GPU→CPU 传输 pinned memory 仍然关键

**发现**: 虽然 `.to('cuda', non_blocking=True)` 的 CPU→GPU 方向足够快（~10 GB/s），但 GPU→CPU 方向仍需要 pinned memory 才能达到 PCIe 带宽上限（13.3 vs 1.4 GB/s）。

**实践**: DataLoader 的 `pin_memory=True` 仍然重要。不要移除。
