# cuda-dark-corners gotchas

## 1. `cross_entropy` + permute 是 LLM 训练的隐藏瓶颈

**发现**: `F.cross_entropy(logits.permute(0, 2, 1), targets)` 对于 LLM 典型形状（B×S×V）慢 17-118×。问题不在 contiguity——`.contiguous()` 无济于事。

**修复**: 使用 `log_softmax + gather` 或 reshape 方式。
```python
# ❌ 慢：permute 触发低效 kernel
loss = F.cross_entropy(logits.permute(0, 2, 1), targets)

# ✅ 快：不改变 layout
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs.reshape(-1, vocab_size), targets.reshape(-1))
```

**适用**: 所有 LLM 训练代码。batch size 越小，影响越大。

## 2. CUDA 12.8 修复了多个"经典"陷阱

以下陷阱在 CUDA 12.8 / PyTorch 2.11 上**未观察到**，旧文档需要更新：

| 旧陷阱 | 旧预期 | 实际观察 | 原因 |
|--------|--------|---------|------|
| 隐式 contiguous copies (layout-001) | 2-10× 减速, 5-15 次 copy | 1.0-1.1×, 1 次 copy | stride-aware kernels 改进 |
| index_select 2D+ 减速 (layout-003) | 2-6× 减速 | 1.0-1.2× | index_select 实现优化 |
| FP16 eps=1e-8 NaN (precision-001) | 50-200 步 NaN | 500 步无 NaN | AMP GradScaler 有效 |
| CUDA 首次调用 1.6s (launch-002) | 1.6s | 389ms | CUDA 12.8 初始化优化 |
| Non-contiguous max under compile (compile-002) | 8× 减速 | 1.0× | inductor 优化 |
| 临时 pin_memory 反模式 (transfer-003) | 1.5-2× 减速 | 0.6-1.2× | 内部 staging buffer |

**实践**: 依赖旧文档的优化建议前，先在目标 CUDA 版本上验证。

## 3. CUDA timing 必须用 synchronize() 或 CUDA events

**发现**: 不调用 `torch.cuda.synchronize()` 的 `time.perf_counter()` 只能测量 CPU 提交延迟（~15-70µs），无论 GPU kernel 实际耗时多少。误差可达 15×。

**修复**: 始终在 benchmark 中使用 `torch.cuda.Event`：
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... GPU work ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

## 4. 小 batch LLM 推理受 layout 陷阱影响最大

**发现**: `cross_entropy` permute 陷阱的加速比随 batch size 增大而减小：
- B=1: 115× 加速
- B=2: 17× 加速
- B=4: 6× 加速

原因是小 batch 时 permute kernel 的固定开销占比更大。
