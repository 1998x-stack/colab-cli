# GPU Check One-Liner

Since `colab exec` has no `-c` flag, pipe the inline Python via stdin:

```bash
echo 'import torch; print("CUDA:", torch.cuda.is_available()); print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"); print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else "N/A")' | colab exec --timeout 15
```

This will print something like:

```
CUDA: True
Device: Tesla T4
VRAM (GB): 15.8
```

## Why this works

- Colab VMs come with PyTorch pre-installed (CUDA 12.8, PyTorch 2.11.0+cu128, per gotchas.md).
- `torch.cuda.is_available()` returns `True` if a CUDA-capable GPU is detected, `False` if you got a CPU fallback.
- `torch.cuda.get_device_name(0)` returns the model name (e.g., `Tesla T4`, `Tesla L4`, `Tesla A100`).
- No files are created or uploaded -- the code is piped directly to the kernel.

## Alternative using nvidia-smi

```bash
echo 'import subprocess; subprocess.run(["nvidia-smi"])' | colab exec --timeout 15
```

The torch approach is preferred since it avoids parsing `nvidia-smi` output and works uniformly regardless of GPU type.
