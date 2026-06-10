# GPU Check One-Liner for Colab Sessions

Since `colab exec` has no `-c` flag (the skill explicitly calls this out as gotcha #12), you cannot do:

```bash
# This DOES NOT work:
colab exec -c "import torch; print(torch.cuda.is_available())"
```

Instead, pipe code through stdin. Two approaches:

## Option 1: `nvidia-smi` (recommended -- no Python dependency)

```bash
echo 'import subprocess; print(subprocess.check_output(["nvidia-smi"]).decode())' | colab exec --timeout 10
```

This dumps the full GPU status table including driver version, GPU name (Tesla T4), memory, and utilization. Works on any GPU VM regardless of which Python packages are installed.

For a more concise GPU name + VRAM output:

```bash
echo 'import subprocess; r=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True); print(r.stdout.strip() or "No GPU found")' | colab exec --timeout 10
```

## Option 2: `torch.cuda` (standard Colab approach)

PyTorch is pre-installed on Colab VMs, so this also works:

```bash
echo 'import torch; print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' | colab exec --timeout 10
```

## Why this works

From the skill reference: Colab T4 VMs ship with CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.12, and a Tesla T4 (16 GB VRAM). Both `nvidia-smi` (system tool) and `torch` (pre-installed Python package) are always available on GPU-backed sessions. If the session had fallen back to CPU (e.g., GPU was rejected or unavailable), both commands will cleanly report no GPU.

## Quick decision

- Want the full `nvidia-smi` status table (driver, processes, memory)? Use Option 1.
- Just want a yes/no + GPU model name? Use Option 2 -- it's slightly simpler.
