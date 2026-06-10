# Common Workflows

## Proxy preamble (China — REQUIRED before any command)

All commands below assume the proxy env vars are set. Copy-paste this preamble before running anything:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

## Pattern 1: One-shot training run

Provision → upload → install deps → launch detached → monitor → download results.

```bash
# 0. Proxy (REQUIRED from China)
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890

# 1. Create session
colab new --gpu T4 --session my-run

# 2. Upload all files
colab upload train.py /content/train.py
colab upload launch.py /content/launch.py
colab upload check_progress.py /content/check_progress.py

# 3. Launch (launch.py installs deps, spawns training, exits)
colab exec -f launch.py --timeout 120

# 4. Wait for warmup, then monitor
sleep 60
colab exec -f check_progress.py --timeout 15

# 5. When done, download checkpoints (must download files individually)
colab download checkpoints/best.pt ./best.pt
colab download checkpoints/final.pt ./final.pt
```

## Pattern 2: Iterative development on VM

```bash
# Create session once
colab new --gpu T4 --session dev

# Edit locally, upload, test
vim script.py
colab upload script.py /content/script.py
colab exec -f script.py --timeout 30

# Check output
colab exec -f check_progress.py --timeout 10
```

## Pattern 3: Ad-hoc exploration

Use stdin for quick experiments without creating files:

```bash
# Check installed packages
echo 'import pkg_resources; print([p.key for p in pkg_resources.working_set])' | colab exec

# Check GPU
echo 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))' | colab exec

# Find files
echo 'import os; [print(f) for f in os.walk("/content")]' | colab exec

# Install a package
echo 'import subprocess, sys; subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name"])' | colab exec
```

## Pattern 4: Ephemeral one-shot (`colab run`)

For simple scripts that don't need monitoring — provisions a fresh VM, runs the script, retrieves output, and auto-teardowns:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890
colab run --gpu T4 script.py
```

No need to manage sessions, uploads, or teardown. Best for batch jobs, benchmarks, or CI-style workflows.

## Pattern 5: Two parallel projects (1 GPU + 1 CPU)

Free tier limits to 1 GPU. Run CPU project concurrently with GPU:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890

# GPU session (project 1)
colab new --gpu T4 -s gpu-project
colab upload train_gpu.py /content/train.py && colab exec -s gpu-project ...

# CPU session (project 2) — in parallel
colab new -s cpu-project
colab upload train_cpu.py /content/train.py && colab exec -s cpu-project ...
```

Or run two GPU projects sequentially: create → train → download → stop, then repeat.

## Dealing with session death

Sessions die silently after ~2-4 hours. Your background processes die with them.

**Before starting a long run:**
- Set up checkpointing in your training script
- Know the download command: `colab download checkpoints/best.pt`

**If a session died:**
```bash
# Check what's left
colab sessions

# Create new session, re-upload, resume from last checkpoint
colab new --gpu T4 --session my-run-2
colab upload train.py /content/train.py
colab upload checkpoints/best.pt /content/checkpoints/best.pt  # if you downloaded it
colab exec -f launch.py --timeout 120
```

## Real-time monitoring loop

For watching a long run over time:

```bash
# Every 5 minutes, check progress
while true; do
  clear
  date
  colab exec -f check_progress.py --timeout 15
  sleep 300
done
```
