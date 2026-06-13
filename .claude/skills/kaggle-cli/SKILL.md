---
name: kaggle-cli
description: >
  Use when working with Kaggle Notebooks from the terminal — pushing GPU training
  jobs, downloading outputs, managing datasets, or monitoring kernel status. Triggers
  on mentions of Kaggle, kaggle CLI, kaggle notebooks, kaggle kernels, running Python
  on kaggle GPU, or needing free cloud GPU compute. Also trigger when the user wants
  to run a training script remotely, push a job to kaggle, check on kaggle training
  progress, manage kaggle datasets, or compare kaggle vs colab for GPU training.
---

# Kaggle CLI

Command-line interface for Kaggle Notebooks — push training scripts to cloud GPU, monitor remotely, and download results. A complementary GPU platform to Colab with a REST-based push model (no WebSocket dependency).

## Mental model

Kaggle Notebooks run Python code on remote VMs with GPU acceleration. Unlike Colab's interactive exec model, Kaggle uses a **push model**: you upload a script, it runs in the background on Kaggle's servers, and you poll for results. No long-lived connection needed — the push is a single REST API call.

**Why this matters from China:** Colab's `exec` depends on a WebSocket (`*.prod.colab.dev`) that frequently drops through proxies. Kaggle's `kernels push` is a single HTTPS request — no connection to maintain, no WebSocket to drop.

**Free-tier GPU:** 30 hours/week, transparent counter, resets weekly. GPU type is auto-assigned — you get either a P100 (16GB) or T4 x2 (~32GB total). Cannot manually choose.

| Resource | Limit | Notes |
|----------|-------|-------|
| GPU hours/week | 30h | Transparent counter, resets weekly |
| Single session max | 12h | Hard cutoff, save checkpoints |
| Idle timeout | ~60 min | Non-active session reclaimed |
| GPU types | P100 (16GB) or T4 x2 (~32GB) | Auto-assigned, not selectable |
| RAM | ~16 GB (up to 31 GB) | Varies by session |
| Disk (/kaggle/working) | ~20 GB | Ephemeral, lost on session end |
| Disk (/kaggle/input) | Read-only | For mounted datasets |

Key distinctions from Colab:

- **Push, don't connect.** `kaggle kernels push` sends everything in one REST call. Script runs on Kaggle's servers — your local terminal can go offline.
- **No WebSocket.** The entire workflow is REST: push, status, logs, output download. Zero long-lived connections.
- **Script mode runs plain `.py` files.** No notebook conversion needed. Set `"kernel_type": "script"` in metadata.
- **`/kaggle/working/` is ephemeral.** Files vanish when the session ends. Save outputs via "Save Version" or download immediately after completion.
- **No Drive mount.** Kaggle can't mount Google Drive. Use Kaggle Datasets for persistent storage between sessions.

## Quick reference

```bash
# Authentication (pick one)
export KAGGLE_API_TOKEN="KGAT_xxxx"                  # env var (takes precedence)
# OR save token to ~/.kaggle/access_token             # file-based (persistent)

# Kernel (notebook/script) management
kaggle kernels init -p ./project-dir                 # generate kernel-metadata.json template
kaggle kernels push -p ./project-dir                 # push + run (the core command)
kaggle kernels status <owner>/<slug>                 # check run status
kaggle kernels logs <owner>/<slug>                   # view stdout/stderr
kaggle kernels output <owner>/<slug> -p ./output     # download output files
kaggle kernels list --mine                           # list your kernels
kaggle kernels list -s <search>                      # search public kernels
kaggle kernels pull <owner>/<slug>                   # download kernel source

# Dataset management
kaggle datasets init -p ./dataset-dir                # initialize dataset metadata
kaggle datasets create -p ./dataset-dir              # create/upload dataset
kaggle datasets download <owner>/<dataset>           # download dataset
kaggle datasets list --mine                          # list your datasets

# Config
kaggle config view                                    # show current config
kaggle config set -n proxy -v http://127.0.0.1:7890  # set HTTP proxy (if needed)
```

## Authentication

Kaggle uses **KGAT tokens** (newer auth, what we use). These are long-lived API tokens, not username+key pairs.

### This project's tokens

Four tokens are stored in `.kaggle/` at the project root. Token 4 (xieming1998) is the active dev account with kernels:

| Token file | Account |
|-----------|---------|
| `.kaggle/access_token4` | **xieming1998** (active) |
| `.kaggle/access_token1` | backup account |
| `.kaggle/access_token2` | backup account |
| `.kaggle/access_token3` | backup account |

**To use a specific account**, set `KAGGLE_API_TOKEN` before each command:

```bash
# Token 4 (xieming1998) — main dev account
export KAGGLE_API_TOKEN="$(cat .kaggle/access_token4)"
kaggle kernels push -p ./project-dir

# Token 1 — backup account
export KAGGLE_API_TOKEN="$(cat .kaggle/access_token1)"
kaggle kernels push -p ./project-dir
```

For persistent auth, copy the desired token to `~/.kaggle/access_token`. The CLI reads from there automatically — no env var needed.

**Token management gotcha:** Unlike Colab where each account has an isolated `$HOME`, Kaggle CLI reads a single `~/.kaggle/access_token` file. To switch accounts, either use `KAGGLE_API_TOKEN` env var (per-command) or overwrite `~/.kaggle/access_token`. The env var approach is safer for multi-account workflows — it can't accidentally persist the wrong account.

## Push workflow (the core loop)

Pushing a script to Kaggle is a 3-step process: configure metadata → push → monitor.

### 1. Project structure

```
projects/my-experiment/
├── train.py                 # Your training script
└── kernel-metadata.json     # Kaggle push configuration
```

### 2. kernel-metadata.json

The minimal config for a GPU training script:

```json
{
  "id": "your-username/your-slug",
  "title": "My Training Run",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
```

Key fields:
- `kernel_type`: `"script"` for plain `.py` files, `"notebook"` for `.ipynb`
- `enable_gpu`: `true` to request GPU (P100 or T4 x2)
- `enable_internet`: `true` to allow pip install and external network access
- `is_private`: `true` to hide from public listings
- `dataset_sources`: list of `"owner/dataset-name"` to mount at `/kaggle/input/`
- `id`: auto-updated on first push — use `your-username/your-slug` format

**Important:** The `id` slug must be unique per kernel. Pushing to an existing slug creates a new version of that kernel, not a separate run. For distinct experiments, use different slugs.

### 3. Push and monitor

```bash
# Push (single REST call — returns immediately)
kaggle kernels push -p ./project-dir

# Check status
kaggle kernels status xieming1998/my-training

# View logs (streams stdout/stderr as JSON with timestamps)
kaggle kernels logs xieming1998/my-training

# Download output when complete
kaggle kernels output xieming1998/my-training -p ./output
```

The push command prints a URL where you can watch progress in the browser:
```
Kernel version 1 successfully pushed.
Please check progress at https://www.kaggle.com/code/xieming1998/my-training
```

### 4. Full automation script

See `scripts/push_and_wait.py` for a Python script that pushes, polls until complete/error, and downloads output — the kaggle equivalent of `colab exec -f launch.py`.

## Environment detection in train.py

Training scripts should auto-detect the platform to set correct paths:

```python
import os

if os.path.exists("/kaggle/working/"):
    # Kaggle environment
    DATA_DIR = "/kaggle/input/my-dataset"
    OUTPUT_DIR = "/kaggle/working/output"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
elif os.path.exists("/content/"):
    # Colab environment
    DATA_DIR = "/content/data"
    OUTPUT_DIR = "/content/output"
    CHECKPOINT_DIR = "/content/checkpoints"
else:
    # Local environment
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output"
    CHECKPOINT_DIR = "./checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

## GPU compatibility (critical gotcha)

Kaggle auto-assigns either a **P100** or **T4 x2**. They have different CUDA compatibility:

| GPU | CUDA Capability | Pre-installed PyTorch | Action needed |
|-----|----------------|----------------------|---------------|
| T4 x2 | sm_75 | Works out of the box | None |
| P100 | sm_60 | **Incompatible** | Reinstall PyTorch with CUDA 12.6 |

The pre-installed PyTorch (`2.10.0+cu128`, CUDA 12.8) dropped support for Pascal GPUs (sm_60). If you get a P100, you'll see:
```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
```

**Fix** — force-reinstall PyTorch with CUDA 12.6 at the top of your script:

```python
import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "--force-reinstall",
    "torch", "torchvision",
    "--extra-index-url", "https://download.pytorch.org/whl/cu126"
], check=True, timeout=300)

import torch  # Now works on both P100 and T4
```

This adds ~3-4 minutes to startup but makes the script work regardless of which GPU Kaggle assigns. On a T4 session, the reinstall is harmless (just wasted time). See `references/gotchas.md` for details on detecting GPU type before reinstalling.

## Monitoring with cron

Kaggle kernels run asynchronously — perfect for cron-based monitoring. No WebSocket, no connection to maintain:

```bash
# Cron: check kernel status every 10 minutes
CronCreate cron="*/10 * * * *" prompt="Check kaggle kernel xieming1998/my-training status. If complete, download output to ./output/" durable=true
```

Unlike Colab, you don't need a watchdog heartbeat — the kernel keeps running on Kaggle's servers regardless.

**Warning: empty logs do NOT mean the kernel is stuck.** Kaggle's log pipeline buffers stdout/stderr for longer-running GPU+internet scripts. All 105 log lines from a 37-minute ViT training run appeared atomically at completion — zero output during the entire run. Before concluding a kernel is stuck:
1. Check elapsed time vs. expected duration. Wait for expected duration + 50% buffer.
2. Check for side effects (files appearing in datasets, external storage).
3. Push ONE minimal GPU test kernel (different account) to verify infrastructure isn't globally down.
4. Never push duplicate kernels assuming the original is dead — you'll exhaust both GPU slots.

See `references/gotchas.md` #17 for the full field example.

## Checkpoint persistence

`/kaggle/working/` is wiped when the session ends. Three strategies:

**A. Save Version (recommended).** In the Kaggle UI, "Save Version" > "Save & Run All" re-executes and persists output files. Programmatic equivalent not yet available via CLI.

**B. Output → Dataset.** After training completes, create a dataset from the output:

```bash
kaggle kernels output xieming1998/my-training -p ./checkpoints/
# Then upload as a new dataset version for the next session
kaggle datasets version -p ./checkpoints/ -m "epoch 10 checkpoint"
```

Next session, mount it: add `"dataset_sources": ["xieming1998/my-checkpoints"]` to `kernel-metadata.json`. Files appear at `/kaggle/input/my-checkpoints/`.

**C. External upload.** At the end of `train.py`, upload checkpoints to HuggingFace Hub, Google Drive API, or your own server.

```python
# Example: upload to HuggingFace Hub at end of training
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/kaggle/working/checkpoints",
    repo_id="your-username/my-model-checkpoints",
    repo_type="model",
)
```

## Datasets as persistent storage

Kaggle Datasets are the equivalent of Google Drive for Colab — they persist across sessions and mount read-only at `/kaggle/input/`.

```bash
# Create a new dataset
kaggle datasets init -p ./my-data
# Edit dataset-metadata.json, then:
kaggle datasets create -p ./my-data --dir-mode zip

# Mount in a kernel by adding to kernel-metadata.json:
# "dataset_sources": ["xieming1998/my-data"]

# Download locally
kaggle datasets download xieming1998/my-data
```

**Workflow for multi-session training:**
1. Session 1: train.py writes checkpoints to `/kaggle/working/checkpoints/`
2. After completion: `kaggle kernels output xieming1998/my-training -p ./ckpt/`
3. Upload as dataset: `kaggle datasets version -p ./ckpt/ -m "epoch 5"`
4. Session 2: mount dataset → train.py reads from `/kaggle/input/my-checkpoints/`

## Integration with colab-cli

Kaggle complements Colab, not replaces it. Use both:

| Scenario | Use |
|----------|-----|
| Quick prototype / debug | Colab (faster iteration) |
| Long training (>2h) | Kaggle (no connection dependency) |
| TPU workloads | Colab (Kaggle has no TPU) |
| Multi-GPU / >16GB VRAM | Kaggle (T4 x2 potential) |
| Batch experiments | Kaggle (push many scripts, poll later) |
| Need Drive mount | Colab |

Both platforms can share the same `train.py` — just include environment detection (see above).

## Comparison with Colab

| Dimension | Colab | Kaggle |
|-----------|-------|--------|
| Execution model | Interactive (WebSocket) | Push (REST) |
| China proxy stability | Poor (WSS drops) | Good (HTTPS only) |
| GPU quota | Opaque, dynamic | 30h/week, transparent |
| GPU types | T4 (16GB) | P100 (16GB) or T4 x2 (~32GB) |
| TPU | Yes | No |
| Drive mount | Yes | No (use Datasets) |
| Session max | ~12h | ~12h |
| Keep-alive needed | Yes (watchdog) | No |
| Stop running job | CLI supported | UI only |
| Script mode | N/A (upload .py) | Native |

## Limitations

- **Cannot stop a running kernel via CLI.** Must use the Kaggle website to cancel.
- **Cannot choose GPU type.** P100 vs T4 x2 is auto-assigned.
- **No API for "Save Version."** Output persistence requires UI interaction or external upload.
- **Single `~/.kaggle/access_token` file.** Multi-account requires env var switching.
- **Kaggle may require phone verification** for first-time GPU use. From China, this can be difficult (reCAPTCHA). Complete verification before relying on Kaggle for training.
- **GPU quota is per-account, per-week.** Running the same script across 4 accounts gives 120h/week total, but check ToS compliance.

## Training outputs — logs, plots, metrics

Every training script should produce three structured artifacts for glance-and-decide monitoring. Use `scripts/log_utils.py` and `scripts/plot_utils.py` for reusable implementations (shared with colab-cli).

### Output directory structure

```
<out_dir>/
├── logs/train.log          # Timestamped training log
├── metrics.csv             # Per-epoch structured metrics
├── pngs/training_curves.png  # Multi-panel visualization
├── checkpoints/            # Model checkpoints (persist to Kaggle Datasets)
└── summary.json            # Final run metadata
```

Kaggle output goes to `/kaggle/working/<project>-output/`:
```python
from log_utils import detect_output_dir, setup_output_dirs

out_dir = detect_output_dir("my-project")  # → /kaggle/working/my-project-output/
setup_output_dirs(out_dir)
```

### log_utils.py — reusable logging

```python
from log_utils import Logger, MetricsCSV, SummaryJSON

# Timestamped log to file + stdout
logger = Logger(f"{out_dir}/logs/train.log")
logger.log("Training started")

# Structured CSV — header written on creation, rows appended
csv = MetricsCSV(f"{out_dir}/metrics.csv",
                 ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                  "elapsed_s", "lr"])
csv.write_row(epoch=1, train_loss=1.23, train_acc=0.45,
              test_loss=1.34, test_acc=0.50, elapsed_s=180, lr=0.089)

# Final summary
summary = SummaryJSON(f"{out_dir}/summary.json")
summary.write({"test_acc": 0.87, "epochs_completed": 5, "total_time_s": 900})
```

### plot_utils.py — reusable visualization

```python
from plot_utils import plot_loss_acc, plot_rl_progress, plot_loss

# Classification (4-panel: loss, accuracy, LR, distribution)
plot_loss_acc(metrics, f"{out_dir}/pngs/training_curves.png",
              title="My Model — Dataset", size_label="small")

# RL training (4-panel: reward, episode length, exploration, Q-values)
plot_rl_progress(metrics, f"{out_dir}/pngs/rl_progress.png",
                 title="TD3 — HalfCheetah", solved_threshold=10000)

# Minimal single-panel loss plot
plot_loss(metrics, f"{out_dir}/pngs/loss.png", title="Loss")
```

### Log format convention

Per-N-batches, one self-contained line:
```
[HH:MM:SS] Ep 1/5 | Batch 1000 | loss=1.3625 | avg100=1.3677 | lr=0.089451 | elapsed=97s
```

Note: Kaggle log output can buffer — all lines may appear atomically at completion (see gotchas.md #17). Do not assume empty logs = stuck.

### Metrics CSV convention

```
epoch,train_loss,train_acc,test_loss,test_acc,elapsed_s,lr
1,1.234000,0.456000,1.345000,0.500000,180.0,0.089000
```

One row per epoch. Crash-safe: write header on creation, append after each epoch.

### PNG conventions

- 4-panel figure (2×2 grid), overwritten periodically
- Panel 1: Loss curve, Panel 2: Accuracy, Panel 3: LR schedule, Panel 4: Loss distribution
- Always include reference lines (baselines, best-so-far)

## File paths

| Path | Read/Write | Persistent | Purpose |
|------|-----------|------------|---------|
| `/kaggle/working/` | Read + Write | No (session only) | Your code runs here, write outputs here |
| `/kaggle/input/` | Read-only | Yes (dataset-backed) | Mounted datasets appear here |
| `/kaggle/temp/` | Read + Write | No | Temporary files |
