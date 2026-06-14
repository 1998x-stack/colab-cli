# Colab GPU Keepalive: Implementation Guide

**Date:** 2026-06-14 | **Based on:** 40+ session root-cause analysis (see `docs/colab-gpu-keepalive.md`)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOUR LOCAL MACHINE                           │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ Terminal │   │  Cron    │   │  Cron    │   │     Cron       │  │
│  │ (ws-1)   │   │ (ws-2)   │   │ (ws-3)   │   │  (fetch.sh)    │  │
│  │ T+0      │   │ T+6min   │   │ T+13min  │   │  every 2-5min  │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └───────┬────────┘  │
│       │              │              │                  │           │
│       │    WSS       │    WSS       │    WSS           │ REST      │
│       │    exec -f   │    exec -f   │    exec -f       │ download  │
│       │              │              │                  │           │
└───────┼──────────────┼──────────────┼──────────────────┼───────────┘
        │              │              │                  │
   ─────┼──────────────┼──────────────┼──────────────────┼───────────
        │         GFW/NAT (~8-12 min per WSS)            │
   ─────┼──────────────┼──────────────┼──────────────────┼───────────
        │              │              │                  │
┌───────┼──────────────┼──────────────┼──────────────────┼───────────┐
│       ▼              ▼              ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    COLAB T4 VM                               │   │
│  │                                                              │   │
│  │  ┌─────────────────────┐    ┌──────────────────────────────┐ │   │
│  │  │  Jupyter Kernel     │    │  Detached Training Process   │ │   │
│  │  │  (serial queue)     │    │  (start_new_session=True)    │ │   │
│  │  │                     │    │                              │ │   │
│  │  │  [ws-1: RUNNING]    │    │  train.py                    │ │   │
│  │  │  [ws-2: QUEUED]     │    │  ├─ logs/train.log           │ │   │
│  │  │  [ws-3: QUEUED]     │    │  ├─ metrics.csv              │ │   │
│  │  │                     │    │  ├─ pngs/training_curves.png │ │   │
│  │  └─────────────────────┘    │  └─ checkpoints/             │ │   │
│  │                              └──────────────────────────────┘ │   │
│  │  Keep-alive daemon: DEAD (403 IAM deadlock, T+61s always)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key facts:**

| Component | Transport | Reliability | Role |
|-----------|-----------|-------------|------|
| `colab exec` WebSocket | WSS to `*.prod.colab.dev` | ~8-12 min from China | **Primary liveness signal** |
| `colab download` | REST to `*.prod.colab.dev` | High (survives WSS drops) | Output retrieval |
| `KeepAliveAssignment` RPC | REST to `colab.pa.googleapis.com` | **0% success** (IAM deadlock) | Dead on arrival |
| Detached training | Local process on VM | High (survives all network drops) | Actual computation |

## 2. The Relay Handoff Protocol

### 2.1 Why relay?

From China, each `colab exec` WebSocket survives ~8-12 minutes before GFW/NAT drops it.
Training sessions need continuous WebSocket coverage because:

- **No WebSocket → session dies in ~2-3 minutes**
- The keep-alive daemon is broken (IAM deadlock), so WebSocket is the ONLY liveness signal

### 2.2 How it works

Jupyter kernels execute code **serially** — one cell at a time. When a new `colab exec`
connects while another is running, its code **queues** behind the current execution.
Critically, the **WebSocket connects immediately** — only code execution waits.

```
Kernel execution queue:
  T+0:00   [ws-1: RUNNING watchdog loop]  ← WebSocket active
  T+6:00   [ws-1: RUNNING] [ws-2: QUEUED]  ← ws-2 WebSocket connected, code waiting
  T+7:00   [ws-1: EXITS]   [ws-2: STARTS]  ← handoff gap: ~5 seconds
  T+13:00  [ws-2: RUNNING] [ws-3: QUEUED]  ← ws-3 WebSocket connected
  T+14:00  [ws-2: EXITS]   [ws-3: STARTS]  ← handoff gap: ~5 seconds
```

**The ~5-second handoff gap** is well within the ~2-3 minute grace period before
Colab reclaims the GPU.

### 2.3 Timing diagram

```
Watchdog  Lifetime    WebSocket  Kernel execution
──────────────────────────────────────────────────
ws-1      0:00-7:00   ████████████████
                               
ws-2      6:00-13:00       ░░░░░░████████████████
                           (queued)(executing)
ws-3      13:00-20:00                  ░░░░░░████████████████
                                      (queued)(executing)
──────────────────────────────────────────────────
Session   0:00 ──────────────────────────────────── 20:00+
liveness  ████████████████████████████████████████████████████
```

**Window design:**
- Watchdog runs for **7 minutes** (safe inside 8-12 min WSS stability window)
- Next watchdog starts at **T+6 minutes** (1-minute overlap for queue safety)
- `--timeout 540` (9 min) on `colab exec` gives 2-min buffer for watchdog to complete

## 3. File Inventory

### 3.1 Scripts you upload to the VM (once)

| File | Purpose | Upload needed? |
|------|---------|----------------|
| `train.py` | Your training script (project-specific) | Yes (`colab upload`) |
| `keepalive/watchdog.py` | Relay watchdog (ws-2, ws-3, ...) | Yes (`colab upload`) |

### 3.2 Scripts you run locally

| File | Purpose | How executed |
|------|---------|-------------|
| `keepalive/launch.py` | Bootstrap + ws-1 watchdog | `colab exec -f launch.py --timeout 540` |
| `keepalive/fetch.sh` | Cron output retrieval | `bash fetch.sh` (local) |

### 3.3 Utility libraries (uploaded once, imported by train.py)

| File | Purpose |
|------|---------|
| `scripts/log_utils.py` | Logger, MetricsCSV, SummaryJSON, Tee |
| `scripts/plot_utils.py` | plot_loss_acc, plot_rl_progress, plot_loss |

### 3.4 VM output structure

```
/content/<project>-output/
├── train.pid                   # Training process PID (written by launch.py)
├── watchdog_counter            # Auto-increment counter for watchdog naming
├── logs/
│   ├── train.log               # Training output (Logger)
│   └── watchdog.log            # Watchdog heartbeats + GPU stats
├── metrics.csv                 # Per-epoch structured metrics
├── pngs/
│   └── training_curves.png     # Overwritten every N epochs
├── checkpoints/                # Model weights (excluded from cron fetch)
└── summary.json                # Final run metadata
```

## 4. Step-by-Step Deployment

### 4.1 One-time setup

```bash
# Create the keepalive scripts directory
mkdir -p .claude/skills/colab-cli/scripts/keepalive

# Ensure proxy is configured (Config B — HTTP CONNECT, most reliable)
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

### 4.2 Session launch

```bash
# Step 1: Provision GPU session
colab new --gpu T4 -s my-training

# Step 2: Upload training script + utilities + watchdog
colab upload -s my-training train.py /content/train.py
colab upload -s my-training scripts/log_utils.py /content/log_utils.py
colab upload -s my-training scripts/plot_utils.py /content/plot_utils.py
colab upload -s my-training scripts/keepalive/watchdog.py /content/watchdog.py

# Step 3: Launch training + ws-1 watchdog
colab exec -s my-training -f scripts/keepalive/launch.py --timeout 540 &
```

### 4.3 Relay watchdogs (via cron)

```bash
# At T+6 minutes — launch ws-2
colab exec -s my-training -f scripts/keepalive/watchdog_remote.py --timeout 540 &
#   ↑ NOTE: watchdog.py is already on the VM at /content/watchdog.py
#   But colab exec -f reads LOCAL files. So you need a thin launcher or use stdin.

# Alternative: Use stdin to run the already-uploaded watchdog on the VM
echo 'exec(open("/content/watchdog.py").read())' | colab exec -s my-training --timeout 540 &

# At T+13 minutes — launch ws-3
echo 'exec(open("/content/watchdog.py").read())' | colab exec -s my-training --timeout 540 &

# ... repeat every 7 minutes
```

### 4.4 Output fetching (via cron, every 2-5 minutes)

```bash
bash scripts/keepalive/fetch.sh
```

## 5. Cron Setup

### 5.1 CronCreate for relay watchdogs

```
CronCreate:
  cron: "6,13,20,27,34,41,48,55 * * * *"   # Every 7 minutes from T+6
  prompt: "Launch next Colab watchdog: echo 'exec(open(\"/content/watchdog.py\").read())' | colab exec -s my-training --timeout 540 &"
  durable: false
```

### 5.2 CronCreate for output fetching

```
CronCreate:
  cron: "*/3 * * * *"          # Every 3 minutes
  prompt: "Fetch Colab outputs: bash scripts/keepalive/fetch.sh"
  durable: false
```

### 5.3 Manual cron (one-shot for known training duration)

For a 30-minute training run with 7-minute watchdog windows:
- ws-1 at T+0 (launch.py)
- ws-2 at T+6 min
- ws-3 at T+13 min
- ws-4 at T+20 min
- ws-5 at T+27 min

```bash
# Launch all watchdogs at once with staggered sleep:
colab exec -s my-training -f scripts/keepalive/launch.py --timeout 540 &
sleep 360  && echo 'exec(open("/content/watchdog.py").read())' | colab exec -s my-training --timeout 540 &
sleep 780  && echo 'exec(open("/content/watchdog.py").read())' | colab exec -s my-training --timeout 540 &
sleep 1200 && echo 'exec(open("/content/watchdog.py").read())' | colab exec -s my-training --timeout 540 &
sleep 1620 && echo 'exec(open("/content/watchdog.py").read())' | colab exec -s my-training --timeout 540 &
```

## 6. Training Script Integration

Your `train.py` must produce structured outputs for the cron fetch to be useful.
Use the utility libraries:

```python
"""train.py — example training script with structured outputs."""
import os, sys, time
import torch

# Import utilities (uploaded to /content/)
sys.path.insert(0, "/content")
from log_utils import Logger, MetricsCSV, SummaryJSON, detect_output_dir, setup_output_dirs
from plot_utils import plot_loss_acc

# ── Setup ───────────────────────────────────────────────
PROJECT = "my-project"
OUT_DIR = detect_output_dir(PROJECT)
setup_output_dirs(OUT_DIR)

logger = Logger(f"{OUT_DIR}/logs/train.log")
csv = MetricsCSV(f"{OUT_DIR}/metrics.csv",
                 ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                  "elapsed_s", "lr"])
summary = SummaryJSON(f"{OUT_DIR}/summary.json")

# Write PID for watchdog monitoring
with open(f"{OUT_DIR}/train.pid", "w") as f:
    f.write(str(os.getpid()))

# ── Training loop ───────────────────────────────────────
logger.log(f"TRAIN_START device={torch.cuda.get_device_name(0)}")
metrics = {"batch_losses": [], "eval_losses": [], "eval_accs": [], "lr_values": []}
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    # ... training ...
    train_loss = ...
    metrics["batch_losses"].extend(batch_losses)

    # ... evaluation ...
    test_loss, test_acc = ...

    # Record metrics (crash-safe — written immediately)
    csv.write_row(epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
                  test_loss=test_loss, test_acc=test_acc,
                  elapsed_s=time.time() - t0, lr=scheduler.get_last_lr()[0])

    # Update visualization (overwritten each epoch)
    plot_loss_acc(metrics, f"{OUT_DIR}/pngs/training_curves.png",
                  title=f"{PROJECT} — Epoch {epoch+1}/{NUM_EPOCHS}")

    logger.log(f"Epoch {epoch+1}/{NUM_EPOCHS} done — "
               f"train_loss={train_loss:.4f} test_acc={test_acc:.4f} "
               f"elapsed={time.time()-t0:.0f}s")

# ── Completion ──────────────────────────────────────────
summary.write({"test_acc": best_acc, "epochs_completed": NUM_EPOCHS,
               "total_time_s": time.time() - t0})
logger.log(f"TRAIN_COMPLETE best_acc={best_acc:.4f}")
```

## 7. The Fetch Loop

### 7.1 What fetch.sh does

1. **Session check** — `colab sessions` to verify session is alive
2. **Tar on VM** — `tar -czf /content/output.tar.gz -C /content <project>-output/` (excludes checkpoints/)
3. **Download tar** — `colab download` via REST (survives WebSocket drops)
4. **Extract locally** — `tar -xzf output.tar.gz`
5. **Report** — tail logs, tail CSV, list PNGs
6. **Death detection** — if session gone, save death notice with last known state

### 7.2 What you see each fetch tick

```
=== [14:32:05] Fetch: my-training ===
--- Session alive: my-training (runtime: 00:18:23) ---
--- Downloaded: output.tar.gz (45 KB) ---

=== TRAIN LOG (last 3 lines) ===
[14:31:48] Epoch 12/50 done — train_loss=0.3421 test_acc=0.7823 elapsed=1080s
[14:31:48] Epoch 13/50 | Batch 100 | loss=0.3102 | avg100=0.3245 | lr=0.0451 | elapsed=31s
[14:32:00] Epoch 13/50 | Batch 200 | loss=0.2891 | avg100=0.3102 | lr=0.0448 | elapsed=62s
  (247 lines total)

=== METRICS CSV (last 3 rows) ===
10,0.4523,0.7123,0.5234,0.7410,900.0,0.056000
11,0.3891,0.7512,0.4623,0.7650,990.0,0.051000
12,0.3421,0.7834,0.4234,0.7823,1080.0,0.045000

=== PNGs ===
training_curves.png (245 KB, modified 14:31:48)
```

## 8. Troubleshooting

### 8.1 Session dies before first watchdog completes

**Symptom:** Session dead at T+8 min despite ws-1 running.

**Causes:**
- Proxy config wrong — WebSocket can't connect. Flip config A↔B.
- GPU quota exhausted — try another account (`cb`, `cc`, `clb`).
- `colab exec --timeout` too short (< watchdog duration).

### 8.2 Handoff gap too large (session dies between watchdogs)

**Symptom:** Session dies at handoff point (T+7min, T+14min, etc.)

**Causes:**
- Next watchdog started too late — start at T+6 (not T+7).
- Watchdog took >7 min to start executing from queue — reduce current watchdog's sleep interval.
- `colab exec` failed to connect — proxy issue. Check `colab sessions` first.

### 8.3 Training runs but no logs appear in fetch

**Symptom:** Fetch returns empty or stale logs.

**Causes:**
- `PYTHONUNBUFFERED=1` not set — stdout buffered in subprocess.
- `python -u` not used — same issue.
- Logger not flushing — ensure `flush=True` on all print() calls.
- Wrong output directory — check `detect_output_dir()` logic.

### 8.4 Fetch download fails (>600MB)

**Symptom:** `IncompleteRead` on tar download.

**Fix:** Exclude checkpoints from the tar. The fetch script should use:
```bash
tar -czf /content/output.tar.gz --exclude='checkpoints' -C /content <project>-output/
```

### 8.5 Watchdog script not found on VM

**Symptom:** `echo 'exec(open("/content/watchdog.py").read())'` fails with FileNotFoundError.

**Fix:** Verify upload succeeded: `colab ls -s my-training /content/watchdog.py`

## 9. Reference: All Constraints

| Constraint | Value | Source |
|-----------|-------|--------|
| China WSS stability | 8-12 min per connection | Empirical, 40+ sessions |
| Session grace period | ~2-3 min after last WSS | Measured (Test 1) |
| Keep-alive daemon lifetime | 61 seconds (always) | Source code + history |
| Handoff gap | ~5 seconds | Measured (Test 2) |
| Watchdog window | 7 minutes | Design choice (safe margin) |
| Watchdog overlap | 1 minute | Design choice |
| Free tier session limit | 12 hours | Google policy |
| Free tier GPU types | T4 (primary), occasionally others | Quota-dependent |
| GPU sessions per account | 1 | Free tier limit |
| `colab exec --timeout` | 540s (9 min) | > watchdog window |
| First-session warmup | 7-10 min overhead | CUDA JIT + data download |
| Checkpoint download limit | ~600 MB through proxy | IncompleteRead |
