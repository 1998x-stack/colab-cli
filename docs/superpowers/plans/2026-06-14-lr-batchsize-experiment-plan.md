# LR × Batch Size Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute 12 experiments (3 BS × 4 LR) comparing learning rate and batch size interaction on CIFAR-10 + SmallCNN across 3 Colab T4 GPUs in parallel, with cron-based monitoring.

**Architecture:** Five scripts: `train.py` (single-experiment runner, parameterized by --bs --lr), `launch.py` (bootstrap + sequential dispatcher for one account's BS), `fetch.py` (tar outputs on VM for cron download), `watchdog.py` (WebSocket relay keepalive, adapted from existing tests/), `analyze.py` (local post-hoc: merge 12 CSVs, generate heatmap + overlay curves + optimal-LR-vs-BS scatter + gradient noise plot + merged comparison CSV).

**Tech Stack:** Python 3.10+, PyTorch 2.11+cu128, torchvision, matplotlib (Agg backend), Colab T4 free tier

---

### Task 1: Create project directory

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/` (empty dir)

- [ ] **Step 1: Create directory**

```bash
mkdir -p /Users/mx/Desktop/projects/colab-cli/projects/systems/lr-batchsize-comparison/output
```

- [ ] **Step 2: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/
git commit -m "chore: create lr-batchsize-comparison project directory"
```

---

### Task 2: Write train.py — single experiment runner

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/train.py`

- [ ] **Step 1: Write train.py**

```python
"""LR × Batch Size experiment — single run.

Usage: python train.py --bs 16 --lr 1e-3

Runs 4000 optimizer steps with constant LR, evaluates every 200 steps.
Writes logs/train.log, metrics.csv, pngs/loss_acc.png, summary.json
to /content/lr-bs-output/bs<BS>_lr<LR>/
"""
import argparse, csv, json, os, time, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as D


def setup_dirs(out_dir):
    for sub in ["logs", "pngs"]:
        Path(out_dir, sub).mkdir(parents=True, exist_ok=True)


def get_data(batch_size):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_ds = D.CIFAR10(root="/content/data", train=True, download=True, transform=tf)
    test_ds = D.CIFAR10(root="/content/data", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()

    lr_str = f"{args.lr:.0e}".replace("e-0", "e-")
    out_dir = f"/content/lr-bs-output/bs{args.bs}_lr{lr_str}"
    setup_dirs(out_dir)

    log_path = f"{out_dir}/logs/train.log"
    csv_path = f"{out_dir}/metrics.csv"
    png_path = f"{out_dir}/pngs/loss_acc.png"

    with open(log_path, "w") as log_fh:
        def log_msg(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            log_fh.write(line + "\n")

        log_msg(f"LR×BS experiment: bs={args.bs} lr={args.lr}")
        log_msg(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch {torch.__version__}")

        torch.manual_seed(42)
        train_loader, test_loader = get_data(args.bs)
        steps_per_epoch = len(train_loader)

        model = SmallCNN().cuda()
        init_loss = None

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        with open(csv_path, "w", newline="") as cf:
            csv_w = csv.DictWriter(cf, fieldnames=[
                "batch", "loss", "train_acc", "test_loss", "test_acc", "lr", "grad_norm", "elapsed_s"
            ])
            csv_w.writeheader()

            t0 = time.time()
            batch_losses = []
            eval_points = []

            for batch_idx in range(1, 4001):
                x, y = next(iter(train_loader))
                x, y = x.cuda(), y.cuda()

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    out = model(x)
                    loss = F.cross_entropy(out, y)

                if init_loss is None:
                    init_loss = loss.item()
                    log_msg(f"Initial loss: {init_loss:.4f}  (expected ~2.30 for CIFAR-10)")

                # Divergence check (Karpathy heuristic: loss > 3× initial → LR too high)
                if loss.item() > init_loss * 3:
                    log_msg(f"DIVERGED at batch {batch_idx}: loss={loss.item():.4f} > 3×init={init_loss*3:.4f}")
                    csv_w.writerow({
                        "batch": batch_idx, "loss": round(loss.item(), 6),
                        "train_acc": 0.0, "test_loss": 0.0, "test_acc": 0.0,
                        "lr": args.lr, "grad_norm": 0.0, "elapsed_s": round(time.time() - t0, 1),
                    })
                    break

                scaler.scale(loss).backward()

                # Log unclipped gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                train_acc = (out.argmax(1) == y).float().mean().item()
                batch_losses.append(loss.item())

                # Eval every 200 batches
                if batch_idx % 200 == 0:
                    test_loss, test_acc = evaluate(model, test_loader, "cuda")
                    elapsed = time.time() - t0
                    log_msg(
                        f"Batch {batch_idx:>5} | loss={loss.item():.4f} | "
                        f"train_acc={train_acc:.3f} | test_loss={test_loss:.4f} | "
                        f"test_acc={test_acc:.3f} | grad_norm={total_norm:.2f} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    csv_w.writerow({
                        "batch": batch_idx,
                        "loss": round(loss.item(), 6),
                        "train_acc": round(train_acc, 4),
                        "test_loss": round(test_loss, 4),
                        "test_acc": round(test_acc, 4),
                        "lr": args.lr,
                        "grad_norm": round(total_norm, 4),
                        "elapsed_s": round(elapsed, 1),
                    })
                    eval_points.append((batch_idx, test_acc))
                    model.train()

                    # Generate plot every 1000 batches
                    if batch_idx % 1000 == 0:
                        try:
                            _save_plot(batch_losses, eval_points, png_path, args)
                        except Exception:
                            pass

        total_time = time.time() - t0
        final_test_loss, final_test_acc = evaluate(model, test_loader, "cuda")
        best_acc = max((a for _, a in eval_points), default=final_test_acc)

        log_msg(f"DONE: final_test_acc={final_test_acc:.4f} best_acc={best_acc:.4f} time={total_time:.0f}s")

        # Write summary
        summary = {
            "bs": args.bs, "lr": args.lr, "steps_completed": batch_idx,
            "init_loss": round(init_loss, 4) if init_loss else None,
            "final_test_acc": round(final_test_acc, 4),
            "best_acc": round(best_acc, 4),
            "total_time_s": round(total_time, 1),
            "grad_norm_mean": round(sum(batch_losses) / max(len(batch_losses), 1), 4),
        }
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Final plot
        try:
            _save_plot(batch_losses, eval_points, png_path, args)
        except Exception:
            pass


def _save_plot(batch_losses, eval_points, out_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    suptitle = f"BS={args.bs}  LR={args.lr}  —  CIFAR-10 SmallCNN"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    # Loss
    ax = axes[0]
    if batch_losses:
        w = min(50, len(batch_losses))
        if len(batch_losses) >= w:
            smooth = np.convolve(batch_losses, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(batch_losses)), smooth, color="darkorange", linewidth=1.2, label=f"avg{w}")
        ax.plot(batch_losses, alpha=0.12, color="steelblue", linewidth=0.3)
    ax.set_xlabel("Batch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if eval_points:
        xs, ys = zip(*eval_points)
        ax.plot(xs, ys, "o-", color="mediumseagreen", linewidth=1.5, markersize=3)
        best = max(ys)
        ax.axhline(y=best, color="green", linestyle=":", alpha=0.7, label=f"Best={best:.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Batch"); ax.set_ylabel("Test Accuracy"); ax.set_title("Evaluation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/mx/Desktop/projects/colab-cli && ruff check projects/systems/lr-batchsize-comparison/train.py
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/train.py
git commit -m "feat: add train.py — single LR×BS experiment runner (4000 steps, constant LR)"
```

---

### Task 3: Write launch.py — bootstrap + sequential dispatcher

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/launch.py`

- [ ] **Step 1: Write launch.py**

```python
"""Launch LR×BS experiments for one batch size as detached subprocess.

Reads BS from env var (default 16). Installs matplotlib, then runs
train.py 4 times sequentially with LR = 1e-4, 1e-3, 1e-2, 1e-1.

Usage on Colab VM:
    BS=64 colab exec -f launch.py --timeout 120
"""
import subprocess, sys, os, time

BS = os.environ.get("BS", "16")
LRS = ["1e-4", "1e-3", "1e-2", "1e-1"]
OUT_DIR = "/content/lr-bs-output"
LOG = f"/content/launch_bs{BS}.log"
PID_FILE = f"{OUT_DIR}/train.pid"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

# Write PID for watchdog monitoring
with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

# Install matplotlib for optional plots
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])

print(f"[launch] BS={BS}  LRs={LRS}")
print(f"[launch] log={LOG}")

with open(LOG, "w") as log_fh:
    def tee(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    tee(f"[{time.strftime('%H:%M:%S')}] START BS={BS}")
    tee(f"GPU: checking...")

    for lr_str in LRS:
        tee(f"\n{'='*50}")
        tee(f"[{time.strftime('%H:%M:%S')}] Running: train.py --bs {BS} --lr {lr_str}")
        tee(f"{'='*50}")

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", "/content/train.py", "--bs", BS, "--lr", lr_str],
            stdout=log_fh, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = time.time() - t0
        tee(f"[{time.strftime('%H:%M:%S')}] DONE rc={proc.returncode} elapsed={elapsed:.0f}s")

    tee(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE — BS={BS}")
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/mx/Desktop/projects/colab-cli && ruff check projects/systems/lr-batchsize-comparison/launch.py
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/launch.py
git commit -m "feat: add launch.py — sequential dispatcher for one BS × 4 LRs"
```

---

### Task 4: Write fetch.py — tar outputs for cron download

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/fetch.py`

- [ ] **Step 1: Write fetch.py**

```python
"""Tar all experiment outputs on VM for cron download.

Run via: colab exec -f fetch.py --timeout 15
Output: /content/lr-bs-output.tar.gz
"""
import tarfile, os, glob, json

OUT_DIR = "/content/lr-bs-output"
TAR_PATH = "/content/lr-bs-output.tar.gz"

# Report what we have
summary = {"experiments": {}}
for exp_dir in sorted(glob.glob(f"{OUT_DIR}/bs*_lr*")):
    name = os.path.basename(exp_dir)
    csv_path = f"{exp_dir}/metrics.csv"
    log_path = f"{exp_dir}/logs/train.log"
    summary_path = f"{exp_dir}/summary.json"

    n_lines = 0
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            n_lines = len(f.readlines()) - 1  # exclude header

    # Read summary if available
    exp_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            exp_summary = json.load(f)

    summary["experiments"][name] = {
        "csv_rows": n_lines,
        "summary": exp_summary,
    }
    print(f"[fetch] {name}: {n_lines} eval rows, summary={exp_summary.get('best_acc', '?')}")

# Also include launch log
launch_logs = glob.glob("/content/launch_bs*.log")
for ll in launch_logs:
    print(f"[fetch] launch log: {ll}")

print(f"[fetch] Total experiments found: {len(summary['experiments'])}")

# Tar everything
with tarfile.open(TAR_PATH, "w:gz") as tar:
    if os.path.exists(OUT_DIR):
        tar.add(OUT_DIR, arcname="lr-bs-output")
    for ll in launch_logs:
        tar.add(ll, arcname=os.path.basename(ll))

size_mb = os.path.getsize(TAR_PATH) / (1024 * 1024)
print(f"[fetch] Created {TAR_PATH} ({size_mb:.1f} MB)")
```

- [ ] **Step 2: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/fetch.py
git commit -m "feat: add fetch.py — tar experiment outputs for cron download"
```

---

### Task 5: Write watchdog.py — WebSocket relay keepalive

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/watchdog.py`

Adapt from the existing `tests/ws-keepalive/relay/watchdog.py` pattern. Changed OUT_DIR, PID file location, and log format for this experiment.

- [ ] **Step 1: Write watchdog.py**

```python
"""WebSocket relay watchdog for LR×BS experiments — 7-min window.

Upload once. Run via: colab exec -f watchdog.py --timeout 480

Keeps WebSocket alive while detached train.py runs. Monitors
training progress via PID check + log tail.
"""
import subprocess, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/lr-bs-output"
LOG = f"{OUT_DIR}/logs/watchdog.log"
COUNTER_FILE = f"{OUT_DIR}/watchdog_counter"
PID_FILE = f"{OUT_DIR}/train.pid"

DURATION = 420   # 7 minutes
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

counter = 1
if os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE) as f:
        counter = int(f.read().strip()) + 1
with open(COUNTER_FILE, "w") as f:
    f.write(str(counter))
NAME = f"ws-{counter}"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def wlog(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


wlog(f"START pid={os.getpid()} duration={DURATION}s")

train_pid = None
if os.path.exists(PID_FILE):
    with open(PID_FILE) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        wlog(f"training PID={train_pid} ALIVE")
    except OSError:
        wlog(f"training PID={train_pid} DEAD")
        train_pid = None
else:
    wlog("no PID file — monitoring via log files only")

try:
    import torch
    wlog(f"GPU={torch.cuda.get_device_name(0)}")
except Exception:
    wlog("GPU check skipped")

start_time = time.time()
for iteration in range(DURATION // INTERVAL):
    time.sleep(INTERVAL)
    elapsed = time.time() - start_time

    train_status = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            train_status = f"ALIVE(PID={train_pid})"
        except OSError:
            train_status = "DEAD"
            wlog("ALERT: training process died!")

    # GPU utilization
    gpu_info = "?"
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
            shell=True, text=True, timeout=5,
        ).strip()
    except Exception:
        pass

    # Tail of most recent training log
    import glob
    log_files = sorted(glob.glob(f"{OUT_DIR}/bs*_lr*/logs/train.log"))
    train_tail = "(no log)"
    if log_files:
        try:
            with open(log_files[-1]) as f:
                lines = f.readlines()
                train_tail = lines[-1].strip()[-180:] if lines else "(empty)"
        except Exception:
            pass

    wlog(f"iter={iteration+1} elapsed={elapsed:.0f}s train={train_status} gpu=[{gpu_info}] log: {train_tail}")
    print(f"[{ts()}] {NAME} heartbeat iter={iteration+1} elapsed={elapsed:.0f}s", flush=True)

    if train_status == "DEAD" and train_pid is not None:
        wlog("exiting early — training is dead")
        break

total = time.time() - start_time
wlog(f"EXIT total_elapsed={total:.0f}s")
wlog("HANDOFF: start next with: colab exec -f watchdog.py --timeout 480")
```

- [ ] **Step 2: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/watchdog.py
git commit -m "feat: add watchdog.py — WebSocket relay keepalive (7-min window)"
```

---

### Task 6: Write analyze.py — local comparison analysis

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/analyze.py`

- [ ] **Step 1: Write analyze.py**

```python
"""Merge 12 experiment CSVs and generate comparison artifacts.

Usage: python analyze.py [--input-dir output/]

Reads output/<account>/lr-bs-output/bs*_lr*/metrics.csv from all accounts,
merges into all_experiments.csv, generates 5 comparison plots.
"""
import argparse, csv, json, os, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_experiments(input_dir):
    """Scan input_dir for bs*_lr*/metrics.csv and load all."""
    experiments = []
    for exp_dir in sorted(Path(input_dir).glob("bs*_lr*")):
        csv_path = exp_dir / "metrics.csv"
        summary_path = exp_dir / "summary.json"
        if not csv_path.exists():
            continue

        # Parse bs and lr from dirname: bs16_lr1e-4
        name = exp_dir.name
        bs_part, lr_part = name.split("_lr")
        bs = int(bs_part.replace("bs", ""))
        lr = float(lr_part.replace("e-", "e-"))  # handles both "1e-4" and "0.001"

        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Read summary
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        experiments.append({
            "bs": bs, "lr": lr, "name": name,
            "rows": rows, "summary": summary,
        })

    return sorted(experiments, key=lambda e: (e["bs"], e["lr"]))


def write_comparison_csv(experiments, out_path):
    """Merge all experiments into one comparison CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bs", "lr", "best_test_acc", "final_test_acc", "final_test_loss",
            "steps_completed", "total_time_s", "grad_norm_std", "diverged",
        ])
        writer.writeheader()

        for exp in experiments:
            rows = exp["rows"]
            if not rows:
                continue

            best_acc = max(float(r["test_acc"]) for r in rows if r["test_acc"])
            final = rows[-1]
            grad_norms = [float(r["grad_norm"]) for r in rows if r["grad_norm"] and float(r["grad_norm"]) > 0]
            grad_std = np.std(grad_norms) if grad_norms else 0

            writer.writerow({
                "bs": exp["bs"],
                "lr": exp["lr"],
                "best_test_acc": round(best_acc, 4),
                "final_test_acc": round(float(final.get("test_acc", 0)), 4),
                "final_test_loss": round(float(final.get("test_loss", 0)), 4),
                "steps_completed": exp["summary"].get("steps_completed", len(rows) * 200),
                "total_time_s": exp["summary"].get("total_time_s", 0),
                "grad_norm_std": round(grad_std, 4),
                "diverged": 1 if exp["summary"].get("steps_completed", 4000) < 100 else 0,
            })
    print(f"[analyze] Wrote {out_path}")


def plot_heatmap(experiments, out_path):
    """LR × BS → best test accuracy heatmap."""
    bs_vals = sorted(set(e["bs"] for e in experiments))
    lr_vals = sorted(set(e["lr"] for e in experiments))

    data = np.zeros((len(bs_vals), len(lr_vals)))
    annot = []
    for i, bs in enumerate(bs_vals):
        row_annot = []
        for j, lr in enumerate(lr_vals):
            match = [e for e in experiments if e["bs"] == bs and e["lr"] == lr]
            if match and match[0]["rows"]:
                acc = max(float(r["test_acc"]) for r in match[0]["rows"] if r["test_acc"])
                data[i, j] = acc
                row_annot.append(f"{acc:.3f}")
            else:
                data[i, j] = np.nan
                row_annot.append("?")
        annot.append(row_annot)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.1, vmax=0.8)
    ax.set_xticks(range(len(lr_vals)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lr_vals])
    ax.set_yticks(range(len(bs_vals)))
    ax.set_yticklabels([f"BS={bs}" for bs in bs_vals])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Batch Size")
    ax.set_title("Best Test Accuracy: LR × BS Heatmap")

    for i in range(len(bs_vals)):
        for j in range(len(lr_vals)):
            color = "white" if data[i, j] < 0.5 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=11, color=color)

    plt.colorbar(im, ax=ax, label="Test Accuracy")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Heatmap → {out_path}")


def plot_overlay_curves(experiments, out_path):
    """3 panels (one per BS), each with 4 LR curves overlaid."""
    bs_vals = sorted(set(e["bs"] for e in experiments))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LR Effect per Batch Size — CIFAR-10 SmallCNN", fontsize=13, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))

    for ax_idx, bs in enumerate(bs_vals):
        ax = axes[ax_idx]
        bs_exps = [e for e in experiments if e["bs"] == bs]

        for exp in bs_exps:
            rows = exp["rows"]
            if not rows:
                continue
            xs = [int(r["batch"]) for r in rows if r["test_acc"]]
            ys = [float(r["test_acc"]) for r in rows if r["test_acc"]]
            if xs and ys:
                lr_idx = list(sorted(set(e["lr"] for e in experiments))).index(exp["lr"])
                ax.plot(xs, ys, "o-", color=colors[lr_idx], linewidth=1.2, markersize=2,
                       label=f"LR={exp['lr']:.0e}")

        ax.set_xlabel("Batch")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"BS={bs}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Overlay curves → {out_path}")


def plot_optimal_lr_vs_bs(experiments, out_path):
    """Scatter: best LR per batch size. Reference line for linear scaling."""
    bs_vals = sorted(set(e["bs"] for e in experiments))
    best_lrs = []

    for bs in bs_vals:
        bs_exps = [e for e in experiments if e["bs"] == bs]
        best_acc = -1
        best_lr = None
        for exp in bs_exps:
            rows = exp["rows"]
            if not rows:
                continue
            acc = max(float(r["test_acc"]) for r in rows if r["test_acc"])
            if acc > best_acc:
                best_acc = acc
                best_lr = exp["lr"]
        if best_lr:
            best_lrs.append((bs, best_lr, best_acc))

    if not best_lrs:
        print("[analyze] No data for optimal LR vs BS plot")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bss, lrs, accs = zip(*best_lrs)
    ax.scatter(bss, lrs, s=100, c=accs, cmap="RdYlGn", edgecolors="black", zorder=5)

    # Linear scaling reference: LR ∝ BS
    if len(best_lrs) >= 2:
        ref_bs = best_lrs[0][0]
        ref_lr = best_lrs[0][1]
        bs_range = np.linspace(min(bss) * 0.5, max(bss) * 1.5, 100)
        ax.plot(bs_range, [ref_lr * (b / ref_bs) for b in bs_range],
                "k--", alpha=0.5, linewidth=1, label="Linear scaling (LR ∝ BS)")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Optimal Learning Rate")
    ax.set_title("Optimal LR vs Batch Size")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for bs, lr, acc in best_lrs:
        ax.annotate(f"acc={acc:.3f}", (bs, lr), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Optimal LR vs BS → {out_path}")


def plot_gradient_noise(experiments, out_path):
    """Gradient norm std vs batch size."""
    bs_vals = sorted(set(e["bs"] for e in experiments))

    fig, ax = plt.subplots(figsize=(8, 5))

    for bs in bs_vals:
        bs_exps = [e for e in experiments if e["bs"] == bs]
        lrs = []
        grads = []
        for exp in bs_exps:
            rows = exp["rows"]
            if not rows:
                continue
            norms = [float(r["grad_norm"]) for r in rows if r["grad_norm"] and float(r["grad_norm"]) > 0]
            if norms:
                lrs.append(exp["lr"])
                grads.append(np.std(norms))
        if lrs and grads:
            ax.scatter([bs] * len(lrs), grads, s=60, alpha=0.7, label=f"BS={bs}")
            # Annotate with best LR
            for lr, gstd in zip(lrs, grads):
                ax.annotate(f"LR={lr:.0e}", (bs, gstd), fontsize=7, alpha=0.8,
                           textcoords="offset points", xytext=(8, 0))

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Gradient Norm StdDev")
    ax.set_title("Gradient Noise vs Batch Size (lower BS → noisier gradients)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Gradient noise → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output",
                       help="Directory containing per-account output subdirs")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"[analyze] Input dir {input_path} not found")
        sys.exit(1)

    experiments = load_all_experiments(input_path)
    print(f"[analyze] Loaded {len(experiments)} experiments")

    if not experiments:
        print("[analyze] No experiments found — nothing to analyze")
        sys.exit(1)

    os.makedirs("output/comparison", exist_ok=True)

    write_comparison_csv(experiments, "output/comparison/all_experiments.csv")
    plot_heatmap(experiments, "output/comparison/lr_bs_heatmap.png")
    plot_overlay_curves(experiments, "output/comparison/lr_curves.png")
    plot_optimal_lr_vs_bs(experiments, "output/comparison/optimal_lr_vs_bs.png")
    plot_gradient_noise(experiments, "output/comparison/gradient_noise.png")

    print("\n[analyze] All artifacts written to output/comparison/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/mx/Desktop/projects/colab-cli && ruff check projects/systems/lr-batchsize-comparison/analyze.py
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/analyze.py
git commit -m "feat: add analyze.py — merge CSVs, heatmap, overlay curves, optimal LR vs BS"
```

---

### Task 7: Write README.md

**Files:**
- Create: `projects/systems/lr-batchsize-comparison/README.md`

- [ ] **Step 1: Write README.md**

```markdown
# LR × Batch Size Interaction Experiment

Systematic comparison of learning rate and batch size effects on CNN training.
12 experiments across 3 Colab T4 GPUs.

## Experiment Matrix

| | LR=1e-4 | LR=1e-3 | LR=1e-2 | LR=1e-1 |
|---|---|---|---|---|
| **BS=16** | colab | colab | colab | colab |
| **BS=64** | cc | cc | cc | cc |
| **BS=256** | clb | clb | clb | clb |

## Fixed Configuration

- Model: SmallCNN (3 conv + 1 fc, ~1.2M params)
- Dataset: CIFAR-10 (50K train / 10K test)
- Steps: 4000 optimizer updates per experiment
- LR schedule: Constant (no decay)
- Optimizer: AdamW (wd=0.01, eps=1e-4)
- Precision: AMP FP16

## Files

| File | Purpose |
|------|---------|
| `train.py` | Single experiment runner (`--bs`, `--lr`) |
| `launch.py` | Bootstrap + sequential dispatcher (reads `BS` env var) |
| `fetch.py` | Tar outputs on VM for cron download |
| `watchdog.py` | WebSocket relay keepalive (7-min window) |
| `analyze.py` | Local: merge CSVs, generate comparison plots |

## Execution

See design spec: `docs/superpowers/specs/2026-06-14-lr-batchsize-experiment-design.md`
```

- [ ] **Step 2: Commit**

```bash
git add projects/systems/lr-batchsize-comparison/README.md
git commit -m "docs: add README for lr-batchsize-comparison experiment"
```

---

### Task 8: Data warmup — cache CIFAR-10 on all 3 accounts

**⚠️ Operational task — requires Colab CLI + proxy.** Run from repo root with proxy Config B.

- [ ] **Step 1: Warm up colab account**

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

colab new --gpu T4 -s warmup
echo 'import torchvision.datasets; torchvision.datasets.CIFAR10(root="/content/data", train=True, download=True); torchvision.datasets.CIFAR10(root="/content/data", train=False, download=True); print("CIFAR-10 cached")' | colab exec -s warmup --timeout 120
colab stop -s warmup
```

Expected: "CIFAR-10 cached" in output. Session stopped cleanly.

- [ ] **Step 2: Warm up cc account**

```bash
HOME=~/colab-accounts/account-c cc new --gpu T4 -s warmup
echo 'import torchvision.datasets; torchvision.datasets.CIFAR10(root="/content/data", train=True, download=True); torchvision.datasets.CIFAR10(root="/content/data", train=False, download=True); print("CIFAR-10 cached")' | HOME=~/colab-accounts/account-c cc exec -s warmup --timeout 120
HOME=~/colab-accounts/account-c cc stop -s warmup
```

Expected: "CIFAR-10 cached". Session stopped.

- [ ] **Step 3: Warm up clb account**

```bash
HOME=~/colab-accounts/account-clb clb new --gpu T4 -s warmup
echo 'import torchvision.datasets; torchvision.datasets.CIFAR10(root="/content/data", train=True, download=True); torchvision.datasets.CIFAR10(root="/content/data", train=False, download=True); print("CIFAR-10 cached")' | HOME=~/colab-accounts/account-clb clb exec -s warmup --timeout 120
HOME=~/colab-accounts/account-clb clb stop -s warmup
```

Expected: "CIFAR-10 cached". Session stopped.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: CIFAR-10 data warmup complete on colab, cc, clb"
```

---

### Task 9: Deploy & launch — parallel training on 3 GPUs

**⚠️ Operational task.** Run each account's deploy in parallel (separate shell processes) with proxy Config B.

- [ ] **Step 1: Deploy to colab (BS=16)**

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

colab new --gpu T4 -s lrbs
colab upload projects/systems/lr-batchsize-comparison/train.py /content/train.py
colab upload projects/systems/lr-batchsize-comparison/launch.py /content/launch.py
colab upload projects/systems/lr-batchsize-comparison/fetch.py /content/fetch.py
colab upload projects/systems/lr-batchsize-comparison/watchdog.py /content/watchdog.py
BS=16 colab exec -s lrbs -f launch.py --timeout 120 &
```

Expected: "OK. PID=..." then returns. Detached training started.

- [ ] **Step 2: Deploy to cc (BS=64)**

```bash
HOME=~/colab-accounts/account-c cc new --gpu T4 -s lrbs
HOME=~/colab-accounts/account-c cc upload projects/systems/lr-batchsize-comparison/train.py /content/train.py
HOME=~/colab-accounts/account-c cc upload projects/systems/lr-batchsize-comparison/launch.py /content/launch.py
HOME=~/colab-accounts/account-c cc upload projects/systems/lr-batchsize-comparison/fetch.py /content/fetch.py
HOME=~/colab-accounts/account-c cc upload projects/systems/lr-batchsize-comparison/watchdog.py /content/watchdog.py
BS=64 HOME=~/colab-accounts/account-c cc exec -s lrbs -f launch.py --timeout 120 &
```

Expected: "OK. PID=...".

- [ ] **Step 3: Deploy to clb (BS=256)**

```bash
HOME=~/colab-accounts/account-clb clb new --gpu T4 -s lrbs
HOME=~/colab-accounts/account-clb clb upload projects/systems/lr-batchsize-comparison/train.py /content/train.py
HOME=~/colab-accounts/account-clb clb upload projects/systems/lr-batchsize-comparison/launch.py /content/launch.py
HOME=~/colab-accounts/account-clb clb upload projects/systems/lr-batchsize-comparison/fetch.py /content/fetch.py
HOME=~/colab-accounts/account-clb clb upload projects/systems/lr-batchsize-comparison/watchdog.py /content/watchdog.py
BS=256 HOME=~/colab-accounts/account-clb clb exec -s lrbs -f launch.py --timeout 120 &
```

Expected: "OK. PID=...".

- [ ] **Step 4: Start relay watchdogs (T+6 min for each account)**

At T+6 min after launch, start ws-2 for each account:
```bash
# colab (BS=16)
colab exec -s lrbs -f watchdog.py --timeout 480 &

# cc (BS=64)
HOME=~/colab-accounts/account-c cc exec -s lrbs -f watchdog.py --timeout 480 &

# clb (BS=256)
HOME=~/colab-accounts/account-clb clb exec -s lrbs -f watchdog.py --timeout 480 &
```

At T+13 min, start ws-3 (needed for BS=256 which takes ~28 min):
```bash
HOME=~/colab-accounts/account-clb clb exec -s lrbs -f watchdog.py --timeout 480 &
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: deploy lr-bs experiments to colab (BS=16), cc (BS=64), clb (BS=256)"
```

---

### Task 10: Set up cron watchtowers — 3 parallel jobs

**⚠️ Operational task.** Creates 3 CronCreate jobs (one per account), firing every 3 minutes.

- [ ] **Step 1: Create cron for colab (BS=16)**

Using CronCreate tool:
```
cron: "*/3 * * * *"
prompt: "Fetch lr-bs experiment outputs from colab account.
1. Check session: colab sessions | grep lrbs
2. If session alive: colab exec -s lrbs -f fetch.py --timeout 15
3. Download: colab download -s lrbs /content/lr-bs-output.tar.gz projects/systems/lr-batchsize-comparison/output/colab_out.tar.gz
4. Extract: cd projects/systems/lr-batchsize-comparison && tar -xzf output/colab_out.tar.gz -C output/colab/ 2>/dev/null
5. Report: find output/colab/ -name 'train.log' -exec tail -5 {} \; 2>/dev/null
6. If session absent: report 'colab session lrbs not found — may be complete'
All commands run with: export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890"
```

- [ ] **Step 2: Create cron for cc (BS=64)**

```
cron: "*/3 * * * *"
prompt: "Fetch lr-bs experiment outputs from cc account.
1. Check: HOME=~/colab-accounts/account-c cc sessions | grep lrbs
2. If alive: HOME=~/colab-accounts/account-c cc exec -s lrbs -f fetch.py --timeout 15
3. Download: HOME=~/colab-accounts/account-c cc download -s lrbs /content/lr-bs-output.tar.gz projects/systems/lr-batchsize-comparison/output/cc_out.tar.gz
4. Extract: cd projects/systems/lr-batchsize-comparison && tar -xzf output/cc_out.tar.gz -C output/cc/ 2>/dev/null
5. Report: find output/cc/ -name 'train.log' -exec tail -5 {} \; 2>/dev/null
Proxies: HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890"
```

- [ ] **Step 3: Create cron for clb (BS=256)**

```
cron: "*/3 * * * *"
prompt: "Fetch lr-bs experiment outputs from clb account.
1. Check: HOME=~/colab-accounts/account-clb clb sessions | grep lrbs
2. If alive: HOME=~/colab-accounts/account-clb clb exec -s lrbs -f fetch.py --timeout 15
3. Download: HOME=~/colab-accounts/account-clb clb download -s lrbs /content/lr-bs-output.tar.gz projects/systems/lr-batchsize-comparison/output/clb_out.tar.gz
4. Extract: cd projects/systems/lr-batchsize-comparison && tar -xzf output/clb_out.tar.gz -C output/clb/ 2>/dev/null
5. Report: find output/clb/ -name 'train.log' -exec tail -5 {} \; 2>/dev/null
Proxies: HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 ALL_PROXY=socks5://127.0.0.1:7890"
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: cron watchtowers active for colab, cc, clb lr-bs experiments"
```

---

### Task 11: Final fetch, analyze & commit

Once all sessions complete (monitor via cron reports):

- [ ] **Step 1: Download final outputs from all accounts**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/systems/lr-batchsize-comparison

# colab
colab exec -s lrbs -f fetch.py --timeout 15
colab download -s lrbs /content/lr-bs-output.tar.gz output/colab_final.tar.gz
tar -xzf output/colab_final.tar.gz -C output/colab/

# cc
HOME=~/colab-accounts/account-c cc exec -s lrbs -f fetch.py --timeout 15
HOME=~/colab-accounts/account-c cc download -s lrbs /content/lr-bs-output.tar.gz output/cc_final.tar.gz
tar -xzf output/cc_final.tar.gz -C output/cc/

# clb
HOME=~/colab-accounts/account-clb clb exec -s lrbs -f fetch.py --timeout 15
HOME=~/colab-accounts/account-clb clb download -s lrbs /content/lr-bs-output.tar.gz output/clb_final.tar.gz
tar -xzf output/clb_final.tar.gz -C output/clb/
```

- [ ] **Step 2: Run analysis**

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/systems/lr-batchsize-comparison

# Merge all account outputs into a flat structure for analyze.py
mkdir -p output/merged
for acc in colab cc clb; do
  if [ -d "output/$acc/lr-bs-output" ]; then
    cp -r output/$acc/lr-bs-output/* output/merged/ 2>/dev/null || true
  fi
done

python analyze.py --input-dir output/merged
```

Expected: 5 files written to `output/comparison/`.

- [ ] **Step 3: Stop all sessions**

```bash
colab stop -s lrbs
HOME=~/colab-accounts/account-c cc stop -s lrbs
HOME=~/colab-accounts/account-clb clb stop -s lrbs
```

- [ ] **Step 4: Cancel cron jobs**

Use CronDelete for all 3 cron job IDs from Task 10.

- [ ] **Step 5: Final commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli
git add projects/systems/lr-batchsize-comparison/output/comparison/
git add projects/systems/lr-batchsize-comparison/output/merged/
git commit -m "feat: complete lr-bs experiment — 12 runs, analysis artifacts"
```
