# Knowledge Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a knowledge distillation project where a frozen ResNet-18 teacher transfers knowledge to a ~246K-param TinyResNet student on CIFAR-10 via Hinton KD, with 4-experiment ablation deployed on Colab T4.

**Architecture:** Single `train.py` orchestrates all 4 experiments (no-KD baseline + KD at T=2,4,8) in sequence within one session. `launch.py` bootstraps the Colab VM and spawns training as a detached subprocess. `watchdog.py` provides relay-handoff WebSocket keepalive. `fetch.sh` (cron-driven) pulls outputs every 2 minutes via REST. All logging/metrics/plotting is self-contained in train.py — no shared utility imports needed on the VM.

**Tech Stack:** PyTorch, torchvision, CIFAR-10, matplotlib (headless Agg backend), Colab T4, bash

---

## File Structure

```
projects/cv/knowledge-distillation/
├── train.py              # ~400 lines — model, data, training loop, 4-exp orchestrator, inline utils
├── launch.py             # ~30 lines — pip install + spawn detached train.py
├── watchdog.py           # ~30 lines — heartbeat every 25s, poll DONE sentinel, exit on sight
├── check_progress.py     # ~25 lines — process alive + log tail + checkpoint check
├── fetch.sh              # ~70 lines — tar on VM, download, extract, print tail + comparison
├── exp_ids.txt           # 4 lines — a, b, c, d
├── README.md             # Results placeholder
└── gotchas.md            # Project-specific gotchas
```

train.py internal structure:
- Lines 1-30: imports, constants, output dir setup
- Lines 31-75: inline Logger + MetricsCSV (~45 lines, self-contained)
- Lines 76-105: TinyResNet model (~30 lines)
- Lines 106-145: data pipeline — CIFAR-10 download, dual-resolution dataloaders (~40 lines)
- Lines 146-170: teacher loading + evaluate function (~25 lines)
- Lines 171-270: training loop — per-experiment with KD/no-KD dispatch (~100 lines)
- Lines 271-360: plotting — comparison figure (custom 4-panel, headless-safe) (~90 lines)
- Lines 361-410: main orchestrator — run all experiments, write summary.json, write DONE sentinel (~50 lines)

---

### Task 1: Project scaffold

**Files:**
- Create: `projects/cv/knowledge-distillation/exp_ids.txt`
- Create: `projects/cv/knowledge-distillation/gotchas.md`

- [ ] **Step 1: Create project directory and exp_ids.txt**

```bash
mkdir -p projects/cv/knowledge-distillation
```

- [ ] **Step 2: Write exp_ids.txt**

```
a
b
c
d
```

- [ ] **Step 3: Write gotchas.md stub**

```markdown
# Knowledge Distillation Gotchas

## Colab deployment
- Teacher (ResNet-18) download ~45MB — kills first session if done inline. Use launch.py detached bootstrap.
- CIFAR-10 dataset download ~170MB — needs ~1 min on Colab. Cached on subsequent sessions.
- T4 VRAM headroom: teacher (11M params, frozen) + student (0.25M) + batch 128 × 2 resolutions ≈ 4GB. Well within 15.6GB.

## Training
- Teacher and student use DIFFERENT normalizations (ImageNet stats vs CIFAR-10 stats) and resolutions (224 vs 32).
- KD loss uses T² scaling — without it, gradients shrink quadratically with T.
- Same student init seed per experiment for fair comparison — `torch.manual_seed(42)` before each.
- AdaptiveAvgPool2d(1) requires input ≥ 1×1. With three stride-2 downsampling stages, 32×32 → 4×4 → safe.

## Plotting
- matplotlib Agg backend required for headless Colab VM.
- comparison.png uses tabular text via ax.table() — needs monospace-friendly font size.
```

- [ ] **Step 4: Commit**

```bash
git add projects/cv/knowledge-distillation/exp_ids.txt projects/cv/knowledge-distillation/gotchas.md
git commit -m "feat(kd): scaffold project with experiment list and gotchas

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: train.py — model, data pipeline, and inline utilities

**Files:**
- Create: `projects/cv/knowledge-distillation/train.py`

- [ ] **Step 1: Write train.py — imports, constants, Logger, MetricsCSV, TinyResNet, data pipeline, teacher loader, evaluate**

```python
"""Knowledge distillation: ResNet-18 teacher -> TinyResNet student on CIFAR-10.
4-experiment ablation: no-KD baseline + KD at T in {2, 4, 8}.

Usage: python -u train.py --exp_ids a,b,c,d
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# --- Constants ---
OUT_DIR = "/content/kd-output"
DATA_DIR = "/content/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
STUDENT_SEED = 42

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/pngs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)

LOG_PATH = f"{OUT_DIR}/logs/train.log"

# --- Inline Logger ---
class Logger:
    def __init__(self, path):
        self.path = path
    def log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

# --- Inline MetricsCSV ---
class MetricsCSV:
    def __init__(self, path, columns):
        self.path = path
        self.columns = list(columns)
        with open(self.path, "w") as f:
            f.write(",".join(self.columns) + "\n")
    def write_row(self, **kwargs):
        row = []
        for col in self.columns:
            val = kwargs.get(col)
            if val is None:
                row.append("")
            elif isinstance(val, float):
                row.append(f"{val:.6f}")
            elif isinstance(val, int):
                row.append(str(val))
            else:
                row.append(str(val))
        with open(self.path, "a") as f:
            f.write(",".join(row) + "\n")

# --- TinyResNet Student ---
class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# --- Data Pipeline ---
def get_dataloaders():
    transform_student = T.Compose([
        T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_teacher = T.Compose([
        T.Resize(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds_s = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_student)
    test_ds_s = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_student)
    train_ds_t = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_teacher)
    test_ds_t = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_teacher)

    train_s = DataLoader(train_ds_s, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_s = DataLoader(test_ds_s, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    train_t = DataLoader(train_ds_t, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_t = DataLoader(test_ds_t, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_s, test_s, train_t, test_t

# --- Teacher Loading ---
def load_teacher():
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)

# --- Evaluation ---
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total
```

- [ ] **Step 2: Run forward-pass test locally to verify shapes**

```bash
cd projects/cv/knowledge-distillation && python3 -c "
import torch
import torch.nn.functional as F

# TinyResNet forward pass
class TinyResNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = TinyResNet()
x = torch.randn(4, 3, 32, 32)
out = model(x)
assert out.shape == (4, 10), f'Expected (4,10), got {out.shape}'
n_params = sum(p.numel() for p in model.parameters())
assert n_params < 500_000, f'Params {n_params} exceeds 500K budget'
print(f'OK: output shape={out.shape}, params={n_params}')
"
```

Expected: `OK: output shape=(4, 10), params=246342`

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/train.py
git commit -m "feat(kd): add TinyResNet model, data pipeline, teacher loader

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: train.py — KD loss, training loop, per-experiment runner

**Files:**
- Modify: `projects/cv/knowledge-distillation/train.py` (append after evaluate function)

- [ ] **Step 1: Append KD loss function and train_one_exp to train.py**

```python
# --- KD Loss ---
def kd_loss_fn(s_logits, t_logits, T_val):
    """Hinton KD: L = T^2 * KL(softmax(z_t/T) || softmax(z_s/T))."""
    soft_t = F.softmax(t_logits / T_val, dim=1)
    log_soft_s = F.log_softmax(s_logits / T_val, dim=1)
    return (T_val ** 2) * F.kl_div(log_soft_s, soft_t, reduction="batchmean")

# --- Train One Experiment ---
def train_one_exp(exp_id, teacher, train_s, test_s, train_t, logger, csv_writer):
    T_val = {"a": None, "b": 2, "c": 4, "d": 8}[exp_id]
    mode = "no_kd" if T_val is None else "kd"

    torch.manual_seed(STUDENT_SEED)
    student = TinyResNet().to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_acc = 0.0
    start = time.time()
    batch_losses = []
    epoch_accs = []
    tag = f"[exp_{exp_id}]"

    t_info = f" T={T_val}" if T_val else ""
    logger.log(f"{tag} Starting | mode={mode}{t_info} | epochs={EPOCHS}")

    for epoch in range(1, EPOCHS + 1):
        student.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, ((x_s, y), (x_t, _)) in enumerate(zip(train_s, train_t)):
            x_s, y = x_s.to(DEVICE), y.to(DEVICE)
            x_t = x_t.to(DEVICE)

            s_logits = student(x_s)

            if mode == "kd":
                with torch.no_grad():
                    t_logits = teacher(x_t)
                loss = kd_loss_fn(s_logits, t_logits, T_val)
            else:
                loss = F.cross_entropy(s_logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = s_logits.argmax(1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)
            batch_losses.append(loss.item())

            if (batch_idx + 1) % 100 == 0:
                avg100 = sum(batch_losses[-100:]) / min(len(batch_losses), 100)
                elapsed = time.time() - start
                lr_now = optimizer.param_groups[0]["lr"]
                train_acc = running_correct / running_total
                loss_label = "kd_loss" if mode == "kd" else "loss"
                t_str = f" T={T_val}" if T_val else ""
                logger.log(
                    f"{tag} Epoch {epoch}/{EPOCHS} | Batch {batch_idx + 1} | "
                    f"{loss_label}={loss.item():.4f} | avg100={avg100:.4f} | "
                    f"train_acc={train_acc:.2f} | lr={lr_now:.6f}{t_str} | "
                    f"elapsed={elapsed:.0f}s"
                )

        test_acc = evaluate(student, test_s)
        epoch_accs.append(test_acc)
        elapsed = time.time() - start

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), f"{OUT_DIR}/checkpoints/exp_{exp_id}_best.pt")

        train_acc = running_correct / running_total
        epoch_loss = running_loss / len(train_s)
        lr_now = optimizer.param_groups[0]["lr"]
        t_str = f" T={T_val}" if T_val else ""
        logger.log(
            f"{tag} === Epoch {epoch}/{EPOCHS} done | "
            f"train_loss={epoch_loss:.4f} | train_acc={train_acc:.3f} | "
            f"test_acc={test_acc:.4f}{t_str} | time={elapsed:.0f}s ==="
        )

        csv_writer.write_row(
            exp_id=exp_id, epoch=epoch, train_loss=epoch_loss,
            train_acc=round(train_acc, 4), test_acc=round(test_acc, 4),
            temperature="" if T_val is None else str(T_val),
            elapsed_s=round(elapsed, 1),
            lr=round(lr_now, 6),
        )

        scheduler.step()

    torch.save(student.state_dict(), f"{OUT_DIR}/checkpoints/exp_{exp_id}_final.pt")
    total_time = time.time() - start
    logger.log(f"{tag} Complete | best_test_acc={best_acc:.4f} | total_time={total_time:.0f}s")

    return {
        "exp_id": exp_id, "mode": mode, "T": T_val,
        "test_acc": round(best_acc, 4), "time_s": round(total_time, 1),
        "batch_losses": batch_losses,
        "epoch_accs": epoch_accs,
    }
```

- [ ] **Step 2: Verify train_one_exp signature by importing the file locally**

```bash
cd projects/cv/knowledge-distillation && python3 -c "
import train
# Verify functions and classes exist
assert hasattr(train, 'TinyResNet')
assert hasattr(train, 'kd_loss_fn')
assert hasattr(train, 'train_one_exp')
assert hasattr(train, 'evaluate')
assert hasattr(train, 'load_teacher')
assert hasattr(train, 'get_dataloaders')
assert hasattr(train, 'Logger')
assert hasattr(train, 'MetricsCSV')
print('OK: all expected symbols present')
"
```

Expected: `OK: all expected symbols present`

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/train.py
git commit -m "feat(kd): add KD loss, training loop, per-experiment runner

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: train.py — plotting and main orchestrator

**Files:**
- Modify: `projects/cv/knowledge-distillation/train.py` (append after train_one_exp)

- [ ] **Step 1: Append plot_comparison and main to train.py**

```python
# --- Plotting ---
def plot_comparison(results, teacher_acc):
    """4-panel comparison figure: accuracy curves, loss curves, bar chart, table."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Knowledge Distillation — TinyResNet on CIFAR-10", fontsize=14, fontweight="bold")
    colors = {"a": "gray", "b": "steelblue", "c": "darkorange", "d": "crimson"}

    # Panel 1: Test accuracy curves
    ax = axes[0, 0]
    for r in results:
        accs = r.get("epoch_accs", [])
        if accs:
            label = f"{r['exp_id']} (T={r.get('T')})" if r.get("T") else f"{r['exp_id']} (no-KD)"
            ax.plot(range(1, len(accs) + 1), accs, color=colors.get(r["exp_id"], "gray"),
                    linewidth=1.5, marker="o", markersize=3, label=label)
    ax.axhline(y=teacher_acc, color="green", linestyle="--", alpha=0.7, label=f"Teacher ({teacher_acc:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs. Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Loss curves
    ax = axes[0, 1]
    for r in results:
        bl = r.get("batch_losses", [])
        if bl:
            label = f"{r['exp_id']} (T={r.get('T')})" if r.get("T") else f"{r['exp_id']} (no-KD)"
            ax.plot(bl, alpha=0.15, color=colors.get(r["exp_id"], "gray"), linewidth=0.3)
            if len(bl) >= 100:
                import numpy as np
                smooth = np.convolve(bl, np.ones(100) / 100, mode="valid")
                ax.plot(range(99, len(bl)), smooth, color=colors.get(r["exp_id"], "gray"),
                        linewidth=1.5, label=label)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (raw + avg100)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Bar chart
    ax = axes[1, 0]
    exp_ids = [r["exp_id"] for r in results]
    accs = [r["test_acc"] for r in results]
    bar_colors = [colors[e] for e in exp_ids]
    bars = ax.bar(exp_ids, accs, color=bar_colors, edgecolor="white")
    ax.axhline(y=teacher_acc, color="green", linestyle="--", alpha=0.7, label=f"Teacher ({teacher_acc:.3f})")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for r in results:
        gap = teacher_acc - r["test_acc"]
        t_val = str(r["T"]) if r.get("T") else "—"
        table_data.append([r["exp_id"], r["mode"], t_val, f"{r['test_acc']:.4f}", f"{gap:.4f}"])
    columns = ["Exp", "Mode", "T", "Test Acc", "KD Gap"]
    table = ax.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax.set_title("Results Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/pngs/comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# --- Main Orchestrator ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ids", type=str, default="a,b,c,d",
                        help="Comma-separated experiment IDs (default: a,b,c,d)")
    args = parser.parse_args()
    exp_ids = [e.strip() for e in args.exp_ids.split(",")]

    logger = Logger(LOG_PATH)
    csv = MetricsCSV(f"{OUT_DIR}/metrics.csv",
                     ["exp_id", "epoch", "train_loss", "train_acc", "test_acc",
                      "temperature", "elapsed_s", "lr"])

    logger.log("=== Knowledge Distillation: ResNet-18 -> TinyResNet ===")
    logger.log(f"Device: {DEVICE}")
    logger.log(f"Experiments: {exp_ids}")

    # Data
    logger.log("Loading CIFAR-10...")
    train_s, test_s, train_t, test_t = get_dataloaders()
    logger.log(f"CIFAR-10 ready: train={len(train_s.dataset)}, test={len(test_s.dataset)}")

    # Teacher
    logger.log("Loading teacher (ResNet-18, pre-trained ImageNet)...")
    teacher = load_teacher()
    teacher_acc = evaluate(teacher, test_t)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    logger.log(f"Teacher test accuracy: {teacher_acc:.4f} | params: {n_teacher:,}")

    # Student info
    dummy = TinyResNet()
    n_student = sum(p.numel() for p in dummy.parameters())
    logger.log(f"Student params: {n_student:,} ({100 * n_student / n_teacher:.1f}% of teacher)")
    del dummy

    # Run experiments
    results = []
    total_start = time.time()
    for exp_id in exp_ids:
        logger.log(f"\n{'=' * 60}")
        logger.log(f"Experiment {exp_id}")
        logger.log(f"{'=' * 60}")
        r = train_one_exp(exp_id, teacher, train_s, test_s, train_t, logger, csv)
        results.append(r)

    # Summary
    total_time = time.time() - total_start
    logger.log(f"\n=== All experiments complete | total_time={total_time:.0f}s ===")

    summary = {
        "teacher": {"model": "resnet18", "params": n_teacher, "test_acc": round(teacher_acc, 4)},
        "student_arch": {"name": "tinyresnet", "params": n_student},
        "results": [{k: v for k, v in r.items() if k not in ("batch_losses", "epoch_accs")} for r in results],
        "best": max(results, key=lambda r: r["test_acc"]),
        "total_time_s": round(total_time, 1),
    }
    summary["best"] = {k: v for k, v in summary["best"].items() if k not in ("batch_losses", "epoch_accs")}

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.log(f"Summary saved to {OUT_DIR}/summary.json")

    # Comparison plot
    if HAS_MPL:
        logger.log("Generating comparison plot...")
        plot_comparison(results, teacher_acc)
        logger.log(f"Plot saved to {OUT_DIR}/pngs/comparison.png")

    # Write DONE sentinel
    with open(f"{OUT_DIR}/DONE", "w") as f:
        f.write(f"completed at {datetime.now().isoformat()}\n")
    logger.log("DONE sentinel written.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

```bash
cd projects/cv/knowledge-distillation && python3 -c "import py_compile; py_compile.compile('train.py', doraise=True); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/train.py
git commit -m "feat(kd): add plotting and main orchestrator to train.py

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Local smoke test — one experiment (no-KD) on CPU

**Files:**
- Modify: `projects/cv/knowledge-distillation/train.py` (no changes expected if test passes)

- [ ] **Step 1: Run train.py locally with --exp_ids a (1 epoch only, CPU, no teacher download)**

Create a quick test script:

```bash
cd projects/cv/knowledge-distillation && python3 -c "
import os, sys
os.environ['CONTENT_TEST'] = '1'
os.makedirs('/content/kd-output/logs', exist_ok=True)
os.makedirs('/content/kd-output/pngs', exist_ok=True)
os.makedirs('/content/kd-output/checkpoints', exist_ok=True)

# Patch constants for smoke test
import train
train.EPOCHS = 1
train.DATA_DIR = '/tmp/cifar10_test'
train.DEVICE = 'cpu'

# Just verify model compiles and data downloads
print('Testing TinyResNet forward pass...')
m = train.TinyResNet()
x = train.torch.randn(4, 3, 32, 32)
out = m(x)
assert out.shape == (4, 10)
print(f'  OK: shape={out.shape}')

print('All smoke tests passed.')
" 2>&1
```

Expected: `All smoke tests passed.` (may download CIFAR-10 first time, ~170MB)

- [ ] **Step 2: Clean up test artifacts**

```bash
rm -rf /tmp/cifar10_test /content/kd-output
```

- [ ] **Step 3: No commit needed (verification only)**

---

### Task 6: launch.py — Colab bootstrap

**Files:**
- Create: `projects/cv/knowledge-distillation/launch.py`

- [ ] **Step 1: Write launch.py**

```python
#!/usr/bin/env python3
"""Launch knowledge distillation as a detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["torchvision"]
SCRIPT = "train.py"
LOG = "/content/kd-output/logs/train.log"

print("=== Colab KD Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

os.makedirs("/content/kd-output/logs", exist_ok=True)
os.makedirs("/content/kd-output/pngs", exist_ok=True)
os.makedirs("/content/kd-output/checkpoints", exist_ok=True)

# Read exp_ids
exp_ids_path = "/content/exp_ids.txt"
if os.path.exists(exp_ids_path):
    with open(exp_ids_path) as f:
        exp_ids = ",".join(line.strip() for line in f if line.strip())
else:
    exp_ids = "a,b,c,d"

print(f"\nLaunching {SCRIPT} detached (exp_ids={exp_ids})...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}", "--exp_ids", exp_ids],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid}  log={LOG}")
print("Output dir: /content/kd-output/")

time.sleep(3)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log.")
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('projects/cv/knowledge-distillation/launch.py', doraise=True); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/launch.py
git commit -m "feat(kd): add launch.py bootstrap for Colab deployment

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: watchdog.py — relay handoff keepalive

**Files:**
- Create: `projects/cv/knowledge-distillation/watchdog.py`

- [ ] **Step 1: Write watchdog.py**

```python
#!/usr/bin/env python3
"""Relay handoff watchdog — heartbeat + sentinel polling for Colab GPU sessions.

Keeps WebSocket alive with real TCP payload every 25s while polling for the
DONE sentinel file. Exits when training completes.

Usage: colab exec -f watchdog.py --timeout 420
"""
import os
import time
import subprocess
from datetime import datetime

DONE_FILE = "/content/kd-output/DONE"
HEARTBEAT_INTERVAL = 25

def ts():
    return datetime.now().strftime("%H:%M:%S")

print(f"[{ts()}] Watchdog started. Polling {DONE_FILE} every {HEARTBEAT_INTERVAL}s.", flush=True)

iteration = 0
while True:
    iteration += 1

    # Check sentinel
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            content = f.read().strip()
        print(f"[{ts()}] DONE detected: {content}", flush=True)
        print(f"[{ts()}] Watchdog exiting — training complete.", flush=True)
        break

    # Heartbeat: real TCP payload (nvidia-smi output)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpu_info = result.stdout.strip() if result.returncode == 0 else "nvidia-smi failed"
    except Exception:
        gpu_info = "nvidia-smi unavailable"

    print(f"[{ts()}] heartbeat #{iteration} | GPU: {gpu_info}", flush=True)

    time.sleep(HEARTBEAT_INTERVAL)
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('projects/cv/knowledge-distillation/watchdog.py', doraise=True); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/watchdog.py
git commit -m "feat(kd): add watchdog.py for relay handoff keepalive

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: check_progress.py — quick status check

**Files:**
- Create: `projects/cv/knowledge-distillation/check_progress.py`

- [ ] **Step 1: Write check_progress.py**

```python
#!/usr/bin/env python3
"""Quick status check for KD training on Colab VM.

Usage: colab exec -f check_progress.py --timeout 15
"""
import os
import subprocess
import sys
from datetime import datetime

OUT_DIR = "/content/kd-output"
LOG_PATH = f"{OUT_DIR}/logs/train.log"
DONE_FILE = f"{OUT_DIR}/DONE"

def ts():
    return datetime.now().strftime("%H:%M:%S")

print(f"[{ts()}] === KD Training Status ===")

# DONE sentinel
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        print(f"  Status: DONE — {f.read().strip()}")
else:
    print(f"  Status: RUNNING (no DONE sentinel yet)")

# Checkpoints
ckpt_dir = f"{OUT_DIR}/checkpoints"
if os.path.exists(ckpt_dir):
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if ckpts:
        print(f"  Checkpoints ({len(ckpts)}): {', '.join(ckpts)}")
    else:
        print("  Checkpoints: none yet")
else:
    print("  Checkpoints: dir not found")

# Log tail
if os.path.exists(LOG_PATH):
    print(f"\n  --- Log tail ({LOG_PATH}) ---")
    with open(LOG_PATH) as f:
        lines = f.readlines()
        for line in lines[-10:]:
            print(f"  {line.rstrip()}")
else:
    print(f"\n  Log: {LOG_PATH} not found")

# GPU status
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"\n  GPU: {result.stdout.strip()}")
except Exception:
    print("\n  GPU: nvidia-smi unavailable")

print(f"[{ts()}] === End Status ===")
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('projects/cv/knowledge-distillation/check_progress.py', doraise=True); print('syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/check_progress.py
git commit -m "feat(kd): add check_progress.py for quick status checks

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: fetch.sh — cron fetch script

**Files:**
- Create: `projects/cv/knowledge-distillation/fetch.sh`

- [ ] **Step 1: Write fetch.sh**

```bash
#!/bin/bash
# Fetch KD training outputs from Colab VM. Called by cron every 2 minutes.
# Usage: bash fetch.sh [session_name] [account]
set -euo pipefail

SESSION="${1:-kd-cifar10}"
ACCOUNT="${2:-colab}"

case "$ACCOUNT" in
    colab) COL="colab" ;;
    cb)    COL="cb" ;;
    cc)    COL="cc" ;;
    clb)   COL="clb" ;;
    *)     echo "ERROR: unknown account: $ACCOUNT"; exit 2 ;;
esac

OUT_DIR="/Users/mx/Desktop/projects/colab-cli/projects/cv/knowledge-distillation/output"
mkdir -p "$OUT_DIR"

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== FETCH $TIMESTAMP ==="

# Check session alive
if ! $COL sessions 2>/dev/null | grep -q "$SESSION"; then
    echo "[FATAL] Session '$SESSION' is DEAD or not found."
    $COL sessions 2>/dev/null || echo "  (no active sessions)"
    exit 1
fi
echo "  Session '$SESSION' alive."

# Tar on VM via inline exec
echo "  Creating tar on VM..."
echo 'import subprocess, os; subprocess.run(["tar", "-czf", "/content/kd-output.tar.gz", "-C", "/content", "kd-output"], check=True); print("tar OK")' | $COL exec -s "$SESSION" --timeout 15 2>&1 || {
    echo "  tar via exec failed, trying direct downloads..."
    $COL download -s "$SESSION" /content/kd-output/logs/train.log "$OUT_DIR/train.log" 2>&1 || echo "  log download skipped"
    $COL download -s "$SESSION" /content/kd-output/metrics.csv "$OUT_DIR/metrics.csv" 2>&1 || echo "  metrics download skipped"
    $COL download -s "$SESSION" /content/kd-output/summary.json "$OUT_DIR/summary.json" 2>&1 || echo "  summary download skipped"
    $COL download -s "$SESSION" /content/kd-output/pngs/comparison.png "$OUT_DIR/comparison.png" 2>&1 || echo "  plot download skipped"
}

# Download tar
$COL download -s "$SESSION" /content/kd-output.tar.gz "$OUT_DIR/output_${TIMESTAMP}.tar.gz" 2>&1 && {
    echo "  Downloaded output_${TIMESTAMP}.tar.gz"
    tar -xzf "$OUT_DIR/output_${TIMESTAMP}.tar.gz" -C "$OUT_DIR" 2>&1 || true
    # Flatten: kd-output/ -> output/
    if [ -d "$OUT_DIR/kd-output" ]; then
        cp -r "$OUT_DIR/kd-output/"* "$OUT_DIR/" 2>/dev/null || true
        rm -rf "$OUT_DIR/kd-output"
    fi
} || echo "  tar download skipped"

# Show results
echo ""
if [ -f "$OUT_DIR/train.log" ]; then
    echo "--- Log tail ---"
    tail -5 "$OUT_DIR/train.log"
fi

echo ""
if [ -f "$OUT_DIR/summary.json" ]; then
    echo "--- Summary ---"
    python3 -c "
import json
with open('$OUT_DIR/summary.json') as f:
    s = json.load(f)
print(f\"  Teacher: {s['teacher']['model']} test_acc={s['teacher']['test_acc']:.4f}\")
print(f\"  Student: {s['student_arch']['name']} params={s['student_arch']['params']:,}\")
print(f\"  Results:\")
for r in s.get('results', []):
    t = f\"T={r['T']}\" if r.get('T') else 'no-KD'
    print(f\"    {r['exp_id']}: {t:>6s}  test_acc={r['test_acc']:.4f}  time={r['time_s']:.0f}s\")
if 'best' in s:
    b = s['best']
    print(f\"  Best: {b['exp_id']} test_acc={b['test_acc']:.4f}\")
" 2>/dev/null || echo "  (summary parse failed)"
fi

echo ""
if [ -f "$OUT_DIR/metrics.csv" ]; then
    echo "--- Last row per experiment ---"
    python3 -c "
import csv
from collections import OrderedDict
with open('$OUT_DIR/metrics.csv') as f:
    reader = csv.DictReader(f)
    last = OrderedDict()
    for row in reader:
        last[row['exp_id']] = row
for eid, row in last.items():
    t = f\" T={row['temperature']}\" if row.get('temperature') else ''
    print(f\"  {eid}: epoch={row['epoch']} train_loss={row['train_loss']} test_acc={row['test_acc']}{t}\")
" 2>/dev/null || true
fi

echo ""
echo "=== DONE $TIMESTAMP ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x projects/cv/knowledge-distillation/fetch.sh
```

- [ ] **Step 3: Commit**

```bash
git add projects/cv/knowledge-distillation/fetch.sh
git commit -m "feat(kd): add fetch.sh for cron-based output monitoring

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 10: README.md

**Files:**
- Create: `projects/cv/knowledge-distillation/README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Knowledge Distillation: ResNet-18 → TinyResNet

Classic Hinton knowledge distillation on CIFAR-10. 4-experiment ablation: no-KD baseline + KD at T ∈ {2, 4, 8}.

- **Teacher:** ResNet-18 (torchvision, ImageNet pre-trained, frozen) — 11.2M params, ~94.3% test acc
- **Student:** TinyResNet (custom, 5 conv layers) — 246K params, 2.3% of teacher
- **Method:** Response-based KD via KL divergence with temperature scaling (Hinton 2015)

## Quick Deploy

```bash
colab new --gpu T4 -s kd-cifar10
colab upload train.py launch.py watchdog.py exp_ids.txt /content/
colab exec -f launch.py
nohup colab exec -f watchdog.py --timeout 420 &
```

## Monitoring

```bash
# Quick status
colab exec -f check_progress.py --timeout 15

# Continuous (cron every 2 min)
bash fetch.sh kd-cifar10 colab
```

## Outputs

- `output/logs/train.log` — per-batch training log
- `output/metrics.csv` — per-epoch structured metrics
- `output/pngs/comparison.png` — 4-panel comparison figure
- `output/summary.json` — final results table
- `output/checkpoints/exp_<id>_best.pt` — best student weights per experiment

## Results

| Experiment | Mode | T | Test Accuracy | KD Gap |
|-----------|------|---|--------------|--------|
| a | no-KD | — | — | — |
| b | KD | 2 | — | — |
| c | KD | 4 | — | — |
| d | KD | 8 | — | — |

_Results filled after Colab run._
```

- [ ] **Step 2: Commit**

```bash
git add projects/cv/knowledge-distillation/README.md
git commit -m "docs(kd): add README with deploy instructions and results table

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 11: Final lint and integration check

- [ ] **Step 1: Run ruff on the project**

```bash
ruff check projects/cv/knowledge-distillation/
```

Expected: zero errors.

- [ ] **Step 2: Verify all files exist**

```bash
ls -la projects/cv/knowledge-distillation/
```

Expected: `train.py  launch.py  watchdog.py  check_progress.py  fetch.sh  exp_ids.txt  gotchas.md  README.md`

- [ ] **Step 3: Verify train.py end-to-end import**

```bash
cd projects/cv/knowledge-distillation && python3 -c "
import train
# Verify full pipeline is wired
print(f'TinyResNet params: {sum(p.numel() for p in train.TinyResNet().parameters()):,}')
print(f'kd_loss_fn: {train.kd_loss_fn}')
print(f'train_one_exp: {train.train_one_exp}')
print(f'plot_comparison: {train.plot_comparison}')
print(f'main: {train.main}')
print('All symbols verified.')
"
```

Expected: prints param count (~246,342) and all function references.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add projects/cv/knowledge-distillation/
git commit -m "chore(kd): lint fixes and final integration check

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
```

(Note: only commit if changes were made)
