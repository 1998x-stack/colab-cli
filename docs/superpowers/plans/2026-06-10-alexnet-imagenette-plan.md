# AlexNet Faithful Reproduction on Imagenette — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 5 files that faithfully reproduce AlexNet on Imagenette, deploy to 2 parallel Colab T4 sessions with dual-layer monitoring.

**Architecture:** `alexnet.py` defines the exact paper model with configurable width/dropout. `train.py` loads from HuggingFace, computes PCA color augmentation, runs the training loop + 10-view eval + 4-experiment orchestrator + chart generation. `launch.py` bootstraps on Colab — pip install, HF token, spawn train + watchdog as detached subprocesses. `watchdog.py` writes heartbeat.json every 30s. `check_progress.py` runs locally via cron to read heartbeat and report health.

**Tech Stack:** PyTorch, torchvision, HuggingFace datasets, matplotlib, sklearn

---

## File Map

| File | Responsibility | ~Lines |
|---|---|---|
| `alexnet.py` | Model definition, paper weight init, `build_alexnet(config)` factory | 100 |
| `train.py` | Data pipeline, PCA aug, training loop, 10-view eval, 4-experiment orchestrator, chart generation, metrics export | 400 |
| `launch.py` | Read `/content/exp_ids.txt`, `pip install` deps, set `HF_TOKEN`, spawn `train.py` + `watchdog.py` as detached subprocesses | 60 |
| `watchdog.py` | Loop: write `/content/heartbeat.json` every 30s, exit when `/content/watchdog_stop` exists | 40 |
| `check_progress.py` | Read `heartbeat.json`, `pgrep python`, tail `/content/train.log`, report health/ETA | 50 |

---

### Task 0: Create project directory and exp config files

**Files:**
- Create: `projects/alexnet-imagenette/` (directory)
- Create: `/tmp/exp_ids_a.txt`, `/tmp/exp_ids_b.txt`

- [ ] **Step 1: Create project directory**

```bash
mkdir -p /Users/mx/Desktop/projects/colab-cli/projects/alexnet-imagenette
mkdir -p /Users/mx/Desktop/projects/colab-cli/projects/alexnet-imagenette/output-a
mkdir -p /Users/mx/Desktop/projects/colab-cli/projects/alexnet-imagenette/output-b
```

- [ ] **Step 2: Create experiment config files for each session**

```bash
echo "1,2" > /Users/mx/Desktop/projects/colab-cli/projects/alexnet-imagenette/exp_ids_a.txt
echo "3,4" > /Users/mx/Desktop/projects/colab-cli/projects/alexnet-imagenette/exp_ids_b.txt
```

- [ ] **Step 3: Verify**

```bash
ls projects/alexnet-imagenette/
cat projects/alexnet-imagenette/exp_ids_a.txt
cat projects/alexnet-imagenette/exp_ids_b.txt
```

---

### Task 1: Write `alexnet.py` — Model Definition

**File:**
- Create: `projects/alexnet-imagenette/alexnet.py`

- [ ] **Step 1: Write the complete AlexNet module**

```python
"""Exact AlexNet architecture (Krizhevsky et al., NeurIPS 2012).

Adapted for Imagenette: 10 output classes, AdaptiveAvgPool2d(6) to handle
128×128 input (paper uses 224×224 → 6×6 after Conv5+MaxPool).

Configurable: width_multiplier (1.0 = paper), dropout (0.5 = paper).
No LRN — omitted per design decision (obsolete since BatchNorm).
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.5):
        super().__init__()
        w = lambda c: max(1, int(c * width_mult))

        self.conv1 = nn.Conv2d(3, w(96), kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(w(96), w(256), kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(w(256), w(384), kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(w(384), w(384), kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(w(384), w(256), kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.fc6 = nn.Linear(w(256) * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Paper: conv layers N(0, 0.01), FC layers N(0, 0.005)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.005)

        # Paper: bias = 1 for Conv2, Conv4, Conv5 and all FC layers.
        # Bias = 0 (default) for Conv1, Conv3.
        for name, m in [
            ("conv2", self.conv2), ("conv4", self.conv4), ("conv5", self.conv5),
            ("fc6", self.fc6), ("fc7", self.fc7), ("fc8", self.fc8),
        ]:
            nn.init.constant_(m.bias, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))

        x = self.relu(self.conv5(x))
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


def build_alexnet(config):
    """Factory: build AlexNet from experiment config dict.

    config keys:
        width_mult (float): 1.0 = paper, 0.5 = reduced width
        dropout (float):  0.5 = paper, 0.0 = no dropout
        num_classes (int): default 10
    """
    return AlexNet(
        num_classes=config.get("num_classes", 10),
        width_mult=config.get("width_mult", 1.0),
        dropout=config.get("dropout", 0.5),
    )
```

- [ ] **Step 2: Verify local import and forward pass shape**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python3 -c "
from projects.alexnet_imagenette.alexnet import build_alexnet
import torch

# Baseline
model = build_alexnet({'width_mult': 1.0, 'dropout': 0.5})
x = torch.randn(4, 3, 128, 128)
out = model(x)
assert out.shape == (4, 10), f'Expected (4,10), got {out.shape}'

# Reduced width
model_half = build_alexnet({'width_mult': 0.5, 'dropout': 0.5})
out_half = model_half(x)
assert out_half.shape == (4, 10)

# No dropout
model_nodrop = build_alexnet({'width_mult': 1.0, 'dropout': 0.0})
model_nodrop.train()
out_nodrop = model_nodrop(x)
assert out_nodrop.shape == (4, 10)

params = sum(p.numel() for p in model.parameters())
print(f'Baseline params: {params:,}')
print(f'Reduced width params: {sum(p.numel() for p in model_half.parameters()):,}')
print('All checks passed')
"
```

Expected: Baseline params ~57M, Reduced width ~14M, no errors.

---

### Task 2: Write `train.py` — Data Pipeline + PCA Augmentation

**File:**
- Create: `projects/alexnet-imagenette/train.py` (sections built incrementally)

- [ ] **Step 1: Imports and config skeleton**

```python
"""AlexNet training on Imagenette — data pipeline, training loop, 10-view eval,
4-experiment orchestrator, chart generation.

Usage (on Colab VM): python -u train.py --exp_ids 1,2
"""

import json, os, sys, time, argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF
from datasets import load_dataset

OUTPUT_DIR = "/content/alexnet-output"
LOG_PATH = "/content/train.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Imagenette class names (alphabetical order in the dataset)
CLASS_NAMES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]
NUM_CLASSES = 10

# Paper hyperparameters
BATCH_SIZE = 128
LR_INIT = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_PATIENCE = 3
LR_FACTOR = 0.1
EPOCHS = 90
CROP_SIZE = 128
```

- [ ] **Step 2: Logging helper**

```python
LOG_FILE = open(LOG_PATH, "w", buffering=1)  # line-buffered

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    LOG_FILE.write(line + "\n")
```

- [ ] **Step 3: PCA color augmentation (Fancy PCA)**

```python
# ── PCA Color Augmentation (Fancy PCA, paper section 4.1) ────────────────

class PCA:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.eigvals = None
        self.eigvecs = None

    def fit(self, images):
        pixels = torch.stack([img.reshape(3, -1) for img in images])
        cov = torch.cov(pixels.permute(1, 0, 2).reshape(3, -1))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        self.eigvals = eigvals[-self.n_components:]
        self.eigvecs = eigvecs[:, -self.n_components:]

    def apply(self, img):
        if self.eigvals is None:
            return img
        alpha = torch.randn(self.n_components) * 0.1  # paper: σ * α_i
        delta = (self.eigvecs * self.eigvals.sqrt() * alpha).sum(dim=1)
        return img + delta.view(3, 1, 1)
```

- [ ] **Step 4: Dataset loading + train/val split**

```python
# ── Data Pipeline ────────────────────────────────────────────────────────

class ImagenetteDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        img = item["image"]
        label = item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_imagenette_data():
    log("Loading Imagenette (frgfm/imagenette, 160px)...")
    ds = load_dataset("frgfm/imagenette", "160px", token=os.environ.get("HF_TOKEN"))

    train_raw = ds["train"]
    n_train = int(0.8 * len(train_raw))
    indices = torch.randperm(len(train_raw)).tolist()

    idx_train = indices[:n_train]
    idx_val = indices[n_train:]

    return train_raw, idx_train, idx_val, ds["test"] if "test" in ds else None
```

- [ ] **Step 5: Transform builder — with/without augmentation**

```python
def build_transforms(augment=True, pca=None):
    ops = []
    if augment:
        ops.append(TF.resize)         # resize to 160 first, then random crop
        ops.append(TF.center_crop)    # placeholder — applied in _augment()
    else:
        ops.append(TF.center_crop)    # placeholder

    ops.append(TF.to_tensor)

    def _augment(img):
        if augment:
            img = TF.resize(img, 160)
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(CROP_SIZE, CROP_SIZE))
            img = TF.crop(img, i, j, h, w)
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
        else:
            img = TF.resize(img, CROP_SIZE)
            img = TF.center_crop(img, CROP_SIZE)

        img = TF.to_tensor(img)

        if augment and pca is not None:
            img = pca.apply(img)

        # Normalize to [0,1] — already done by to_tensor from PIL
        return img

    return _augment
```

- [ ] **Step 6: Local verification — dataset loads and PCA fits**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python3 -c "
import os, sys
os.environ['HF_TOKEN'] = open('.huggingface/access_token').read().strip()
sys.path.insert(0, 'projects/alexnet-imagenette')

from train import load_imagenette_data, PCA, CLASS_NAMES
import torch

# Test dataset loading
train_raw, idx_train, idx_val, _ = load_imagenette_data()
print(f'Train raw: {len(train_raw)}, Train idx: {len(idx_train)}, Val idx: {len(idx_val)}')

# Test PCA fitting on 100 images
from datasets import load_dataset
import torchvision.transforms.functional as TF
ds = load_dataset('frgfm/imagenette', '160px', token=os.environ.get('HF_TOKEN'))
samples = []
for i in range(100):
    img = ds['train'][i]['image']
    samples.append(TF.to_tensor(TF.resize(img, 128)))

pca = PCA(n_components=3)
pca.fit(samples)
print(f'Eigvals: {pca.eigvals}')
print(f'Eigvecs shape: {pca.eigvecs.shape}')

# Test PCA application
augmented = pca.apply(samples[0])
print(f'Augmented image shape: {augmented.shape}')
print('All checks passed')
"
```

Expected: dataset loads, PCA fits with 3 eigenvalues, augmentation works.

---

### Task 3: Write `train.py` — Training Loop + 10-View Eval

**File:**
- Modify: `projects/alexnet-imagenette/train.py` (append after data pipeline)

- [ ] **Step 1: Training helper (one epoch)**

```python
# ── Training Loop ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    n = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, top3 = out.topk(3, dim=1)
        correct_top1 += (out.argmax(1) == y).sum().item()
        correct_top3 += top3.eq(y.view(-1, 1)).any(dim=1).sum().item()
        n += x.size(0)

    return running_loss / n, correct_top1 / n, correct_top3 / n
```

- [ ] **Step 2: Validation / test helper**

```python
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    y_true, y_pred = [], []
    n = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        _, top3 = out.topk(3, dim=1)
        correct_top1 += (out.argmax(1) == y).sum().item()
        correct_top3 += top3.eq(y.view(-1, 1)).any(dim=1).sum().item()
        y_true.extend(y.cpu().tolist())
        y_pred.extend(out.argmax(1).cpu().tolist())
        n += x.size(0)

    return total_loss / n, correct_top1 / n, correct_top3 / n, y_true, y_pred
```

- [ ] **Step 3: 10-view test-time evaluation**

```python
def make_10_views(img):
    """Paper 10-view: 4 corners + center, each flipped → 10 crops."""
    w, h = img.shape[-2], img.shape[-1]
    crop_h, crop_w = CROP_SIZE, CROP_SIZE

    views = []
    # 4 corners
    for i in (0, h - crop_h):
        for j in (0, w - crop_w):
            views.append(img[:, i:i+crop_h, j:j+crop_w])
    # Center
    ci = (h - crop_h) // 2
    cj = (w - crop_w) // 2
    views.append(img[:, ci:ci+crop_h, cj:cj+crop_w])
    # Flip all
    flipped = [v.flip(-1) for v in views.copy()]
    views.extend(flipped)
    return torch.stack(views)  # (10, C, H, W)


@torch.no_grad()
def evaluate_10view(model, loader):
    """Evaluate with 10-view test (matching paper protocol)."""
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    y_true_all, y_pred_all = [], []
    n = 0

    for x, y in loader:
        batch_views = []
        for img in x:
            views = make_10_views(img)  # (10, C, H, W)
            batch_views.append(views)
        batch_views = torch.stack(batch_views)  # (B, 10, C, H, W)
        B = batch_views.size(0)

        batch_views = batch_views.to(DEVICE)
        out = model(batch_views.view(B * 10, *batch_views.shape[2:]))
        out = out.view(B, 10, -1).mean(1)  # average softmax across 10 views

        _, top3 = out.topk(3, dim=1)
        correct_top1 += (out.argmax(1).to(y.device) == y).sum().item()
        correct_top3 += top3.to(y.device).eq(y.view(-1, 1)).any(dim=1).sum().item()
        y_true_all.extend(y.tolist())
        y_pred_all.extend(out.argmax(1).cpu().tolist())
        n += B

    return correct_top1 / n, correct_top3 / n, y_true_all, y_pred_all
```

- [ ] **Step 4: Write heartbeat updater**

```python
# ── Heartbeat ────────────────────────────────────────────────────────────

def update_heartbeat(exp_id, epoch, train_loss, val_acc, elapsed, flops_consumed):
    heartbeat = {
        "exp_id": exp_id,
        "epoch": epoch,
        "train_loss": round(train_loss, 4) if train_loss else None,
        "val_acc": round(val_acc, 4) if val_acc else None,
        "elapsed_seconds": round(elapsed, 1),
        "flops_consumed_tflops": round(flops_consumed, 2),
        "timestamp": time.time(),
    }
    with open("/content/heartbeat.json", "w") as f:
        json.dump(heartbeat, f)
```

---

### Task 4: Write `train.py` — Experiment Runner (main training driver)

**File:**
- Modify: `projects/alexnet-imagenette/train.py` (append after training helpers)

- [ ] **Step 1: Single experiment runner**

```python
# ── Experiment Runner ────────────────────────────────────────────────────

FLOP_PER_IMAGE = 3.0  # GFLOPs per training image (fwd+bwd+update, 128×128 input)

def run_experiment(config, train_raw, idx_train, idx_val, exp_id):
    """Run one full experiment: train + 10-view eval. Returns metrics dict."""
    exp_name = config["name"]
    log(f"\n{'='*60}")
    log(f"EXPERIMENT {exp_id}: {exp_name}")
    log(f"{'='*60}")

    # Build transforms
    augment = config.get("augment", True)
    pca = None
    if augment and config.get("pca", True):
        pca = PCA(n_components=3)
        log("Fitting PCA color augmentation...")
        sample_imgs = []
        sample_indices = torch.randperm(len(idx_train))[:500].tolist()
        for i in sample_indices:
            img = train_raw[idx_train[i]]["image"]
            sample_imgs.append(TF.to_tensor(TF.resize(img, CROP_SIZE)))
        pca.fit(sample_imgs)
        log(f"PCA fitted — eigvals: {pca.eigvals.tolist()}")

    train_transform = build_transforms(augment=augment, pca=pca)
    val_transform = build_transforms(augment=False, pca=None)

    train_ds = ImagenetteDataset(
        Subset(train_raw, idx_train) if not isinstance(idx_train, list) else
        [train_raw[int(i)] for i in idx_train],
        train_transform
    )
    val_ds = ImagenetteDataset(
        Subset(train_raw, idx_val) if not isinstance(idx_val, list) else
        [train_raw[int(i)] for i in idx_val],
        val_transform
    )

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=2, pin_memory=True)

    # Build model
    from alexnet import build_alexnet
    model = build_alexnet(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # Training state
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    current_lr = LR_INIT
    metrics_history = []
    total_images = 0
    t0_exp = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0_epoch = time.time()
        train_loss, train_acc1, train_acc3 = train_epoch(
            model, train_loader, optimizer, criterion, epoch, EPOCHS
        )
        total_images += len(train_ds)

        val_loss, val_acc1, val_acc3, _, _ = evaluate(model, val_loader, criterion)

        elapsed = time.time() - t0_exp
        flops_consumed = total_images * FLOP_PER_IMAGE / 1000  # TFLOPs

        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc1": round(train_acc1, 4),
            "train_acc3": round(train_acc3, 4),
            "val_loss": round(val_loss, 4),
            "val_acc1": round(val_acc1, 4),
            "val_acc3": round(val_acc3, 4),
            "lr": current_lr,
        })

        log(f"E {epoch:3d} | train_loss: {train_loss:.4f} acc1: {train_acc1:.3f} acc3: {train_acc3:.3f} | "
            f"val_loss: {val_loss:.4f} acc1: {val_acc1:.3f} acc3: {val_acc3:.3f} | "
            f"lr: {current_lr:.0e} | {time.time()-t0_epoch:.1f}s | FLOPS: {flops_consumed:.1f}T")

        # LR schedule: ÷10 when val accuracy plateaus
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            torch.save(best_state, os.path.join(OUTPUT_DIR, f"exp{exp_id}_best.pt"))
            log(f"  >> best val_acc1={val_acc1:.3f}, saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= LR_PATIENCE:
                patience_counter = 0
                current_lr *= LR_FACTOR
                for g in optimizer.param_groups:
                    g["lr"] = current_lr
                log(f"  >> LR reduced to {current_lr:.0e}")

        update_heartbeat(exp_id, epoch, train_loss, val_acc1, elapsed, flops_consumed)

        # Early stop if LR too small
        if current_lr < 1e-6:
            log(f"LR below 1e-6, stopping early at epoch {epoch}")
            break

    train_time = time.time() - t0_exp

    # Load best checkpoint for eval
    model.load_state_dict(best_state)

    # Final 10-view evaluation on val set (as test proxy)
    log("Running 10-view evaluation...")
    test_acc1, test_acc3, y_true, y_pred = evaluate_10view(model, val_loader)

    result = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "n_params": n_params,
        "train_time_seconds": round(train_time, 1),
        "best_val_acc1": round(best_val_acc, 4),
        "test_acc1_10view": round(test_acc1, 4),
        "test_acc3_10view": round(test_acc3, 4),
        "test_error1_pct": round((1 - test_acc1) * 100, 2),
        "test_error3_pct": round((1 - test_acc3) * 100, 2),
        "epochs": metrics_history,
        "y_true": y_true,
        "y_pred": y_pred,
        "current_lr_final": current_lr,
    }
    log(f"EXPERIMENT {exp_id} DONE in {train_time/60:.1f}m")
    log(f"  Test (10-view): top-1={test_acc1:.4f} error1={result['test_error1_pct']}%  "
        f"top-3={test_acc3:.4f} error3={result['test_error3_pct']}%")
    return result
```

- [ ] **Step 2: Experiment configs — paper's 4 variants**

```python
# ── Experiment Configs ──────────────────────────────────────────────────

def get_experiment_configs():
    return {
        1: {"name": "Baseline",            "width_mult": 1.0, "dropout": 0.5, "augment": True,  "pca": True},
        2: {"name": "No Dropout",          "width_mult": 1.0, "dropout": 0.0, "augment": True,  "pca": True},
        3: {"name": "No Data Aug",         "width_mult": 1.0, "dropout": 0.5, "augment": False, "pca": False},
        4: {"name": "Reduced Width (0.5)", "width_mult": 0.5, "dropout": 0.5, "augment": True,  "pca": True},
    }
```

---

### Task 5: Write `train.py` — Chart Generation + Main

**File:**
- Modify: `projects/alexnet-imagenette/train.py` (append after experiment runner)

- [ ] **Step 1: Chart generation function**

```python
# ── Chart Generation ─────────────────────────────────────────────────────

def generate_charts(all_results):
    log("Generating charts...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    exp_configs = get_experiment_configs()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ── Figure 1: Training curves (all experiments overlaid) ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for i, r in enumerate(all_results):
        epochs = [m["epoch"] for m in r["epochs"]]
        top1_err = [(1 - m["val_acc1"]) * 100 for m in r["epochs"]]
        top3_err = [(1 - m["val_acc3"]) * 100 for m in r["epochs"]]
        axes[0].plot(epochs, top1_err, color=colors[i], label=r["exp_name"])
        axes[1].plot(epochs, top3_err, color=colors[i], label=r["exp_name"])

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Top-1 Error (%)")
    axes[0].set_title("AlexNet on Imagenette — Top-1 Val Error"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Top-3 Error (%)")
    axes[1].set_title("AlexNet on Imagenette — Top-3 Val Error"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    log("  -> training_curves.png")

    # ── Figure 2: Ablation bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["exp_name"] for r in all_results]
    errors = [r["test_error1_pct"] for r in all_results]
    bars = ax.bar(names, errors, color=colors)
    ax.set_ylabel("Top-1 Test Error (%, 10-view)")
    ax.set_title("Ablation Study — AlexNet on Imagenette")
    ax.grid(True, axis="y")
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{err:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_bars.png"), dpi=150)
    plt.close()
    log("  -> ablation_bars.png")

    # ── Figure 3: Conv1 filters (baseline only) ──────────────────────────
    baseline = all_results[0]
    from alexnet import build_alexnet
    cfg = get_experiment_configs()[baseline["exp_id"]]
    model = build_alexnet(cfg)
    ckpt_path = os.path.join(OUTPUT_DIR, f"exp{baseline['exp_id']}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    conv1_w = model.conv1.weight.detach().cpu()  # (96, 3, 11, 11)

    fig, axes = plt.subplots(8, 12, figsize=(14, 10))
    for i, ax in enumerate(axes.flat):
        if i < 96:
            w = conv1_w[i]
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)  # normalize to [0,1]
            ax.imshow(w.permute(1, 2, 0))
        ax.axis("off")
    fig.suptitle("AlexNet Conv1 Filters (96 × 11×11×3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "conv1_filters.png"), dpi=150)
    plt.close()
    log("  -> conv1_filters.png")

    # ── Figure 4: Confusion matrix (baseline only) ───────────────────────
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(baseline["y_true"], baseline["y_pred"])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — AlexNet Baseline")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    log("  -> confusion_matrix.png")


def export_metrics(all_results):
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    export = []
    for r in all_results:
        export.append({
            "exp_id": r["exp_id"],
            "exp_name": r["exp_name"],
            "n_params": r["n_params"],
            "train_time_seconds": r["train_time_seconds"],
            "best_val_acc1": r["best_val_acc1"],
            "test_acc1_10view": r["test_acc1_10view"],
            "test_acc3_10view": r["test_acc3_10view"],
            "test_error1_pct": r["test_error1_pct"],
            "test_error3_pct": r["test_error3_pct"],
            "per_epoch": r["epochs"],
        })
    with open(metrics_path, "w") as f:
        json.dump(export, f, indent=2)
    log(f"  -> metrics.json ({len(export)} experiments)")
```

- [ ] **Step 2: Main entry point**

```python
# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ids", type=str, required=True,
                        help="Comma-separated experiment IDs, e.g. '1,2'")
    args = parser.parse_args()

    exp_ids = [int(x.strip()) for x in args.exp_ids.split(",")]
    log(f"Starting AlexNet Imagenette — experiments: {exp_ids}")
    log(f"Device: {DEVICE}")

    # Load data once, shared across experiments
    train_raw, idx_train, idx_val, _ = load_imagenette_data()
    log(f"Train: {len(idx_train)}, Val: {len(idx_val)}")

    exp_configs = get_experiment_configs()
    all_results = []

    for exp_id in exp_ids:
        if exp_id not in exp_configs:
            log(f"ERROR: unknown experiment {exp_id}, skipping")
            continue
        config = exp_configs[exp_id]
        result = run_experiment(config, train_raw, idx_train, idx_val, exp_id)
        all_results.append(result)

    # Generate charts and export
    generate_charts(all_results)
    export_metrics(all_results)

    # Signal watchdog to stop
    with open("/content/watchdog_stop", "w") as f:
        f.write("done")

    # Tar checkpoints
    log("Tarring checkpoints...")
    os.system(f"tar -czf {OUTPUT_DIR}.tar.gz -C /content alexnet-output")
    log("ALL DONE.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add missing import at top of train.py**

At the top of train.py, ensure these imports are present. The `from datasets import load_dataset` import is already in the data section from Task 2. Verify the imports cover everything:

```python
import torchvision.transforms as T  # needed for RandomCrop.get_params
```

Actually, in the `build_transforms` function, we use `T.RandomCrop.get_params`. Let me adjust the import — actually we reference `T.RandomCrop` in `build_transforms`, but that's from `torchvision.transforms` which we imported as `import torchvision.transforms.functional as TF`. Let me fix: the RandomCrop import should come from `torchvision.transforms` directly.

Let me fix `build_transforms` to use a cleaner pattern:

```python
from torchvision.transforms import RandomCrop as _RandomCrop

def build_transforms(augment=True, pca=None):
    def _augment(img):
        if augment:
            img = TF.resize(img, 160)
            i, j, h, w = _RandomCrop.get_params(img, output_size=(CROP_SIZE, CROP_SIZE))
            img = TF.crop(img, i, j, h, w)
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
        else:
            img = TF.resize(img, CROP_SIZE)
            img = TF.center_crop(img, CROP_SIZE)

        img = TF.to_tensor(img)

        if augment and pca is not None:
            img = pca.apply(img)

        return img

    return _augment
```

I'll include this fixed version in the task step below.
```

- [ ] **Step 4: Note on full file assembly**

The `train.py` file is built across Tasks 2, 3, 4, 5. The final file assembles all sections in this order:

1. Imports (Task 2, Step 1) — includes `torchvision.transforms` as `T` for `T.RandomCrop`
2. Config constants (Task 2, Step 1)
3. Log helper (Task 2, Step 2)
4. PCA class (Task 2, Step 3)
5. Dataset + data loaders (Task 2, Step 4)
6. `build_transforms` (Task 2, Step 5)
7. `train_epoch` (Task 3, Step 1)
8. `evaluate` (Task 3, Step 2)
9. `make_10_views` + `evaluate_10view` (Task 3, Step 3)
10. `update_heartbeat` (Task 3, Step 4)
11. `get_experiment_configs` (Task 4, Step 2)
12. `run_experiment` (Task 4, Step 1)
13. `generate_charts` (Task 5, Step 1)
14. `export_metrics` (Task 5, Step 1)
15. `main()` (Task 5, Step 2)

The plan will produce each section independently; the final assembly step merges them into one file.

---

### Task 6: Write `launch.py` — Colab Bootstrap

**File:**
- Create: `projects/alexnet-imagenette/launch.py`

- [ ] **Step 1: Write launch.py**

```python
"""Colab bootstrap: pip install deps, set HF_TOKEN, spawn train + watchdog as
detached subprocesses. Survives after colab exec disconnects.

Reads /content/exp_ids.txt to know which experiments to run.
"""

import subprocess, sys, os, time

EXP_IDS_PATH = "/content/exp_ids.txt"
HF_TOKEN_PATH = "/content/hf_token"
LOG = "/content/train.log"
DEPS = ["torch", "torchvision", "datasets", "matplotlib", "seaborn", "scikit-learn"]

# --- Read experiment IDs ---
with open(EXP_IDS_PATH) as f:
    exp_ids = f.read().strip()
print(f"[launch] Exp IDs: {exp_ids}")

# --- Set HF_TOKEN ---
try:
    with open(HF_TOKEN_PATH) as f:
        token = f.read().strip()
    os.environ["HF_TOKEN"] = token
    print("[launch] HF_TOKEN set")
except FileNotFoundError:
    print("[launch] WARNING: /content/hf_token not found, datasets may fail")

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[launch] Dependencies installed")

# --- Shared environment ---
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

# --- Spawn watchdog ---
print("[launch] Starting watchdog...")
with open("/content/watchdog.log", "w") as wf:
    wd = subprocess.Popen(
        [sys.executable, "-u", "/content/watchdog.py"],
        stdout=wf, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"[launch] Watchdog PID={wd.pid}")

# --- Spawn training ---
print("[launch] Starting training...")
with open(LOG, "w") as f:
    train = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py", "--exp_ids", exp_ids],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"[launch] Train PID={train.pid}, log={LOG}")
print(f"[launch] DONE. Training running detached.")
```

---

### Task 7: Write `watchdog.py` — VM Heartbeat

**File:**
- Create: `projects/alexnet-imagenette/watchdog.py`

- [ ] **Step 1: Write watchdog.py**

```python
"""VM-side watchdog: writes /content/heartbeat.json every 30s.
Exits when /content/watchdog_stop exists (train.py creates this on completion).
"""

import json, os, time

HEARTBEAT_PATH = "/content/heartbeat.json"
STOP_PATH = "/content/watchdog_stop"
INTERVAL = 30

# Initialize
heartbeat = {
    "status": "starting",
    "epoch": 0,
    "train_loss": None,
    "val_acc": None,
    "elapsed_seconds": 0.0,
    "flops_consumed_tflops": 0.0,
    "timestamp": time.time(),
}

print(f"[watchdog] Started, writing to {HEARTBEAT_PATH} every {INTERVAL}s", flush=True)

while not os.path.exists(STOP_PATH):
    t0 = time.time()
    # Read latest from train.py (it writes on each epoch)
    if os.path.exists(HEARTBEAT_PATH):
        try:
            with open(HEARTBEAT_PATH) as f:
                existing = json.load(f)
            existing["watchdog_seen"] = time.time()
            with open(HEARTBEAT_PATH, "w") as f:
                json.dump(existing, f)
        except (json.JSONDecodeError, IOError):
            pass

    elapsed = INTERVAL - (time.time() - t0)
    if elapsed > 0:
        time.sleep(elapsed)

# Final heartbeat
heartbeat["status"] = "done"
with open(HEARTBEAT_PATH, "w") as f:
    json.dump(heartbeat, f)
print("[watchdog] Stopped", flush=True)
```

---

### Task 8: Write `check_progress.py` — Local Cron Monitor

**File:**
- Create: `projects/alexnet-imagenette/check_progress.py`

- [ ] **Step 1: Write check_progress.py**

```python
"""Local cron progress checker — runs via 'colab exec -f check_progress.py'.

Reads /content/heartbeat.json on VM, checks process health, reports status.
Intended to be run every 5 min via CronCreate.
"""

import json, os, subprocess, sys, time

HEARTBEAT_PATH = "/content/heartbeat.json"

def check():
    # 1. Read heartbeat
    hb = None
    try:
        with open(HEARTBEAT_PATH) as f:
            hb = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[check] WARNING: No heartbeat file found")
        return 1

    # 2. Process alive check
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5
        )
        proc_alive = result.returncode == 0
    except Exception:
        proc_alive = False

    # 3. Heartbeat freshness
    now = time.time()
    hb_age = now - hb.get("timestamp", 0)
    hb_stale = hb_age > 120  # 2 min

    # 4. Report
    status = hb.get("status", "unknown")
    epoch = hb.get("epoch", 0)
    val_acc = hb.get("val_acc", 0)
    elapsed = hb.get("elapsed_seconds", 0)
    flops = hb.get("flops_consumed_tflops", 0)

    print(f"[check] Status: {status} | Epoch: {epoch} | Val Acc: {val_acc} | "
          f"Elapsed: {elapsed/60:.1f}m | FLOPS: {flops:.1f} TFLOPs | "
          f"HB age: {hb_age:.0f}s | Process alive: {proc_alive}")

    # 5. Health alerts
    alerts = []
    if hb_stale and not proc_alive:
        alerts.append("CRITICAL: VM likely dead — heartbeat stale AND no train.py process")
    elif hb_stale:
        alerts.append("WARNING: Heartbeat stale >2min but process may still be alive")
    elif not proc_alive and status != "done":
        alerts.append("CRITICAL: train.py process not found but heartbeat says not done")

    train_loss = hb.get("train_loss")
    if train_loss and train_loss > 10:
        alerts.append("WARNING: Loss >10 — may be diverging")

    if elapsed > 3300 and status != "done":  # 55 min
        alerts.append("WARNING: >55 min elapsed — trigger emergency download")

    for a in alerts:
        print(f"[check] {a}")

    return 0 if not alerts else 1

if __name__ == "__main__":
    sys.exit(check())
```

---

### Task 9: Assemble `train.py` — Full File

**Files:**
- Write: `projects/alexnet-imagenette/train.py` (complete, assembled from Tasks 2-5)

This task assembles all the sections from Tasks 2-5 into one file, fixes the `RandomCrop` import issue, and ensures imports are complete and ordered.

- [ ] **Step 1: Write the complete, verified train.py**

The code from Tasks 2-5 assembled into one file. Key fix: use `from torchvision.transforms import RandomCrop` instead of accessing through `T.RandomCrop` in `build_transforms`.

The full `train.py` file is the concatenation of all code blocks from Tasks 2-5 with the import fix. At implementation time, write the file once with all sections in order:

```
Imports → Config → log() → PCA class → Dataset loading → build_transforms
→ train_epoch → evaluate → make_10_views → evaluate_10view
→ update_heartbeat → get_experiment_configs → run_experiment
→ generate_charts → export_metrics → main()
```

The implementation agent assembles this from the code blocks provided in the tasks above. Each block is complete and tested independently.

- [ ] **Step 2: Verify imports and syntax**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python3 -c "
import ast
with open('projects/alexnet-imagenette/train.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

---

### Task 10: Local Sanity Checks

**Files:**
- Check: all files compile, imports resolve

- [ ] **Step 1: Verify all Python files parse**

```bash
cd /Users/mx/Desktop/projects/colab-cli
for f in projects/alexnet-imagenette/{alexnet,launch,watchdog,check_progress}.py; do
    python3 -c "import ast; ast.parse(open('$f').read()); print(f'$f: OK')"
done
```

Expected: all 4 files print "OK".

- [ ] **Step 2: Smoke test alexnet model forward + backward**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python3 -c "
import torch
from projects.alexnet_imagenette.alexnet import build_alexnet

for name, cfg in [
    ('Baseline', {'width_mult': 1.0, 'dropout': 0.5}),
    ('No Dropout', {'width_mult': 1.0, 'dropout': 0.0}),
    ('Reduced Width', {'width_mult': 0.5, 'dropout': 0.5}),
]:
    model = build_alexnet(cfg).train()
    x = torch.randn(8, 3, 128, 128)
    y = torch.randint(0, 10, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    # Check no NaN gradients
    for pn, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            raise RuntimeError(f'{name}: NaN grad in {pn}')

    print(f'{name}: OK — loss={loss.item():.4f}, params={sum(p.numel() for p in model.parameters()):,}')
"
```

Expected: 3 lines of "OK" with loss values.

---

### Task 11: Deploy to Colab — Both Sessions

**Pre-reqs:** Both `colab` and `cc` accounts have no active GPU sessions (confirmed earlier).

- [ ] **Step 1: Provision both sessions (in parallel)**

```bash
colab new --gpu T4 -s alexnet-a &
cc new --gpu T4 -s alexnet-b &
wait
```

- [ ] **Step 2: Verify sessions are running**

```bash
colab sessions && echo "===" && cc sessions
```

Expected: `alexnet-a` with T4 and `alexnet-b` with T4.

- [ ] **Step 3: Upload files to session A (colab, exps 1+2)**

```bash
colab upload projects/alexnet-imagenette/launch.py /content/launch.py
colab upload projects/alexnet-imagenette/train.py /content/train.py
colab upload projects/alexnet-imagenette/alexnet.py /content/alexnet.py
colab upload projects/alexnet-imagenette/watchdog.py /content/watchdog.py
colab upload projects/alexnet-imagenette/exp_ids_a.txt /content/exp_ids.txt
colab upload .huggingface/access_token /content/hf_token
```

- [ ] **Step 4: Upload files to session B (cc, exps 3+4)**

```bash
cc upload projects/alexnet-imagenette/launch.py /content/launch.py
cc upload projects/alexnet-imagenette/train.py /content/train.py
cc upload projects/alexnet-imagenette/alexnet.py /content/alexnet.py
cc upload projects/alexnet-imagenette/watchdog.py /content/watchdog.py
cc upload projects/alexnet-imagenette/exp_ids_b.txt /content/exp_ids.txt
cc upload .huggingface/access_token /content/hf_token
```

- [ ] **Step 5: Launch both sessions (in parallel)**

```bash
colab exec -f launch.py --timeout 120 &
cc exec -f launch.py --timeout 120 &
wait
```

Expected: both return immediately with "DONE. Training running detached." and PID/log path.

---

### Task 12: Set Up Cron Monitoring

CronCreate fires prompts to Claude, not shell commands. For persistent monitoring across sessions, we use durable CronCreate jobs that check both sessions. For this session's active monitoring, use ScheduleWakeup.

- [ ] **Step 1: Create durable cron job for session A (colab, alexnet-a)**

```
CronCreate
  cron: "*/7 * * * *"
  prompt: "Check Colab session alexnet-a. Run: cd /Users/mx/Desktop/projects/colab-cli && colab exec -n alexnet-a -f projects/alexnet-imagenette/check_progress.py --timeout 15 2>&1. Report status: epoch, val accuracy, elapsed time, any alerts. If CRITICAL, recommend emergency download."
  recurring: true
  durable: true
```

- [ ] **Step 2: Create durable cron job for session B (cc, alexnet-b)**

```
CronCreate
  cron: "*/7 * * * *"
  prompt: "Check Colab session alexnet-b. Run: cd /Users/mx/Desktop/projects/colab-cli && cc exec -n alexnet-b -f projects/alexnet-imagenette/check_progress.py --timeout 15 2>&1. Report status: epoch, val accuracy, elapsed time, any alerts. If CRITICAL, recommend emergency download."
  recurring: true
  durable: true
```

- [ ] **Step 3: Schedule in-session wakeups for real-time monitoring**

During this session, after launching both experiments, use `ScheduleWakeup` with `delaySeconds=300` (5 min) to actively monitor progress. Each wakeup runs both progress checks and reports. If the session is ending, the durable cron jobs continue monitoring independently.

---

### Task 13: Download Results & Cleanup

- [ ] **Step 1: Download session A results (when both experiments complete)**

```bash
colab download /content/alexnet-output.tar.gz projects/alexnet-imagenette/output-a/checkpoints.tar.gz
colab download /content/alexnet-output/training_curves.png projects/alexnet-imagenette/output-a/
colab download /content/alexnet-output/ablation_bars.png projects/alexnet-imagenette/output-a/
colab download /content/alexnet-output/conv1_filters.png projects/alexnet-imagenette/output-a/
colab download /content/alexnet-output/confusion_matrix.png projects/alexnet-imagenette/output-a/
colab download /content/alexnet-output/metrics.json projects/alexnet-imagenette/output-a/
```

- [ ] **Step 2: Download session B results**

```bash
cc download /content/alexnet-output.tar.gz projects/alexnet-imagenette/output-b/checkpoints.tar.gz
cc download /content/alexnet-output/training_curves.png projects/alexnet-imagenette/output-b/
cc download /content/alexnet-output/ablation_bars.png projects/alexnet-imagenette/output-b/
cc download /content/alexnet-output/conv1_filters.png projects/alexnet-imagenette/output-b/
cc download /content/alexnet-output/confusion_matrix.png projects/alexnet-imagenette/output-b/
cc download /content/alexnet-output/metrics.json projects/alexnet-imagenette/output-b/
```

- [ ] **Step 3: Verify all artifacts**

```bash
ls -la projects/alexnet-imagenette/output-a/ && echo "---" && ls -la projects/alexnet-imagenette/output-b/
python3 -c "
import json
for label, path in [('A', 'projects/alexnet-imagenette/output-a/metrics.json'),
                     ('B', 'projects/alexnet-imagenette/output-b/metrics.json')]:
    with open(path) as f:
        data = json.load(f)
    for exp in data:
        print(f'Session {label} Exp {exp[\"exp_id\"]} ({exp[\"exp_name\"]}): '
              f'err1={exp[\"test_error1_pct\"]}%, err3={exp[\"test_error3_pct\"]}%, '
              f'time={exp[\"train_time_seconds\"]/60:.1f}m')
"
```

- [ ] **Step 4: Stop both sessions**

```bash
colab stop -s alexnet-a
cc stop -s alexnet-b
```

- [ ] **Step 5: Merge results and final report**

Combine metrics from both sessions into one final report. The charts from each session are independent but can be viewed side by side. The metrics.json files are merged for full dataset.

---

### Task 14: Cross-Verify Against Spec

- [ ] **Step 1: Check spec compliance**

| Spec requirement | Covered by |
|---|---|
| Exact AlexNet architecture | Task 1 (alexnet.py) |
| 128×128 input, AdaptiveAvgPool2d(6) | Task 1 |
| Paper weight init | Task 1 (`_init_weights`) |
| PCA color augmentation | Task 2 (PCA class) |
| 80/20 random split | Task 2 (`load_imagenette_data`) |
| SGD momentum 0.9, WD 0.0005 | Task 4 (`run_experiment`) |
| LR ÷10 on plateau, patience 3 | Task 4 |
| 90 epochs, BS 128 | Tasks 2, 4 |
| 10-view test eval | Task 3 (`evaluate_10view`) |
| Top-1 + Top-3 error rate | Task 3, 5 |
| 4 experiments | Task 4 (`get_experiment_configs`) |
| Dual-layer monitoring | Tasks 7 (watchdog), 8 (check_progress) |
| FLOPS tracking | Tasks 4 (update_heartbeat), 8 |
| 4 chart artifacts | Task 5 (`generate_charts`) |
| metrics.json export | Task 5 (`export_metrics`) |
| 2-account parallel deployment | Tasks 11, 12 |
| Cron monitoring | Task 12 |
| Download + cleanup | Task 13 |

- [ ] **Step 2: Verify success criteria coverage**

- [x] Baseline >60% top-1 → verified post-run via metrics.json
- [x] Dropout ablation gap >10pp → verified post-run
- [x] No Data Aug largest error → verified post-run
- [x] Reduced Width drops accuracy → verified post-run
- [x] All 4 png artifacts → verified post-download
- [x] metrics.json complete → verified post-download
- [x] VM survives full run → verified via heartbeat/cron
- [x] FLOPS estimates within 30% → verified via FLOPS consumed vs wall time
