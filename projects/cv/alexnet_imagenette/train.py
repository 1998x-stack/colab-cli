"""AlexNet training on Imagenette — data pipeline, training loop, 10-view eval,
4-experiment orchestrator, chart generation.

Usage (on Colab VM): python -u train.py --exp_ids 1,2
"""

import json
import os
import time
import argparse
import tarfile
import urllib.request
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from torchvision.datasets import ImageFolder

OUTPUT_DIR = "/content/alexnet-output"
LOG_PATH = "/content/train.log"
DATA_DIR = "/content/imagenette2-160"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "tench", "English springer", "cassette player", "chain saw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute",
]
NUM_CLASSES = 10

BATCH_SIZE = 128
LR_INIT = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_PATIENCE = 3
LR_FACTOR = 0.1
EPOCHS = 20
CROP_SIZE = 128

LOG_FILE = open(LOG_PATH, "w", buffering=1)

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    LOG_FILE.write(line + "\n")


# ── Data pipeline ────────────────────────────────────────────────────────

def download_imagenette():
    """Download and extract Imagenette-160 to DATA_DIR if not present."""
    if os.path.exists(DATA_DIR):
        log(f"Imagenette already at {DATA_DIR}")
        return

    tgz_path = "/content/imagenette2.tgz"
    log(f"Downloading Imagenette from {IMAGENETTE_URL}...")
    urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path)
    log(f"Downloaded {os.path.getsize(tgz_path)/1024/1024:.0f}MB. Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path="/content", filter="data")
    os.remove(tgz_path)
    log(f"Extracted to {DATA_DIR}")


def load_imagenette_data():
    """Load Imagenette via ImageFolder. Returns train_dataset, val_dataset."""
    download_imagenette()

    train_dir = os.path.join(DATA_DIR, "train")
    full_train = ImageFolder(train_dir)
    n_train = int(0.8 * len(full_train))
    n_val = len(full_train) - n_train
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    log(f"Train: {n_train}, Val: {n_val}")
    return full_train, train_ds, val_ds


# ── PCA Color Augmentation (Fancy PCA) ────────────────────────────────────

class PCA:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.eigvals = None
        self.eigvecs = None

    def fit(self, images):
        """Fit PCA on a list of image tensors (each C×H×W, [0,1] range)."""
        pixels = torch.stack([img.reshape(3, -1) for img in images])
        cov = torch.cov(pixels.permute(1, 0, 2).reshape(3, -1))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        self.eigvals = eigvals[-self.n_components:]
        self.eigvecs = eigvecs[:, -self.n_components:]

    def apply(self, img):
        """Apply PCA color augmentation to a single image tensor (C×H×W)."""
        if self.eigvals is None:
            return img
        alpha = torch.randn(self.n_components) * 0.1
        delta = (self.eigvecs * self.eigvals.sqrt() * alpha).sum(dim=1)
        return img + delta.view(3, 1, 1)


def build_transforms(augment=True, pca=None):
    def _augment(img):
        if augment:
            img = TF.resize(img, 160)
            i, j, h, w = RandomCrop.get_params(img, output_size=(CROP_SIZE, CROP_SIZE))
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


# ── Training helpers ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    n = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, top3 = out.topk(3, dim=1)
        correct_top1 += (out.argmax(1) == y).sum().item()
        correct_top3 += top3.eq(y.view(-1, 1)).any(dim=1).sum().item()
        n += x.size(0)

    return running_loss / n, correct_top1 / n, correct_top3 / n


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


# ── 10-View Evaluation ───────────────────────────────────────────────────

def make_10_views(img):
    h, w = img.shape[-2], img.shape[-1]
    crop_h, crop_w = CROP_SIZE, CROP_SIZE

    views = []
    for i in (0, h - crop_h):
        for j in (0, w - crop_w):
            views.append(img[:, i:i+crop_h, j:j+crop_w])
    ci = (h - crop_h) // 2
    cj = (w - crop_w) // 2
    views.append(img[:, ci:ci+crop_h, cj:cj+crop_w])
    flipped = [v.flip(-1) for v in views.copy()]
    views.extend(flipped)
    return torch.stack(views)


@torch.no_grad()
def evaluate_10view(model, loader):
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    y_true_all, y_pred_all = [], []
    n = 0

    for x, y in loader:
        batch_views = []
        for img in x:
            views = make_10_views(img)
            batch_views.append(views)
        batch_views = torch.stack(batch_views)
        B = batch_views.size(0)

        batch_views = batch_views.to(DEVICE)
        out = model(batch_views.view(B * 10, *batch_views.shape[2:]))
        out = out.view(B, 10, -1).mean(1)

        _, top3 = out.topk(3, dim=1)
        correct_top1 += (out.argmax(1).to(y.device) == y).sum().item()
        correct_top3 += top3.to(y.device).eq(y.view(-1, 1)).any(dim=1).sum().item()
        y_true_all.extend(y.tolist())
        y_pred_all.extend(out.argmax(1).cpu().tolist())
        n += B

    return correct_top1 / n, correct_top3 / n, y_true_all, y_pred_all


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


# ── Experiment Configs ──────────────────────────────────────────────────

def get_experiment_configs():
    return {
        1: {"name": "Baseline",            "width_mult": 1.0, "dropout": 0.5, "augment": True,  "pca": True},
        2: {"name": "No Dropout",          "width_mult": 1.0, "dropout": 0.0, "augment": True,  "pca": True},
        3: {"name": "No Data Aug",         "width_mult": 1.0, "dropout": 0.5, "augment": False, "pca": False},
        4: {"name": "Reduced Width (0.5)", "width_mult": 0.5, "dropout": 0.5, "augment": True,  "pca": True},
    }


# ── Experiment Runner ────────────────────────────────────────────────────

FLOP_PER_IMAGE = 3.0

class TransformedDataset(Dataset):
    """Apply a callable transform to an ImageFolder subset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]
        if self.transform:
            img = self.transform(img)
        return img, label


def run_experiment(config, full_train, train_subset, val_subset, exp_id):
    """Run one full experiment: train + 10-view eval. Returns metrics dict."""
    exp_name = config["name"]
    log(f"\n{'='*60}")
    log(f"EXPERIMENT {exp_id}: {exp_name}")
    log(f"{'='*60}")

    augment = config.get("augment", True)
    pca = None
    if augment and config.get("pca", True):
        pca = PCA(n_components=3)
        log("Fitting PCA color augmentation...")
        sample_imgs = []
        n_samples = min(500, len(train_subset))
        for j in range(n_samples):
            img, _ = full_train[train_subset.indices[j]]
            sample_imgs.append(TF.to_tensor(TF.resize(img, [CROP_SIZE, CROP_SIZE])))
        pca.fit(sample_imgs)
        log(f"PCA fitted — eigvals: {pca.eigvals.tolist()}")

    train_transform = build_transforms(augment=augment, pca=pca)
    val_transform = build_transforms(augment=False, pca=None)

    train_ds = TransformedDataset(train_subset, train_transform)
    val_ds = TransformedDataset(val_subset, val_transform)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=2, pin_memory=True)

    from alexnet import build_alexnet
    model = build_alexnet(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

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
            model, train_loader, optimizer, criterion
        )
        total_images += len(train_ds)

        val_loss, val_acc1, val_acc3, _, _ = evaluate(model, val_loader, criterion)

        elapsed = time.time() - t0_exp
        flops_consumed = total_images * FLOP_PER_IMAGE / 1000

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

        if current_lr < 1e-6:
            log(f"LR below 1e-6, stopping early at epoch {epoch}")
            break

    train_time = time.time() - t0_exp
    model.load_state_dict(best_state)

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
    }
    log(f"EXPERIMENT {exp_id} DONE in {train_time/60:.1f}m")
    log(f"  Test (10-view): top-1={test_acc1:.4f} error1={result['test_error1_pct']}%  "
        f"top-3={test_acc3:.4f} error3={result['test_error3_pct']}%")
    return result


# ── Chart Generation ─────────────────────────────────────────────────────

def generate_charts(all_results):
    log("Generating charts...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as cm_fn

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ── Figure 1: Training curves ─────────────
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

    # ── Figure 2: Ablation bar chart ─────────
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

    # ── Figure 3: Conv1 filters ──────────────
    baseline = all_results[0]
    from alexnet import build_alexnet
    cfg = get_experiment_configs()[baseline["exp_id"]]
    model = build_alexnet(cfg)
    ckpt_path = os.path.join(OUTPUT_DIR, f"exp{baseline['exp_id']}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    conv1_w = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(8, 12, figsize=(14, 10))
    for i, ax in enumerate(axes.flat):
        if i < 96:
            w = conv1_w[i]
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax.imshow(w.permute(1, 2, 0))
        ax.axis("off")
    fig.suptitle("AlexNet Conv1 Filters (96 x 11x11x3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "conv1_filters.png"), dpi=150)
    plt.close()
    log("  -> conv1_filters.png")

    # ── Figure 4: Confusion matrix ───────────
    cm = cm_fn(baseline["y_true"], baseline["y_pred"])
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ids", type=str, required=True,
                        help="Comma-separated experiment IDs, e.g. '1,2'")
    args = parser.parse_args()

    exp_ids = [int(x.strip()) for x in args.exp_ids.split(",")]
    log(f"Starting AlexNet Imagenette — experiments: {exp_ids}")
    log(f"Device: {DEVICE}")

    full_train, train_subset, val_subset = load_imagenette_data()

    exp_configs = get_experiment_configs()
    all_results = []

    for exp_id in exp_ids:
        if exp_id not in exp_configs:
            log(f"ERROR: unknown experiment {exp_id}, skipping")
            continue
        config = exp_configs[exp_id]
        result = run_experiment(config, full_train, train_subset, val_subset, exp_id)
        all_results.append(result)

    generate_charts(all_results)
    export_metrics(all_results)

    with open("/content/watchdog_stop", "w") as f:
        f.write("done")

    log("Tarring checkpoints...")
    os.system(f"tar -czf {OUTPUT_DIR}.tar.gz -C /content alexnet-output")
    log("ALL DONE.")


if __name__ == "__main__":
    main()
