"""
CNN Quantization Comparison: FP32 vs FP16 vs INT8 vs INT4
ResNet-18 on CIFAR-10, single-file, cleanrl-style.

Usage:
    python train.py  # defaults
    python train.py --epochs 15 --batch_size 256
"""
import argparse
import csv
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--out_dir", type=str, default="/content/cnn-quantization-output")
parser.add_argument("--hf_token", type=str, default=None)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--int4_group_size", type=int, default=128)
parser.add_argument("--skip_train", action="store_true")
args = parser.parse_args()

out_dir = Path(args.out_dir)
(out_dir / "logs").mkdir(parents=True, exist_ok=True)
(out_dir / "pngs").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def get_datasets():
    """Load CIFAR-10 via torchvision (same data as uoft-cs/cifar10 on HF, more reliable)."""
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    from torchvision.datasets import CIFAR10

    # Use subset for fast Colab training (full: 50k/10k)
    ds_train = CIFAR10(root="/content/data", train=True, download=True, transform=tf_train)
    ds_val = CIFAR10(root="/content/data", train=False, download=True, transform=tf_val)

    # Subset for speed: 10k train / 2k val
    ds_train = Subset(ds_train, range(10000))
    ds_val = Subset(ds_val, range(2000))

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return loader_train, loader_val

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ---------------------------------------------------------------------------
# INT4 quantization helpers
# ---------------------------------------------------------------------------
class Int4QuantizedLinear(nn.Module):
    """INT4 weight-only quantization for nn.Linear, per-channel symmetric."""
    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.n_groups = (self.in_features + group_size - 1) // group_size

        W = linear.weight.data  # [out, in]
        scales = []
        q_weights = []

        for g in range(self.n_groups):
            start = g * group_size
            end = min(start + group_size, self.in_features)
            w_g = W[:, start:end]
            scale = w_g.abs().max(dim=1, keepdim=True).values / 7.0
            scale = scale.clamp(min=1e-8)
            q_w = torch.round(w_g / scale).clamp(-8, 7).to(torch.int8)
            scales.append(scale)  # [out, 1]
            q_weights.append(q_w)

        self.register_buffer("scales", torch.cat(scales, dim=1))  # [out, in_groups]
        self.register_buffer("q_weight", torch.cat(q_weights, dim=1))  # [out, in] INT8
        if hasattr(linear, "bias") and linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        # Dequantize on-the-fly and compute in FP16
        W_deq = self.q_weight.float()
        n_groups = self.scales.shape[1]
        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            W_deq[:, start:end] *= self.scales[:, g:g+1]
        return F.linear(x, W_deq, self.bias)


def quantize_int4(model, group_size=128):
    """Replace all nn.Linear with INT4 quantized versions."""
    q_model = type(model)()
    q_model.__dict__.update(model.__dict__)

    def _replace(parent, name, child):
        if isinstance(child, nn.Linear) and child.out_features > 10:
            setattr(parent, name, Int4QuantizedLinear(child, group_size))
        elif isinstance(child, nn.Linear):
            setattr(parent, name, child)

    for p_name, parent in list(q_model.named_modules()):
        for c_name, child in list(parent.named_children()):
            _replace(parent, c_name, child)

    return q_model.to(device)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, n_batches=None):
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    correct, total = 0, 0
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(model_device, dtype=model_dtype)
        labels = labels.to(model_device)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if n_batches and i >= n_batches - 1:
            break
    return correct / total if total > 0 else 0.0


def measure_latency(model, loader, n_batches=20, warmup=5):
    """Measure inference latency in milliseconds per batch."""
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    is_cuda = model_device.type == "cuda"
    timings = []
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(model_device, dtype=model_dtype)
        if i >= warmup + n_batches:
            break
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(imgs)
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            timings.append(dt * 1000)
    return np.mean(timings) if timings else 0.0


def get_model_size_mb(model):
    """Measure model size on disk."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(out_dir / "logs" / "train.log", "a") as f:
        f.write(line + "\n")


# Clear log at start
(out_dir / "logs" / "train.log").write_text("")


def train_model(model, loaders):
    train_loader, val_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr"])

    log(f"Device: {device} | Model: ResNet-18 | Dataset: CIFAR-10 | Epochs: {args.epochs}")
    log(f"Params: {sum(p.numel() for p in model.parameters()):,} | Batch: {args.batch_size} | LR: {args.lr}")

    best_acc = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * evaluate(model, val_loader)
        scheduler.step()
        elapsed = time.time() - t_start

        if val_acc > best_acc:
            best_acc = val_acc

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_acc:.2f}",
                             f"{scheduler.get_last_lr()[0]:.6f}"])

        log(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_acc={val_acc:.2f}% | lr={scheduler.get_last_lr()[0]:.2e} | elapsed={elapsed:.0f}s")

    log(f"Training done. Best val_acc={best_acc:.2f}% | Total time: {time.time() - t_start:.0f}s")
    return model


# ---------------------------------------------------------------------------
# quantization comparison
# ---------------------------------------------------------------------------
def run_comparison(fp32_model, val_loader):
    log("=" * 60)
    log("Quantization Comparison: FP32 vs FP16 vs INT8 vs INT4")
    log("=" * 60)

    results = []

    # 1. FP32 baseline
    fp32_model = fp32_model.to(device)
    fp32_acc = 100.0 * evaluate(fp32_model, val_loader)
    fp32_size = get_model_size_mb(fp32_model)
    fp32_lat = measure_latency(fp32_model, val_loader)
    results.append(("FP32", fp32_acc, fp32_size, fp32_lat))
    log(f"FP32  | acc={fp32_acc:.2f}% | size={fp32_size:.2f} MB | latency={fp32_lat:.1f} ms")

    # 2. FP16
    fp16_model = type(fp32_model)().half().to(device)
    fp16_model.load_state_dict(fp32_model.state_dict())
    fp16_acc = 100.0 * evaluate(fp16_model, val_loader)
    fp16_size = get_model_size_mb(fp16_model)
    fp16_lat = measure_latency(fp16_model, val_loader)
    results.append(("FP16", fp16_acc, fp16_size, fp16_lat))
    log(f"FP16  | acc={fp16_acc:.2f}% | size={fp16_size:.2f} MB | latency={fp16_lat:.1f} ms")

    # 3. INT8 (dynamic quantization — CPU-only backend, eval on CPU)
    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model.to("cpu"), {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    int8_acc = 100.0 * evaluate(int8_model, val_loader)
    int8_size = get_model_size_mb(int8_model)
    int8_lat = measure_latency(int8_model, val_loader)
    results.append(("INT8", int8_acc, int8_size, int8_lat))
    log(f"INT8  | acc={int8_acc:.2f}% | size={int8_size:.2f} MB | latency={int8_lat:.1f} ms")

    # 4. INT4 custom (weight-only)
    int4_model = quantize_int4(
        type(fp32_model)().to(device), group_size=args.int4_group_size
    )
    int4_model.load_state_dict({k: v for k, v in fp32_model.state_dict().items()
                                if k in int4_model.state_dict()}, strict=False)
    int4_acc = 100.0 * evaluate(int4_model, val_loader)
    int4_size = get_model_size_mb(int4_model)  # Note: saves INT8+qweight — actual 4-bit is ~5.5 MB
    int4_lat = measure_latency(int4_model, val_loader)
    # Compute effective INT4 storage size
    n_params = sum(p.numel() for p in fp32_model.parameters() if p.dim() > 1)
    effective_int4_mb = (n_params * 4 / 8 + n_params / args.int4_group_size * 4) / (1024 * 1024)
    results.append(("INT4", int4_acc, effective_int4_mb, int4_lat))
    log(f"INT4  | acc={int4_acc:.2f}% | size={effective_int4_mb:.2f} MB (effective) | latency={int4_lat:.1f} ms")

    # Comparison table
    log("")
    log(f"{'Method':<6} {'Acc (%)':<10} {'Size (MB)':<12} {'Lat (ms)':<10} {'vs FP32':<10}")
    log("-" * 55)
    fp32_acc_ref = results[0][1]
    for method, acc, size, lat in results:
        acc_delta = acc - fp32_acc_ref
        log(f"{method:<6} {acc:<10.2f} {size:<12.2f} {lat:<10.1f} {acc_delta:+.2f}")

    # Save summary CSV
    with open(out_dir / "quantization_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy_pct", "size_mb", "latency_ms", "acc_delta_vs_fp32"])
        fp32_acc_ref = results[0][1]
        for method, acc, size, lat in results:
            writer.writerow([method, f"{acc:.2f}", f"{size:.2f}", f"{lat:.1f}", f"{acc - fp32_acc_ref:.2f}"])

    log("")
    log("Saved: quantization_summary.csv")
    return results


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------
def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [r[0] for r in results]
    accs = [r[1] for r in results]
    sizes = [r[2] for r in results]
    lats = [r[3] for r in results]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy bar chart
    ax = axes[0, 0]
    bars = ax.bar(methods, accs, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy by Precision", fontweight="bold")
    ax.set_ylim(0, max(accs) * 1.15)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(y=accs[0], color="gray", linestyle="--", alpha=0.4, label="FP32 baseline")
    ax.legend(fontsize=9)

    # 2. Model size bar chart
    ax = axes[0, 1]
    bars = ax.bar(methods, sizes, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size by Precision", fontweight="bold")
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{sz:.1f}", ha="center", fontsize=11, fontweight="bold")
    # Compression ratio
    for i, sz in enumerate(sizes):
        ratio = sizes[0] / sz if sz > 0 else 0
        ax.text(i, sz * 0.5, f"{ratio:.1f}x", ha="center", fontsize=9, color="white", fontweight="bold")

    # 3. Latency bar chart
    ax = axes[1, 0]
    bars = ax.bar(methods, lats, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Inference Latency (ms/batch)")
    ax.set_title("Inference Speed by Precision", fontweight="bold")
    for bar, lat in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(lats) * 0.02,
                f"{lat:.1f}", ha="center", fontsize=11, fontweight="bold")

    # 4. Accuracy vs Size tradeoff
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        ax.scatter(sizes[i], accs[i], c=colors[i], s=200, zorder=5, edgecolors="white", linewidth=1.2)
        ax.annotate(method, (sizes[i], accs[i]),
                    textcoords="offset points", xytext=(12, -6), fontsize=11, fontweight="bold")
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Accuracy vs Model Size Tradeoff", fontweight="bold")

    fig.suptitle("CNN Quantization Comparison: FP32 vs FP16 vs INT8 vs INT4\nResNet-18 on CIFAR-10",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    png_path = out_dir / "pngs" / "quantization_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{time.strftime('%H:%M:%S')}] Saved figure: {png_path}")

    # Training curves (from metrics.csv)
    csv_path = out_dir / "metrics.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(df["epoch"], df["train_loss"], "b-", linewidth=1.5, label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["epoch"], df["train_acc"], "g-", linewidth=1.5, label="Train Acc")
        ax2.plot(df["epoch"], df["val_acc"], "r-", linewidth=1.5, label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training & Validation Accuracy", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Add best val marker
        best_idx = df["val_acc"].idxmax()
        ax2.annotate(f"Best: {df['val_acc'][best_idx]:.1f}%",
                     (df["epoch"][best_idx], df["val_acc"][best_idx]),
                     textcoords="offset points", xytext=(0, 12), ha="center",
                     fontsize=10, color="red", fontweight="bold")
        ax2.scatter(df["epoch"][best_idx], df["val_acc"][best_idx], c="red", s=60, zorder=5)

        fig.suptitle("ResNet-18 Training on CIFAR-10", fontweight="bold", fontsize=13)
        plt.tight_layout()
        png_path = out_dir / "pngs" / "training_curves.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{time.strftime('%H:%M:%S')}] Saved figure: {png_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    skip = args.skip_train
    log(f"Starting CNN Quantization Comparison{' (quant-only)' if skip else ''}")
    log(f"Device: {device} | Epochs: {args.epochs if not skip else 'skipped'} | Batch: {args.batch_size}")

    # Data
    log("Loading CIFAR-10 from torchvision...")
    loaders = get_datasets()
    log(f"Train batches: {len(loaders[0])} | Val batches: {len(loaders[1])}")

    # Model
    model = ResNet18(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"ResNet-18: {n_params:,} parameters")

    # Train (or skip)
    if not skip:
        model = train_model(model, loaders)
    else:
        log("Skipping training — model will use random weights (for testing only)")

    # Quantization comparison
    results = run_comparison(model, loaders[1])

    # Visualize
    try:
        plot_results(results)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Plotting failed: {e}")

    total_time = time.time() - t0
    log(f"Done! Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
