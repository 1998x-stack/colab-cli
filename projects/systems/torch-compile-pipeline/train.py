"""Benchmark torch.compile backends: eager, aot_eager, inductor.

Measures compile time, throughput, and GPU memory per backend to
quantify AOTAutograd's contribution in the compilation pipeline.

Pipeline stages:
  eager     — Dynamo graph capture only (no autograd, no kernel fusion)
  aot_eager — Dynamo + AOTAutograd (traced backward, no Triton kernels)
  inductor  — Full stack: Dynamo + AOTAutograd + Triton kernel compilation

Output: /content/torch-compile-pipeline-output/
"""

import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/torch-compile-pipeline-output"
BATCH_SIZE = 64
INPUT_SHAPE = (3, 64, 64)  # C, H, W
NUM_CLASSES = 10
WARMUP_ITERS = 10
MEASURE_ITERS = 100
GRAD_CLIP = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Logging ─────────────────────────────────────────────────────────────────
def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)


# ── Model ───────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(c_in, c_out, stride),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
        )
        self.downsample = None
        if stride != 1 or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride),
                nn.BatchNorm2d(c_out),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.conv(x)
        return self.relu(out + identity)


class BenchmarkCNN(nn.Module):
    """A non-trivial CNN with residual blocks — enough depth to stress the compiler."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ── Benchmark helpers ───────────────────────────────────────────────────────
def reset_memory_stats():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def benchmark_forward(model, x):
    """Time forward pass only — measures Dynamo overhead."""
    reset_memory_stats()

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = model(x)
    torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    mem_mb = get_peak_memory_mb()
    throughput = (BATCH_SIZE * MEASURE_ITERS) / elapsed
    return elapsed, throughput, mem_mb


def benchmark_forward_backward(model, x, optimizer, criterion):
    """Time forward + backward + optimizer step — full training iteration."""
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
    reset_memory_stats()

    # Warmup
    for _ in range(WARMUP_ITERS):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    mem_mb = get_peak_memory_mb()
    throughput = (BATCH_SIZE * MEASURE_ITERS) / elapsed
    return elapsed, throughput, mem_mb


def benchmark_compile(model, x, backend_name):
    """Apply torch.compile with given backend, measure compile time + runtime."""
    log(f"  Compiling with backend={backend_name} ...")
    t0 = time.perf_counter()
    try:
        compiled = torch.compile(model, backend=backend_name)
    except Exception as e:
        log(f"  ERROR during compile: {e}")
        return None, None

    # Trigger compilation (first forward pass)
    _ = compiled(x)
    torch.cuda.synchronize()
    compile_trigger_time = time.perf_counter() - t0
    log(f"  Compile + trigger: {compile_trigger_time:.3f}s")

    return compiled, compile_trigger_time


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    log(f"Device: {DEVICE}")
    if DEVICE.type != "cuda":
        log("ERROR: CUDA not available — aborting")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"GPU: {gpu_name} ({gpu_mem_total:.1f} GB)")

    # Shared inputs
    x = torch.randn(BATCH_SIZE, *INPUT_SHAPE, device=DEVICE)
    criterion = nn.CrossEntropyLoss()

    results = []
    backends = ["eager (no compile)", "eager", "aot_eager", "inductor"]

    for label in backends:
        log(f"\n{'='*60}")
        log(f"Backend: {label}")
        log(f"{'='*60}")

        # Fresh model each time
        model = BenchmarkCNN().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        n_params = sum(p.numel() for p in model.parameters())

        if label == "eager (no compile)":
            compiled = model  # raw PyTorch eager
            compile_time = 0.0
        else:
            compiled, compile_time = benchmark_compile(model, x, label)
            if compiled is None:
                continue

        # Forward-only benchmark
        log("  Benchmarking forward pass ...")
        fwd_elapsed, fwd_throughput, fwd_mem = benchmark_forward(compiled, x)
        log(f"    Fwd: {fwd_elapsed:.3f}s total, {fwd_throughput:.0f} samples/s, peak={fwd_mem:.1f} MB")

        # Forward + backward benchmark
        log("  Benchmarking forward+backward pass ...")
        fb_elapsed, fb_throughput, fb_mem = benchmark_forward_backward(compiled, x, optimizer, criterion)
        log(f"    Fwd+Bwd: {fb_elapsed:.3f}s total, {fb_throughput:.0f} samples/s, peak={fb_mem:.1f} MB")

        results.append({
            "backend": label,
            "params": n_params,
            "compile_time_s": round(compile_time, 4),
            "fwd_elapsed_s": round(fwd_elapsed, 4),
            "fwd_throughput": round(fwd_throughput, 0),
            "fwd_mem_mb": round(fwd_mem, 1),
            "fb_elapsed_s": round(fb_elapsed, 4),
            "fb_throughput": round(fb_throughput, 0),
            "fb_mem_mb": round(fb_mem, 1),
        })

    # ── Baseline-relative speedups ──────────────────────────────────────────
    baseline = results[0]
    for r in results:
        if baseline["fwd_throughput"] > 0:
            r["fwd_speedup"] = round(r["fwd_throughput"] / baseline["fwd_throughput"], 2)
        if baseline["fb_throughput"] > 0:
            r["fb_speedup"] = round(r["fb_throughput"] / baseline["fb_throughput"], 2)

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    fieldnames = [
        "backend", "params", "compile_time_s",
        "fwd_elapsed_s", "fwd_throughput", "fwd_mem_mb", "fwd_speedup",
        "fb_elapsed_s", "fb_throughput", "fb_mem_mb", "fb_speedup",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    log(f"\nSaved: {csv_path}")

    # ── Log summary ─────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("Summary")
    log(f"{'='*60}")
    for r in results:
        log(f"  {r['backend']:<22s} | compile={r['compile_time_s']:>6.2f}s | "
            f"fwd={r['fwd_throughput']:>7.0f} samp/s ({r.get('fwd_speedup','-')}x) | "
            f"fb={r['fb_throughput']:>7.0f} samp/s ({r.get('fb_speedup','-')}x) | "
            f"mem={r['fb_mem_mb']:>6.0f} MB")

    # ── PNG visualization ───────────────────────────────────────────────────
    log("Generating comparison plots ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["backend"] for r in results]
    x_pos = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#6c757d", "#0d6efd", "#6f42c1", "#d63384"]

    # 1. Forward throughput
    ax = axes[0][0]
    bars = ax.bar(x_pos, [r["fwd_throughput"] for r in results], color=colors, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Samples / second")
    ax.set_title("Forward Pass Throughput")
    ax.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, results):
        sp = r.get("fwd_speedup", 1.0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{sp:.2f}x", ha="center", fontsize=10, fontweight="bold")

    # 2. Forward+backward throughput
    ax = axes[0][1]
    bars = ax.bar(x_pos, [r["fb_throughput"] for r in results], color=colors, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Samples / second")
    ax.set_title("Forward + Backward Throughput")
    ax.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, results):
        sp = r.get("fb_speedup", 1.0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{sp:.2f}x", ha="center", fontsize=10, fontweight="bold")

    # 3. Peak GPU memory (fwd+bwd)
    ax = axes[1][0]
    bars = ax.bar(x_pos, [r["fb_mem_mb"] for r in results], color=colors, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory — Forward + Backward")
    ax.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{r['fb_mem_mb']:.0f} MB", ha="center", fontsize=9)

    # 4. Compile time
    ax = axes[1][1]
    compile_times = [r["compile_time_s"] for r in results]
    bars = ax.bar(x_pos, compile_times, color=colors, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Compile Time (s)")
    ax.set_title("First-call Compilation Overhead")
    ax.grid(axis="y", alpha=0.3)
    for bar, t in zip(bars, compile_times):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{t:.2f}s", ha="center", fontsize=9)

    fig.suptitle(f"torch.compile Pipeline Analysis — {gpu_name}\n"
                 f"Model: {n_params:,} params  |  Input: {BATCH_SIZE}x{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}x{INPUT_SHAPE[2]}  |  "
                 f"Warmup={WARMUP_ITERS}  Measure={MEASURE_ITERS}",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "compile_pipeline_comparison.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {png_path}")

    # ── Pipeline stage breakdown (annotated diagram) ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))

    stages = [
        ("Raw PyTorch\n(eager)", "torch.nn.Module\nforward()", "autograd\nbackward()", "CUDA kernels\n(cuDNN/cuBLAS)"),
        ("Dynamo\n(backend=eager)", "Dynamo\nFX graph capture", "autograd\nbackward()", "CUDA kernels\n(cuDNN/cuBLAS)"),
        ("Dynamo + AOTAutograd\n(backend=aot_eager)", "Dynamo\nFX graph capture", "AOTAutograd\ntraced backward", "CUDA kernels\n(cuDNN/cuBLAS)"),
        ("Full Inductor\n(backend=inductor)", "Dynamo\nFX graph capture", "AOTAutograd\ntraced backward", "Triton kernels\n(fused + compiled)"),
    ]

    # Draw pipeline boxes
    y = 0
    for i, (label, s1, s2, s3) in enumerate(stages):
        colors_i = ["#e9ecef", "#dee2e6", "#ced4da", "#adb5bd"]
        ax.add_patch(plt.Rectangle((0.1, y + 0.1), 2.6, 0.9, fill=True, facecolor=colors_i[0], edgecolor="gray", lw=1))
        ax.add_patch(plt.Rectangle((2.8, y + 0.1), 2.6, 0.9, fill=True, facecolor=colors_i[1], edgecolor="gray", lw=1))
        ax.add_patch(plt.Rectangle((5.5, y + 0.1), 2.6, 0.9, fill=True, facecolor=colors_i[2], edgecolor="gray", lw=1))

        ax.text(1.4, y + 0.55, s1, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(4.1, y + 0.55, s2, ha="center", va="center", fontsize=8)
        ax.text(6.8, y + 0.55, s3, ha="center", va="center", fontsize=8)

        ax.text(0.0, y + 0.55, label, ha="right", va="center", fontsize=9, fontweight="bold")

        # Arrows between stages
        for x_arrow in [2.75, 5.45]:
            ax.annotate("", xy=(x_arrow + 0.05, y + 0.55), xytext=(x_arrow - 0.1, y + 0.55),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

        # Throughput annotation on right
        r = results[i]
        ax.text(8.5, y + 0.55, f"{r['fb_throughput']:.0f} samp/s\n({r.get('fb_speedup', 1.0):.2f}x)",
                ha="left", va="center", fontsize=8, fontweight="bold", color=colors[i])

        y += 1.1

    ax.set_xlim(-2.5, 11)
    ax.set_ylim(-0.2, y)
    ax.axis("off")
    ax.set_title("torch.compile Pipeline Stages — What each backend runs", fontsize=12, fontweight="bold", y=1.02)

    diagram_path = os.path.join(OUTPUT_DIR, "pipeline_stages.png")
    plt.savefig(diagram_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved: {diagram_path}")

    log("\nDone. All artifacts in {OUTPUT_DIR}/")
    log("  - metrics.csv")
    log("  - compile_pipeline_comparison.png")
    log("  - pipeline_stages.png")


if __name__ == "__main__":
    main()
