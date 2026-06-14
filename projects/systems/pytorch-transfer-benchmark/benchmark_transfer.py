"""Benchmark PyTorch CPU-GPU transfer throughput across configurations.

Tests CPU->GPU and GPU->CPU transfers with:
- Default .to() (no pinned memory)
- Pinned memory + .to() or copy_()
- non_blocking variants
- Multiple tensor sizes from 1 MB to 2 GB

Generates: logs/benchmark.log, metrics.csv, pngs/transfer_benchmark.png
"""

import torch
import time
import os
import sys
import json
import csv
from pathlib import Path

# --- Config ---
OUT_DIR = os.environ.get("OUT_DIR", "/content/pytorch-transfer-benchmark-output")
WARMUP_ITERS = 3
MEASURE_ITERS = 5

# Tensor sizes in bytes (1 MB to 2 GB, using float16 so nelement = bytes / 2)
SIZES_BYTES = [
    (1 * 1024**2, "1 MB"),
    (10 * 1024**2, "10 MB"),
    (100 * 1024**2, "100 MB"),
    (500 * 1024**2, "500 MB"),
    (1024**3, "1 GB"),
    (2 * 1024**3, "2 GB"),
]

# --- Output dirs ---
def setup_dirs(out_dir):
    for sub in ["logs", "pngs"]:
        Path(out_dir, sub).mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", buffering=1)
        self._ts_start = time.time()

    def log(self, msg):
        ts = time.time() - self._ts_start
        line = f"[{ts:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()


class MetricsCSV:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
        self._write_header()

    def _write_header(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def write_row(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(kwargs)


def measure_transfer(logger, name, fn, n_warmup=WARMUP_ITERS, n_measure=MEASURE_ITERS):
    """Measure throughput of fn() which transfers a tensor. Returns GB/s values."""
    # Warmup
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()

    # Measure
    times = []
    for i in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg_s = sum(times) / len(times)
    min_s = min(times)
    max_s = max(times)
    return avg_s, min_s, max_s


def create_tensor_cpu(size_bytes, dtype=torch.float16, pinned=False):
    """Create a CPU tensor of given byte size."""
    nelem = size_bytes // dtype.itemsize
    if pinned:
        return torch.empty(nelem, dtype=dtype, pin_memory=True)
    return torch.empty(nelem, dtype=dtype)


def run_benchmark(logger, csv_out):
    logger.log("=" * 60)
    logger.log("PyTorch CPU-GPU Transfer Benchmark")
    logger.log(f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}")
    logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.log(f"Warmup: {WARMUP_ITERS} iters  |  Measure: {MEASURE_ITERS} iters")
    logger.log("=" * 60)

    for size_bytes, size_label in SIZES_BYTES:
        size_gb = size_bytes / 1024**3
        nelem = size_bytes // 2  # float16 = 2 bytes

        # Skip if tensor won't fit (need ~3x: src CPU + src GPU + dst)
        vram_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        if size_bytes * 2 > vram_free * 0.85:
            logger.log(f"\n{'='*40}")
            logger.log(f"SIZE: {size_label} ({size_gb:.3f} GB) — SKIPPED (insufficient VRAM)")
            logger.log(f"  Free VRAM: {vram_free / 1024**3:.1f} GB, need ~{size_bytes * 2 / 1024**3:.1f} GB")
            continue

        logger.log(f"\n{'='*40}")
        logger.log(f"SIZE: {size_label} ({size_gb:.3f} GB)  |  nelem: {nelem:,}  |  dtype: float16")
        logger.log(f"{'='*40}")

        # --- CPU -> GPU ---
        logger.log("\n--- CPU -> GPU ---")

        # 1. Default .to('cuda')
        x_cpu = create_tensor_cpu(size_bytes, pinned=False)
        t0 = time.perf_counter()
        x_gpu = x_cpu.to("cuda")
        torch.cuda.synchronize()
        logger.log(f"  [Warmup] .to('cuda'): {(size_bytes / (time.perf_counter() - t0)) / 1e9:.2f} GB/s")
        del x_gpu, x_cpu

        x_cpu = create_tensor_cpu(size_bytes, pinned=False)
        avg_s, min_s, max_s = measure_transfer(
            logger, ".to('cuda')",
            lambda: x_cpu.to("cuda"),
        )
        gbps_default_c2g = size_bytes / avg_s / 1e9
        logger.log(f"  .to('cuda')           → {gbps_default_c2g:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="CPU->GPU", size_label=size_label, size_bytes=size_bytes,
            method="to(cuda)", pinned=False, non_blocking=False,
            throughput_gbps=round(gbps_default_c2g, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        gpu_ref = x_cpu.to("cuda")  # keep on GPU for GPU->CPU tests
        del x_cpu

        # 2. .to('cuda', non_blocking=True)
        x_cpu = create_tensor_cpu(size_bytes, pinned=False)
        avg_s, min_s, max_s = measure_transfer(
            logger, ".to('cuda', nb=True)",
            lambda: x_cpu.to("cuda", non_blocking=True),
        )
        gbps_nb_c2g = size_bytes / avg_s / 1e9
        logger.log(f"  .to('cuda', nb=True)   → {gbps_nb_c2g:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="CPU->GPU", size_label=size_label, size_bytes=size_bytes,
            method="to(cuda, nb=True)", pinned=False, non_blocking=True,
            throughput_gbps=round(gbps_nb_c2g, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        del x_cpu

        # 3. Pinned memory + .to('cuda')
        x_cpu_pinned = create_tensor_cpu(size_bytes, pinned=True)
        torch.cuda.synchronize()
        avg_s, min_s, max_s = measure_transfer(
            logger, "pinned .to('cuda')",
            lambda: x_cpu_pinned.to("cuda"),
        )
        gbps_pinned_c2g = size_bytes / avg_s / 1e9
        logger.log(f"  pinned .to('cuda')     → {gbps_pinned_c2g:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="CPU->GPU", size_label=size_label, size_bytes=size_bytes,
            method="pinned.to(cuda)", pinned=True, non_blocking=False,
            throughput_gbps=round(gbps_pinned_c2g, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )

        # 4. Pinned + .to('cuda', non_blocking=True)
        avg_s, min_s, max_s = measure_transfer(
            logger, "pinned .to('cuda', nb=True)",
            lambda: x_cpu_pinned.to("cuda", non_blocking=True),
        )
        gbps_pinned_nb_c2g = size_bytes / avg_s / 1e9
        logger.log(f"  pinned .to('cuda',nb)  → {gbps_pinned_nb_c2g:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="CPU->GPU", size_label=size_label, size_bytes=size_bytes,
            method="pinned.to(cuda, nb=True)", pinned=True, non_blocking=True,
            throughput_gbps=round(gbps_pinned_nb_c2g, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        del x_cpu_pinned

        # --- GPU -> CPU ---
        logger.log("--- GPU -> CPU ---")

        # 5. Default .to('cpu')
        x_gpu2 = gpu_ref.clone()
        avg_s, min_s, max_s = measure_transfer(
            logger, ".to('cpu')",
            lambda: x_gpu2.to("cpu"),
        )
        gbps_default_g2c = size_bytes / avg_s / 1e9
        logger.log(f"  .to('cpu')            → {gbps_default_g2c:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="GPU->CPU", size_label=size_label, size_bytes=size_bytes,
            method="to(cpu)", pinned=False, non_blocking=False,
            throughput_gbps=round(gbps_default_g2c, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        del x_gpu2

        # 6. .to('cpu', non_blocking=True)
        x_gpu2 = gpu_ref.clone()
        avg_s, min_s, max_s = measure_transfer(
            logger, ".to('cpu', nb=True)",
            lambda: x_gpu2.to("cpu", non_blocking=True),
        )
        gbps_nb_g2c = size_bytes / avg_s / 1e9
        logger.log(f"  .to('cpu', nb=True)    → {gbps_nb_g2c:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="GPU->CPU", size_label=size_label, size_bytes=size_bytes,
            method="to(cpu, nb=True)", pinned=False, non_blocking=True,
            throughput_gbps=round(gbps_nb_g2c, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        del x_gpu2

        # 7. Pinned destination + copy_()
        x_cpu_dst = create_tensor_cpu(size_bytes, pinned=True)
        x_gpu2 = gpu_ref.clone()
        torch.cuda.synchronize()
        avg_s, min_s, max_s = measure_transfer(
            logger, "pinned dst .copy_(gpu)",
            lambda: x_cpu_dst.copy_(x_gpu2),
        )
        gbps_pinned_g2c = size_bytes / avg_s / 1e9
        logger.log(f"  pinned .copy_(gpu)     → {gbps_pinned_g2c:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="GPU->CPU", size_label=size_label, size_bytes=size_bytes,
            method="pinned.copy_(gpu)", pinned=True, non_blocking=False,
            throughput_gbps=round(gbps_pinned_g2c, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        del x_gpu2

        # 8. Pinned destination + copy_(non_blocking=True)
        x_cpu_dst2 = create_tensor_cpu(size_bytes, pinned=True)
        x_gpu2 = gpu_ref.clone()
        torch.cuda.synchronize()
        avg_s, min_s, max_s = measure_transfer(
            logger, "pinned dst .copy_(gpu, nb=True)",
            lambda: x_cpu_dst2.copy_(x_gpu2, non_blocking=True),
        )
        gbps_pinned_nb_g2c = size_bytes / avg_s / 1e9
        logger.log(f"  pinned .copy_(gpu,nb)  → {gbps_pinned_nb_g2c:6.2f} GB/s  (avg {avg_s*1000:.1f} ms, min {min_s*1000:.1f}, max {max_s*1000:.1f})")
        csv_out.write_row(
            direction="GPU->CPU", size_label=size_label, size_bytes=size_bytes,
            method="pinned.copy_(gpu, nb=True)", pinned=True, non_blocking=True,
            throughput_gbps=round(gbps_pinned_nb_g2c, 3),
            avg_ms=round(avg_s * 1000, 2), min_ms=round(min_s * 1000, 2), max_ms=round(max_s * 1000, 2),
        )
        del x_gpu2, x_cpu_dst, x_cpu_dst2

        # --- Summary for this size ---
        speedup_g2c = gbps_pinned_g2c / gbps_default_g2c if gbps_default_g2c > 0 else 0
        logger.log(f"\n  >> GPU→CPU speedup (pinned vs default): {speedup_g2c:.1f}×")
        logger.log(f"  >> CPU→GPU default: {gbps_default_c2g:.1f} GB/s  |  GPU→CPU default: {gbps_default_g2c:.1f} GB/s  |  GPU→CPU pinned: {gbps_pinned_g2c:.1f} GB/s")

        del gpu_ref
        torch.cuda.empty_cache()

    logger.log("\n" + "=" * 60)
    logger.log("BENCHMARK COMPLETE")
    logger.log("=" * 60)


# --- Plotting ---
def make_plots(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Read CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["size_bytes"] = int(row["size_bytes"])
            row["throughput_gbps"] = float(row["throughput_gbps"])
            row["avg_ms"] = float(row["avg_ms"])
            row["pinned"] = row["pinned"] == "True"
            row["non_blocking"] = row["non_blocking"] == "True"
            rows.append(row)

    # Group by (direction, method, pinned, non_blocking)
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (row["direction"], row["method"], row["pinned"], row["non_blocking"])
        groups[key].append(row)

    sizes_ordered = sorted(set(r["size_bytes"] for r in rows))
    size_labels_ordered = [rows[0]["size_label"] for rows in zip(*[
        [r for r in rows if r["size_bytes"] == s] for s in sizes_ordered
    ])]
    # Simpler: just map from first occurrence
    size_label_map = {}
    for r in rows:
        if r["size_bytes"] not in size_label_map:
            size_label_map[r["size_bytes"]] = r["size_label"]
    size_labels_ordered = [size_label_map[s] for s in sizes_ordered]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- Panel 1: CPU->GPU throughput by method ---
    ax = axes[0, 0]
    methods_c2g = [
        ("to(cuda)", False, False),
        ("to(cuda, nb=True)", False, True),
        ("pinned.to(cuda)", True, False),
        ("pinned.to(cuda, nb=True)", True, True),
    ]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#27ae60"]
    markers = ["o", "s", "D", "^"]
    x = np.arange(len(sizes_ordered))
    width = 0.2

    for i, (method, pinned, nb) in enumerate(methods_c2g):
        key = ("CPU->GPU", method, pinned, nb)
        vals = groups.get(key, [])
        val_map = {r["size_bytes"]: r["throughput_gbps"] for r in vals}
        y = [val_map.get(s, 0) for s in sizes_ordered]
        bars = ax.bar(x + i * width, y, width, label=method, color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("CPU → GPU Transfer Throughput")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(size_labels_ordered)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 2: GPU->CPU throughput by method ---
    ax = axes[0, 1]
    methods_g2c = [
        ("to(cpu)", False, False),
        ("to(cpu, nb=True)", False, True),
        ("pinned.copy_(gpu)", True, False),
        ("pinned.copy_(gpu, nb=True)", True, True),
    ]
    colors_g2c = ["#e74c3c", "#e67e22", "#2ecc71", "#27ae60"]
    for i, (method, pinned, nb) in enumerate(methods_g2c):
        key = ("GPU->CPU", method, pinned, nb)
        vals = groups.get(key, [])
        val_map = {r["size_bytes"]: r["throughput_gbps"] for r in vals}
        y = [val_map.get(s, 0) for s in sizes_ordered]
        ax.bar(x + i * width, y, width, label=method, color=colors_g2c[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("GPU → CPU Transfer Throughput")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(size_labels_ordered)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 3: Speedup factor (pinned / default) ---
    ax = axes[1, 0]
    speedups = []
    for s in sizes_ordered:
        def_g2c = None
        pin_g2c = None
        for r in rows:
            if r["size_bytes"] == s and r["direction"] == "GPU->CPU":
                if r["method"] == "to(cpu)" and not r["pinned"] and not r["non_blocking"]:
                    def_g2c = r["throughput_gbps"]
                if r["method"] == "pinned.copy_(gpu)" and r["pinned"] and not r["non_blocking"]:
                    pin_g2c = r["throughput_gbps"]
        speedups.append(pin_g2c / def_g2c if def_g2c and pin_g2c else 0)

    bars = ax.bar(x, speedups, width * 3, color=["#3498db" if v >= 1 else "#e74c3c" for v in speedups], edgecolor="white")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="No speedup (1×)")
    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Speedup Factor")
    ax.set_title("GPU→CPU Pinned Memory Speedup (pinned / default)")
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels_ordered)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, speedups):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}×", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # --- Panel 4: Latency comparison (GPU->CPU) ---
    ax = axes[1, 1]
    for i, (method, pinned, nb) in enumerate(methods_g2c):
        key = ("GPU->CPU", method, pinned, nb)
        vals = groups.get(key, [])
        val_map = {r["size_bytes"]: r["avg_ms"] for r in vals}
        y = [val_map.get(s, 0) for s in sizes_ordered]
        ax.plot(x, y, marker=markers[i] if i < len(markers) else "o", label=method,
                color=colors_g2c[i], linewidth=2, markersize=6)

    ax.set_xlabel("Tensor Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("GPU → CPU Transfer Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels_ordered)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    plt.suptitle("PyTorch CPU-GPU Transfer Benchmark\n"
                 f"{torch.cuda.get_device_name(0)}  |  PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "pngs", "transfer_benchmark.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[plot] Saved {plot_path}")


# --- Summary JSON ---
def write_summary(csv_path, out_dir):
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            row["size_bytes"] = int(row["size_bytes"])
            row["throughput_gbps"] = float(row["throughput_gbps"])
            rows.append(row)

    # Find key results
    largest_size = max(r["size_bytes"] for r in rows)
    largest_rows = [r for r in rows if r["size_bytes"] == largest_size]

    def_g2c = next((r for r in largest_rows if r["direction"] == "GPU->CPU" and r["method"] == "to(cpu)" and r["pinned"] == "False"), None)
    pin_g2c = next((r for r in largest_rows if r["direction"] == "GPU->CPU" and r["method"] == "pinned.copy_(gpu)" and r["pinned"] == "True"), None)
    def_c2g = next((r for r in largest_rows if r["direction"] == "CPU->GPU" and r["method"] == "to(cuda)" and r["pinned"] == "False"), None)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "largest_tensor_size_bytes": largest_size,
        "largest_tensor_size_label": largest_rows[0]["size_label"],
        "gpu_to_cpu_default_gbps": float(def_g2c["throughput_gbps"]) if def_g2c else None,
        "gpu_to_cpu_pinned_gbps": float(pin_g2c["throughput_gbps"]) if pin_g2c else None,
        "gpu_to_cpu_speedup": round(float(pin_g2c["throughput_gbps"]) / float(def_g2c["throughput_gbps"]), 1) if def_g2c and pin_g2c else None,
        "cpu_to_gpu_default_gbps": float(def_c2g["throughput_gbps"]) if def_c2g else None,
        "pcie_theoretical_gbps": 15.75,  # PCIe 3.0 x16
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] Saved {summary_path}")
    return summary


def main():
    out_dir = OUT_DIR
    setup_dirs(out_dir)

    logger = Logger(os.path.join(out_dir, "logs", "benchmark.log"))
    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_out = MetricsCSV(csv_path, [
        "direction", "size_label", "size_bytes",
        "method", "pinned", "non_blocking",
        "throughput_gbps", "avg_ms", "min_ms", "max_ms",
    ])

    try:
        run_benchmark(logger, csv_out)
    finally:
        logger.close()

    # Generate plots and summary
    make_plots(csv_path, out_dir)
    summary = write_summary(csv_path, out_dir)

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    if summary["gpu_to_cpu_speedup"]:
        print(f"  GPU→CPU default:  {summary['gpu_to_cpu_default_gbps']:.1f} GB/s")
        print(f"  GPU→CPU pinned:   {summary['gpu_to_cpu_pinned_gbps']:.1f} GB/s")
        print(f"  GPU→CPU speedup:  {summary['gpu_to_cpu_speedup']}×")
    print(f"  CPU→GPU default:  {summary['cpu_to_gpu_default_gbps']:.1f} GB/s")
    print(f"  PCIe theoretical: {summary['pcie_theoretical_gbps']:.1f} GB/s (PCIe 3.0 x16)")
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
