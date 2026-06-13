"""
Parse nanochat training log and generate plots. Also packages all outputs for download.

Run via: cb exec -f visualize.py --timeout 60

Outputs:
  /content/plots/          — PNG visualizations
  /content/nanochat-output.tar.gz  — all artifacts for download
"""

import os
import sys
import re
import subprocess
import tarfile

LOG_FILE = "/content/train.log"
PLOTS_DIR = "/content/plots"
BASE_DIR = "/content/nanochat-data"
OUTPUT_TAR = "/content/nanochat-output.tar.gz"

# ── Ensure matplotlib ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# ── Parse log ──────────────────────────────────────────────────────
print(f"[visualize] Parsing {LOG_FILE}...")

if not os.path.exists(LOG_FILE):
    print(f"ERROR: {LOG_FILE} not found")
    sys.exit(1)

lines = open(LOG_FILE).readlines()
print(f"[visualize] {len(lines)} lines read")

# Extract step metrics
step_pattern = re.compile(
    r"step\s+(\d+)/(\d+)\s+\(([\d.]+)%\)\s*\|\s*"
    r"loss:\s*([\d.]+)\s*\|\s*"
    r"lrm:\s*([\d.]+)\s*\|\s*"
    r"dt:\s*([\d.]+)ms\s*\|\s*"
    r"tok/sec:\s*([\d,]+)\s*\|\s*"
    r"bf16_mfu:\s*([\d.]+)"
)

val_pattern = re.compile(r"Step\s+(\d+)\s*\|\s*Validation bpb:\s*([\d.]+)")
core_pattern = re.compile(r"Step\s+(\d+)\s*\|\s*CORE metric:\s*([\d.]+)")
min_val_pattern = re.compile(r"Minimum validation bpb:\s*([\d.]+)")
peak_mem_pattern = re.compile(r"Peak memory usage:\s*([\d.]+)MiB")
total_time_pattern = re.compile(r"Total training time:\s*([\d.]+)m")

steps, losses, lrms, dts, tok_per_sec, mfus = [], [], [], [], [], []
val_steps, val_bpbs = [], []
core_steps, core_metrics = [], []
min_val_bpb, peak_mem, total_time = None, None, None

for line in lines:
    m = step_pattern.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(4)))
        lrms.append(float(m.group(5)))
        dts.append(float(m.group(6)))
        tok_per_sec.append(int(m.group(7).replace(",", "")))
        mfus.append(float(m.group(8)))
        continue

    m = val_pattern.search(line)
    if m:
        val_steps.append(int(m.group(1)))
        val_bpbs.append(float(m.group(2)))
        continue

    m = core_pattern.search(line)
    if m:
        core_steps.append(int(m.group(1)))
        core_metrics.append(float(m.group(2)))
        continue

    m = min_val_pattern.search(line)
    if m:
        min_val_bpb = float(m.group(1))

    m = peak_mem_pattern.search(line)
    if m:
        peak_mem = float(m.group(1))

    m = total_time_pattern.search(line)
    if m:
        total_time = float(m.group(1))

print(f"[visualize] Parsed {len(steps)} training steps, {len(val_steps)} validation points")
if min_val_bpb:
    print(f"[visualize] Min val bpb: {min_val_bpb:.6f}")
if total_time:
    print(f"[visualize] Total training time: {total_time:.1f}m")

if not steps:
    print("ERROR: No training steps found in log. Training may not have started.")
    sys.exit(1)

# ── Generate plots ─────────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")

# Colors
C_TRAIN = "#2196F3"
C_VAL = "#FF5722"
C_TOK = "#4CAF50"
C_MFU = "#9C27B0"

# Plot 1: Training & validation loss
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(steps, losses, color=C_TRAIN, linewidth=0.8, alpha=0.7, label="Train loss (per step)")
# Smooth
if len(losses) > 10:
    import numpy as np
    window = min(50, len(losses) // 5)
    if window > 2:
        kernel = np.ones(window) / window
        smooth = np.convolve(losses, kernel, mode="valid")
        ax1.plot(steps[window-1:], smooth, color=C_TRAIN, linewidth=2, label=f"Train loss (MA-{window})")

ax1.set_ylabel("Training Loss", fontsize=12, color=C_TRAIN)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(val_steps, val_bpbs, "o-", color=C_VAL, linewidth=2, markersize=4, label="Validation BPB")
if min_val_bpb:
    ax2.axhline(y=min_val_bpb, color=C_VAL, linestyle="--", alpha=0.5, label=f"Min: {min_val_bpb:.4f}")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Validation BPB", fontsize=12, color=C_VAL)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle(f"nanochat d=6 Training on Colab T4\nTotal time: {total_time:.1f}m | Min val BPB: {min_val_bpb:.4f}" if total_time and min_val_bpb
             else "nanochat d=6 Training on Colab T4", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "loss.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[visualize] Saved loss.png")

# Plot 2: Throughput & MFU
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(steps, tok_per_sec, color=C_TOK, linewidth=1.5, label="Tokens/sec")
ax1.set_ylabel("Tokens/sec", fontsize=12, color=C_TOK)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(steps, mfus, color=C_MFU, linewidth=1.5, label="MFU %")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("MFU %", fontsize=12, color=C_MFU)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("nanochat Training Throughput", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "throughput.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[visualize] Saved throughput.png")

# Plot 3: Step time & LR schedule
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(steps, dts, color="#FF9800", linewidth=1, label="Step time (ms)")
ax1.set_ylabel("Step Time (ms)", fontsize=12)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(steps, lrms, color="#E91E63", linewidth=1.5, label="LR multiplier")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("LR Multiplier", fontsize=12)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("nanochat Step Timing & LR Schedule", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "timing.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[visualize] Saved timing.png")

# Plot 4: Combined dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

ax_loss = fig.add_subplot(gs[0, :2])
ax_val = fig.add_subplot(gs[1, :2])
ax_tok = fig.add_subplot(gs[2, :2])
ax_info = fig.add_subplot(gs[:, 2])

ax_loss.plot(steps, losses, color=C_TRAIN, linewidth=0.6, alpha=0.6)
if len(losses) > 10:
    ax_loss.plot(steps[window-1:], smooth, color=C_TRAIN, linewidth=1.5)
ax_loss.set_ylabel("Loss", fontsize=10)
ax_loss.set_title("Training Loss", fontsize=11, fontweight="bold")
ax_loss.grid(True, alpha=0.3)

ax_val.plot(val_steps, val_bpbs, "o-", color=C_VAL, linewidth=1.5, markersize=3)
if min_val_bpb:
    ax_val.axhline(y=min_val_bpb, color=C_VAL, linestyle="--", alpha=0.5)
ax_val.set_xlabel("Step", fontsize=10)
ax_val.set_ylabel("Val BPB", fontsize=10)
ax_val.set_title("Validation BPB", fontsize=11, fontweight="bold")
ax_val.grid(True, alpha=0.3)

ax_tok.plot(steps, tok_per_sec, color=C_TOK, linewidth=1)
ax_tok.set_xlabel("Step", fontsize=10)
ax_tok.set_ylabel("Tok/sec", fontsize=10)
ax_tok.set_title("Training Throughput", fontsize=11, fontweight="bold")
ax_tok.grid(True, alpha=0.3)

info_lines = [
    "── nanochat T4 Run ──",
    "",
    "Model: d=6, head-dim=64",
    "Seq len: 256",
    "Batch size: 1",
    "",
    f"Steps: {steps[-1]}",
    f"Total time: {total_time:.1f}m" if total_time else "",
    f"Peak VRAM: {peak_mem:.0f} MiB" if peak_mem else "",
    "",
    f"Final loss: {losses[-1]:.4f}",
    f"Min val BPB: {min_val_bpb:.4f}" if min_val_bpb else "",
    f"Final tok/sec: {tok_per_sec[-1]:,}",
    f"Peak tok/sec: {max(tok_per_sec):,}",
    f"Avg MFU: {sum(mfus)/len(mfus):.2f}%",
]
info_text = "\n".join(l for l in info_lines if l is not None)
ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
ax_info.set_xticks([])
ax_info.set_yticks([])

fig.suptitle("nanochat Training Dashboard — Colab T4", fontsize=15, fontweight="bold")
plt.savefig(os.path.join(PLOTS_DIR, "dashboard.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[visualize] Saved dashboard.png")

# ── Package outputs ────────────────────────────────────────────────
print(f"\n[visualize] Packaging outputs to {OUTPUT_TAR}...")
with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
    # plots
    for fn in sorted(os.listdir(PLOTS_DIR)):
        tar.add(os.path.join(PLOTS_DIR, fn), arcname=f"plots/{fn}")
        print(f"  + plots/{fn}")
    # log
    tar.add(LOG_FILE, arcname="train.log")
    print("  + train.log")

    # checkpoints (if any)
    ckpt_dir = os.path.join(BASE_DIR, "base_checkpoints", "d6")
    if os.path.exists(ckpt_dir):
        for item in sorted(os.listdir(ckpt_dir)):
            path = os.path.join(ckpt_dir, item)
            tar.add(path, arcname=f"checkpoints/d6/{item}")
        print("  + checkpoints/d6/")

    # tokenizer
    tok_dir = os.path.join(BASE_DIR, "tokenizer")
    if os.path.exists(tok_dir):
        for item in sorted(os.listdir(tok_dir)):
            path = os.path.join(tok_dir, item)
            tar.add(path, arcname=f"tokenizer/{item}")
        print("  + tokenizer/")

size_mb = os.path.getsize(OUTPUT_TAR) / (1024 * 1024)
print(f"\n[visualize] Done: {OUTPUT_TAR} ({size_mb:.1f} MB)")
print(f"[visualize] To download: cb download {OUTPUT_TAR} ./nanochat-output.tar.gz")
print(f"[visualize] Or from local: cd projects/nanochat-colab && cb download {OUTPUT_TAR} ./")
