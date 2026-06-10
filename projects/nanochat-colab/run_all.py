"""
nanochat end-to-end Colab run — setup, train, visualize, package.

Run via: cb run --gpu T4 run_all.py

Everything runs inline so the VM stays alive until completion.
Estimated total time: ~15 minutes (setup ~3min + training ~10min + plots ~1min)
"""

import subprocess, sys, os, time, shutil, re, tarfile

NANOCHAT_DIR = "/content/nanochat"
LOG_FILE = "/content/train.log"
BASE_DIR = "/content/nanochat-data"
PLOTS_DIR = "/content/plots"
OUTPUT_TAR = "/content/nanochat-output.tar.gz"

# ── Step 0: System info ────────────────────────────────────────────
print("=" * 60)
print("[setup] System info")
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True)
print(f"  GPU: {result.stdout.strip()}")
print(f"  Python: {sys.version.split()[0]}")
print("=" * 60)

# ── Step 1: Install uv ─────────────────────────────────────────────
if not shutil.which("uv"):
    print("[setup] Installing uv...")
    subprocess.check_call(["curl", "-LsSf", "https://astral.sh/uv/install.sh"], stdout=subprocess.DEVNULL)
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")

# ── Step 2: Clone nanochat ─────────────────────────────────────────
if not os.path.exists(NANOCHAT_DIR):
    print("[setup] Cloning nanochat...")
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/karpathy/nanochat.git", NANOCHAT_DIR])

# ── Step 3: uv sync GPU deps ───────────────────────────────────────
print("[setup] uv sync --extra gpu...")
t0 = time.time()
subprocess.check_call(["uv", "sync", "--extra", "gpu"], cwd=NANOCHAT_DIR)
print(f"[setup] uv sync done in {time.time() - t0:.1f}s")

# ── Step 4: Dataset + tokenizer ────────────────────────────────────
os.makedirs(BASE_DIR, exist_ok=True)
os.environ["NANOCHAT_BASE_DIR"] = BASE_DIR
venv_python = os.path.join(NANOCHAT_DIR, ".venv", "bin", "python")

print("[setup] Downloading 5 ClimbMix shards...")
subprocess.check_call([venv_python, "-m", "nanochat.dataset", "-n", "5"], cwd=NANOCHAT_DIR)

print("[setup] Training BPE tokenizer (500M chars)...")
subprocess.check_call(
    [venv_python, "-m", "scripts.tok_train", "--max-chars=500000000"],
    cwd=NANOCHAT_DIR,
)

# ── Step 5: Train inline ───────────────────────────────────────────
print("[train] Starting base_train (running inline, ~10 min)...")

train_args = [
    venv_python, "-u", "-m", "scripts.base_train",
    "--depth=6",
    "--head-dim=64",
    "--max-seq-len=256",
    "--device-batch-size=1",
    "--total-batch-size=16384",
    "--num-iterations=250",
    "--eval-every=50",
    "--eval-tokens=16384",
    "--core-metric-every=-1",
    "--sample-every=-1",
    "--save-every=-1",
    "--window-pattern=L",
    "--warmup-steps=20",
    "--final-lr-frac=0.1",
    "--target-param-data-ratio=-1",
    "--run=dummy",
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["NANOCHAT_BASE_DIR"] = BASE_DIR
env["NANOCHAT_DTYPE"] = "float16"

t0 = time.time()
with open(LOG_FILE, "w") as f:
    result = subprocess.run(
        train_args,
        stdout=f, stderr=subprocess.STDOUT,
        cwd=NANOCHAT_DIR,
        env=env,
    )
train_time = time.time() - t0
print(f"[train] Done in {train_time/60:.1f}m (exit={result.returncode})")

# ── Step 6: Generate plots ─────────────────────────────────────────
print("[visualize] Generating plots...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs(PLOTS_DIR, exist_ok=True)
lines = open(LOG_FILE).readlines()

# Parse training log
step_pat = re.compile(
    r"step\s+(\d+)/(\d+)\s+\(([\d.]+)%\)\s*\|\s*"
    r"loss:\s*([\d.]+)\s*\|\s*"
    r"lrm:\s*([\d.]+)\s*\|\s*"
    r"dt:\s*([\d.]+)ms\s*\|\s*"
    r"tok/sec:\s*([\d,]+)\s*\|\s*"
    r"bf16_mfu:\s*([\d.]+)"
)
val_pat = re.compile(r"Step\s+(\d+)\s*\|\s*Validation bpb:\s*([\d.]+)")
min_val_pat = re.compile(r"Minimum validation bpb:\s*([\d.]+)")
peak_mem_pat = re.compile(r"Peak memory usage:\s*([\d.]+)MiB")
total_time_pat = re.compile(r"Total training time:\s*([\d.]+)m")

steps, losses, lrms, dts, tok_per_sec, mfus = [], [], [], [], [], []
val_steps, val_bpbs = [], []
min_val_bpb = peak_mem = total_time_m = None

for line in lines:
    m = step_pat.search(line)
    if m:
        steps.append(int(m.group(1))); losses.append(float(m.group(4)))
        lrms.append(float(m.group(5))); dts.append(float(m.group(6)))
        tok_per_sec.append(int(m.group(7).replace(",", ""))); mfus.append(float(m.group(8)))
        continue
    m = val_pat.search(line)
    if m:
        val_steps.append(int(m.group(1))); val_bpbs.append(float(m.group(2)))
        continue
    m = min_val_pat.search(line)
    if m: min_val_bpb = float(m.group(1))
    m = peak_mem_pat.search(line)
    if m: peak_mem = float(m.group(1))
    m = total_time_pat.search(line)
    if m: total_time_m = float(m.group(1))

print(f"[visualize] Parsed {len(steps)} steps, {len(val_steps)} val points")
if not steps:
    print("[visualize] WARNING: No training steps found!")
else:
    # Dashboard plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :2])
    ax_val = fig.add_subplot(gs[1, :2])
    ax_tok = fig.add_subplot(gs[2, :2])
    ax_info = fig.add_subplot(gs[:, 2])

    ax_loss.plot(steps, losses, color="#2196F3", linewidth=0.6, alpha=0.6)
    if len(losses) > 10:
        w = min(50, len(losses) // 5)
        if w > 2:
            smooth = np.convolve(losses, np.ones(w)/w, mode="valid")
            ax_loss.plot(steps[w-1:], smooth, color="#2196F3", linewidth=1.5)
    ax_loss.set_ylabel("Loss"); ax_loss.set_title("Training Loss", fontweight="bold")
    ax_loss.grid(True, alpha=0.3)

    ax_val.plot(val_steps, val_bpbs, "o-", color="#FF5722", linewidth=1.5, markersize=3)
    if min_val_bpb:
        ax_val.axhline(y=min_val_bpb, color="#FF5722", linestyle="--", alpha=0.5)
    ax_val.set_xlabel("Step"); ax_val.set_ylabel("Val BPB")
    ax_val.set_title("Validation BPB", fontweight="bold"); ax_val.grid(True, alpha=0.3)

    ax_tok.plot(steps, tok_per_sec, color="#4CAF50", linewidth=1)
    ax_tok.set_xlabel("Step"); ax_tok.set_ylabel("Tok/sec")
    ax_tok.set_title("Training Throughput", fontweight="bold"); ax_tok.grid(True, alpha=0.3)

    info = [
        "── nanochat d=6 on T4 ──", "",
        f"Steps: {steps[-1]}", f"Total time: {total_time_m:.1f}m" if total_time_m else "",
        f"Peak VRAM: {peak_mem:.0f} MiB" if peak_mem else "",
        f"", f"Final loss: {losses[-1]:.4f}",
        f"Min val BPB: {min_val_bpb:.4f}" if min_val_bpb else "",
        f"Avg tok/sec: {sum(tok_per_sec)//len(tok_per_sec):,}",
        f"Final tok/sec: {tok_per_sec[-1]:,}",
    ]
    info_text = "\n".join(l for l in info if l is not None)
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax_info.set_xticks([]); ax_info.set_yticks([])

    fig.suptitle(f"nanochat Training — Colab T4 ({train_time/60:.1f}m total)", fontsize=15, fontweight="bold")
    fig.savefig(os.path.join(PLOTS_DIR, "dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[visualize] Saved dashboard.png")

    # Loss + val plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(steps, losses, color="#2196F3", linewidth=0.6, alpha=0.6)
    if len(losses) > 10 and 'smooth' in dir():
        ax1.plot(steps[w-1:], smooth, color="#2196F3", linewidth=2, label=f"MA-{w}")
    ax1.set_ylabel("Training Loss"); ax1.legend(loc="upper right"); ax1.grid(True, alpha=0.3)
    ax2.plot(val_steps, val_bpbs, "o-", color="#FF5722", linewidth=2, markersize=4)
    if min_val_bpb:
        ax2.axhline(y=min_val_bpb, color="#FF5722", linestyle="--", alpha=0.5, label=f"Min: {min_val_bpb:.4f}")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Validation BPB"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle("nanochat Loss Curves", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(PLOTS_DIR, "loss.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[visualize] Saved loss.png")

# ── Step 7: Package outputs (skip checkpoints — too large for proxy download) ──
print(f"[package] Creating {OUTPUT_TAR} (log + plots only, no checkpoints)...")
with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
    for fn in sorted(os.listdir(PLOTS_DIR)):
        tar.add(os.path.join(PLOTS_DIR, fn), arcname=f"plots/{fn}")
    tar.add(LOG_FILE, arcname="train.log")

    tok_dir = os.path.join(BASE_DIR, "tokenizer")
    if os.path.exists(tok_dir):
        for item in sorted(os.listdir(tok_dir)):
            tar.add(os.path.join(tok_dir, item), arcname=f"tokenizer/{item}")

size_kb = os.path.getsize(OUTPUT_TAR) / 1024
print(f"[package] Done: {OUTPUT_TAR} ({size_kb:.0f} KB)")

# Save key metrics to stdout for easy capture
print("=" * 60)
print("TRAINING RESULTS:")
if steps:
    print(f"  Steps: {steps[-1]}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Min val BPB: {min_val_bpb:.6f}" if min_val_bpb else "  Min val BPB: N/A")
    print(f"  Avg tok/sec: {sum(tok_per_sec)//len(tok_per_sec):,}")
    print(f"  Total train time: {total_time_m:.2f}m" if total_time_m else "  Total train time: N/A")
print("=" * 60)
