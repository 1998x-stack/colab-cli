"""
nanochat Colab launcher — sets up the full nanochat pipeline on a Colab VM.

Run via: cb exec -f launch.py --timeout 600

What it does:
  1. Installs uv, clones nanochat, syncs GPU deps
  2. Downloads ~5 ClimbMix dataset shards
  3. Trains a BPE tokenizer
  4. Spawns base_train as a detached subprocess (survives exec timeout)

Training config is tuned for T4 GPU (16GB VRAM, SM 7.5, no bf16):
  - NANOCHAT_DTYPE=float16 (GradScaler enabled automatically)
  - depth=6, head-dim=64, max-seq-len=256, device-batch-size=1
  - 500 iterations (~30-60 min on T4)
  - Checkpoints every 100 steps, eval every 50 steps
"""

import subprocess
import sys
import os
import time
import shutil

NANOCHAT_DIR = "/content/nanochat"
LOG_FILE = "/content/train.log"
BASE_DIR = "/content/nanochat-data"
PLOTS_DIR = "/content/plots"

# ── Step 0: System info ────────────────────────────────────────────
print("=" * 60)
print("[setup] System info")
print(f"  Python: {sys.version}")
print(f"  uname: {os.uname()}")

result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True)
print(f"  GPU: {result.stdout.strip()}")
result = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
print(f"  Disk: {result.stdout.strip().split(chr(10))[-1].strip()}")
print("=" * 60)

# ── Step 1: Install uv ─────────────────────────────────────────────
if not shutil.which("uv"):
    print("[setup] Installing uv...")
    subprocess.check_call(["curl", "-LsSf", "https://astral.sh/uv/install.sh"], stdout=subprocess.DEVNULL)
    # uv installs to ~/.local/bin
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")
    subprocess.check_call(["uv", "--version"])
else:
    print("[setup] uv already installed")

# ── Step 2: Clone nanochat ─────────────────────────────────────────
if not os.path.exists(NANOCHAT_DIR):
    print("[setup] Cloning nanochat...")
    subprocess.check_call(["git", "clone", "https://github.com/karpathy/nanochat.git", NANOCHAT_DIR])
else:
    print("[setup] nanochat already cloned")
    # pull latest
    subprocess.check_call(["git", "-C", NANOCHAT_DIR, "pull", "--ff-only"])

# ── Step 3: uv sync GPU deps ───────────────────────────────────────
print("[setup] Installing nanochat dependencies (uv sync --extra gpu)...")
t0 = time.time()
subprocess.check_call(["uv", "sync", "--extra", "gpu"], cwd=NANOCHAT_DIR)
print(f"[setup] uv sync completed in {time.time() - t0:.1f}s")

# ── Step 4: Download dataset shards ─────────────────────────────────
os.makedirs(BASE_DIR, exist_ok=True)
os.environ["NANOCHAT_BASE_DIR"] = BASE_DIR

print("[setup] Downloading 5 ClimbMix dataset shards...")
venv_python = os.path.join(NANOCHAT_DIR, ".venv", "bin", "python")
subprocess.check_call([venv_python, "-m", "nanochat.dataset", "-n", "5"], cwd=NANOCHAT_DIR)

# ── Step 5: Train tokenizer ────────────────────────────────────────
print("[setup] Training BPE tokenizer (on 500M chars)...")
subprocess.check_call(
    [venv_python, "-m", "scripts.tok_train", "--max-chars=500000000"],
    cwd=NANOCHAT_DIR,
)

# ── Step 6: Spawn training as detached subprocess ──────────────────
print("[launch] Starting base_train (detached)...")

train_args = [
    venv_python, "-u", "-m", "scripts.base_train",
    "--depth=6",
    "--head-dim=64",
    "--max-seq-len=256",
    "--device-batch-size=1",
    "--num-iterations=500",
    "--eval-every=50",
    "--eval-tokens=16384",
    "--core-metric-every=-1",
    "--sample-every=100",
    "--save-every=100",
    "--window-pattern=L",
    "--warmup-steps=20",
    "--final-lr-frac=0.1",
    "--target-param-data-ratio=-1",
    "--run=dummy",
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["NANOCHAT_BASE_DIR"] = BASE_DIR
env["NANOCHAT_DTYPE"] = "float16"  # T4 has no bf16, use fp16 with GradScaler

with open(LOG_FILE, "w") as f:
    proc = subprocess.Popen(
        train_args,
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=NANOCHAT_DIR,
        env=env,
    )

print(f"[launch] OK. PID={proc.pid}")
print(f"[launch] Log: {LOG_FILE}")
print("[launch] Check progress: cb exec -f check_progress.py --timeout 15")
print("[launch] When done, run visualize.py to generate plots")
