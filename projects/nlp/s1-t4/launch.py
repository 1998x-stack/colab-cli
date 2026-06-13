"""Colab bootstrap for s1-t4: pip install deps, set HF_TOKEN, spawn train.py
as detached subprocess. Survives after colab exec disconnects.

Reads /content/hf_token for Hugging Face authentication.
"""

import subprocess
import sys
import os
import shutil

HF_TOKEN_PATH = "/content/hf_token"
LOG_DIR = "/content/s1-t4/logs"
TRAIN_SCRIPT = "/content/s1-t4/train.py"
DATA_PATH = "/content/s1-t4/s1k_filtered.jsonl"
CHECKPOINT_DIR = "/content/s1-t4/checkpoints"
RESULTS_DIR = "/content/s1-t4/results"
PNGS_DIR = "/content/s1-t4/pngs"
DEPS = ["bitsandbytes", "peft", "datasets", "matplotlib", "tqdm"]

# --- Create output directories ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PNGS_DIR, exist_ok=True)

# --- Set HF_TOKEN ---
try:
    with open(HF_TOKEN_PATH) as f:
        token = f.read().strip()
    os.environ["HF_TOKEN"] = token
    print("[launch] HF_TOKEN set")
except FileNotFoundError:
    print("[launch] WARNING: /content/hf_token not found — HF Hub access may fail")

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[launch] Dependencies installed")

# --- Clear stale HF datasets cache (avoids LocalFileSystem error) ---
hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
    print("[launch] Cleared HF datasets cache")

# --- Spawn training ---
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cmd = (
    f"{sys.executable} -u {TRAIN_SCRIPT} "
    f"--data {DATA_PATH} "
    f"--output_dir {CHECKPOINT_DIR}"
)

log_path = os.path.join(LOG_DIR, "train.log")
print(f"[launch] Running: {cmd}")
with open(log_path, "w") as f:
    proc = subprocess.Popen(
        cmd.split(),
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"[launch] Train PID={proc.pid}, log={log_path}")
print(f"[launch] Checkpoints: {CHECKPOINT_DIR}")
print("[launch] DONE. Training running detached.")
print("[launch] Monitor: colab exec -s <session> -f /content/s1-t4/check_progress.py")
