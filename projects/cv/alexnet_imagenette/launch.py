"""Colab bootstrap: pip install deps, set HF_TOKEN, spawn train + watchdog as
detached subprocesses. Survives after colab exec disconnects.

Reads /content/exp_ids.txt to know which experiments to run.
"""

import subprocess
import sys
import os

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
print("[launch] DONE. Training running detached.")
