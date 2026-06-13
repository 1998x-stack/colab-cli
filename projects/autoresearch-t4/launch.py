"""Launch autoresearch on Colab T4. Run via: colab exec -f launch.py --timeout 600

Step 1: Install deps + run prepare.py (download TinyStories, train tokenizer)
Step 2: Run train.py (5-min GPT training with MuonAdamW)
"""

import subprocess
import sys
import os

print("[launch] Installing Python dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets", "rustbpe", "tiktoken", "torch", "numpy", "matplotlib",
])

print("[launch] Running prepare.py (data download + tokenizer training)...")
subprocess.check_call([sys.executable, "-u", "/content/prepare.py"])

print("[launch] Starting GPT training (5-min budget)...")
logfile = "/content/train.log"
with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/train.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"[launch] OK. PID={proc.pid} log={logfile}")
