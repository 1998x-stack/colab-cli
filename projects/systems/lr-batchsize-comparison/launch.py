"""Launch LR x BS experiments for one batch size on Colab VM.

Reads BS from env var (default 16). Installs matplotlib, then runs
train.py 4 times sequentially with LR = 1e-4, 1e-3, 1e-2, 1e-1.

Upload to VM then: BS=64 colab exec -f launch.py --timeout 120
"""
import subprocess
import sys
import os
import time

BS = os.environ.get("BS", "16")
LRS = ["1e-4", "1e-3", "1e-2", "1e-1"]
OUT_DIR = "/content/lr-bs-output"
LOG = f"/content/launch_bs{BS}.log"
PID_FILE = f"{OUT_DIR}/train.pid"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

# Write PID for watchdog monitoring
with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

# Install matplotlib for optional plots
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])

print(f"[launch] BS={BS}  LRs={LRS}")
print(f"[launch] log={LOG}")

with open(LOG, "w") as log_fh:
    def tee(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    tee(f"[{time.strftime('%H:%M:%S')}] START BS={BS}")

    for lr_str in LRS:
        tee(f"\n{'='*50}")
        tee(f"[{time.strftime('%H:%M:%S')}] Running: train.py --bs {BS} --lr {lr_str}")
        tee(f"{'='*50}")

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-u", "/content/train.py", "--bs", BS, "--lr", lr_str],
            stdout=log_fh, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = time.time() - t0
        tee(f"[{time.strftime('%H:%M:%S')}] DONE rc={proc.returncode} elapsed={elapsed:.0f}s")

    tee(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE --- BS={BS}")
