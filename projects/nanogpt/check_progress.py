"""Check nanoGPT training progress on Colab VM."""
import os, subprocess, sys

script = os.environ.get("CHECK_SCRIPT", "train_nanogpt.py")
log = os.environ.get("CHECK_LOG", "/content/nanogpt_train.log")
ckpt_dir = os.environ.get("CHECK_CKPT", "/content/out-nanogpt")

# 1. Check if training process is alive
try:
    result = subprocess.run(["pgrep", "-f", script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] Training process running (PID: {result.stdout.strip()})")
    else:
        print("[WARN] Training process NOT found. It may have completed or crashed.")
except FileNotFoundError:
    print("[WARN] pgrep not available")

# 2. Tail last 20 lines of log
print(f"\n--- Last 20 lines of {log} ---")
try:
    with open(log) as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())
except FileNotFoundError:
    print(f"Log file {log} not found")

# 3. List checkpoints
print(f"\n--- Checkpoints in {ckpt_dir} ---")
if os.path.isdir(ckpt_dir):
    for fname in sorted(os.listdir(ckpt_dir)):
        fpath = os.path.join(ckpt_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname}  ({size_kb:.1f} KB)")
else:
    print("  No checkpoint directory yet")

# 4. Show generated plots if available
for plot in ["loss_curve.png", "time_plot.png"]:
    ppath = os.path.join(ckpt_dir, plot)
    if os.path.exists(ppath):
        print(f"\n[OK] {plot} exists ({os.path.getsize(ppath)/1024:.1f} KB)")
