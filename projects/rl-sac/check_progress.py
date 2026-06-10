"""Check SAC training progress on Colab VM."""
import os

logfile = "/content/sac_train.log"
ckpt_dir = "/content/checkpoints"

# Check if process is alive
import subprocess
result = subprocess.run(["pgrep", "-f", "sac_mountaincar"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"Training RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Training NOT RUNNING (process not found)")

# Latest log lines
print("\n── Last 15 log lines ──")
if os.path.exists(logfile):
    with open(logfile) as f:
        lines = f.readlines()
        for line in lines[-15:]:
            print(line.rstrip())
else:
    print("(no log file yet)")

# Checkpoints
print("\n── Checkpoints ──")
if os.path.exists(ckpt_dir):
    files = sorted(os.listdir(ckpt_dir))
    for f in files:
        path = os.path.join(ckpt_dir, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
else:
    print("(no checkpoints yet)")
