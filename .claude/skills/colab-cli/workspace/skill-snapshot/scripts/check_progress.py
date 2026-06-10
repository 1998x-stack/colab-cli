"""Check training progress on Colab VM — with proxy health check."""
import os
import subprocess

SCRIPT = os.environ.get("CHECK_SCRIPT", "train.py")
LOGFILE = os.environ.get("CHECK_LOG", "/content/train.log")
CKPT_DIR = os.environ.get("CHECK_CKPT", "/content/checkpoints")

# ── Proxy health ──────────────────────────────────────────────────────────
result = subprocess.run(
    ["curl", "-s", "--max-time", "3", "-x", "http://127.0.0.1:7890",
     "https://www.google.com", "-o", "/dev/null", "-w", "%{http_code}"],
    capture_output=True, text=True,
)
if result.stdout.strip() == "200":
    print("Proxy: OK (http://127.0.0.1:7890)")
else:
    print("Proxy: NOT REACHABLE — downloads may fail")

# ── Process ───────────────────────────────────────────────────────────────
result = subprocess.run(["pgrep", "-f", SCRIPT], capture_output=True, text=True)
if result.stdout.strip():
    print(f"\nTraining RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("\nTraining NOT RUNNING (process not found)")

# ── Log tail ──────────────────────────────────────────────────────────────
print(f"\n── Last 15 lines of {LOGFILE} ──")
if os.path.exists(LOGFILE):
    with open(LOGFILE) as f:
        lines = f.readlines()
        for line in lines[-15:]:
            print(line.rstrip())
else:
    print("(no log file yet)")

# ── Checkpoints ───────────────────────────────────────────────────────────
print(f"\n── Checkpoints ({CKPT_DIR}) ──")
if os.path.exists(CKPT_DIR):
    files = sorted(os.listdir(CKPT_DIR))
    for f in files:
        path = os.path.join(CKPT_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
else:
    print("(no checkpoints yet)")
