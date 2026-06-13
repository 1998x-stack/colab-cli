"""Check seq2seq training progress on Colab VM.

Usage:
    colab exec -f check_progress.py --timeout 15
"""
import os
import subprocess

SCRIPT = "train.py"
LOGFILE = "/content/seq2seq-t4/logs/train.log"
CKPT_DIR = "/content/seq2seq-t4/checkpoints"
PNGS_DIR = "/content/seq2seq-t4/pngs"
METRICS_CSV = "/content/seq2seq-t4/metrics.csv"

# ── Process ───────────────────────────────────────────────────────────────
result = subprocess.run(["pgrep", "-f", SCRIPT], capture_output=True, text=True)
if result.stdout.strip():
    print(f"Training RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Training NOT RUNNING (process not found)")

# ── Log tail ──────────────────────────────────────────────────────────────
print(f"\n── Last 15 lines of {LOGFILE} ──")
if os.path.exists(LOGFILE):
    with open(LOGFILE) as f:
        lines = f.readlines()
        for line in lines[-15:]:
            print(line.rstrip())
else:
    print("(no log file yet)")

# ── Metrics ───────────────────────────────────────────────────────────────
print(f"\n── Metrics ({METRICS_CSV}) ──")
if os.path.exists(METRICS_CSV):
    result = subprocess.run(["tail", "-5", METRICS_CSV], capture_output=True, text=True)
    print(result.stdout)
else:
    print("(no metrics CSV yet)")

# ── Checkpoints ───────────────────────────────────────────────────────────
print(f"── Checkpoints ({CKPT_DIR}) ──")
if os.path.exists(CKPT_DIR):
    files = sorted(os.listdir(CKPT_DIR))
    for f in files:
        path = os.path.join(CKPT_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
else:
    print("(no checkpoints yet)")

# ── Figures ───────────────────────────────────────────────────────────────
print(f"\n── Figures ({PNGS_DIR}) ──")
if os.path.exists(PNGS_DIR):
    pngs = sorted([f for f in os.listdir(PNGS_DIR) if f.endswith(".png")])
    for f in pngs:
        print(f"  {f}")
else:
    print("(no figures yet)")
