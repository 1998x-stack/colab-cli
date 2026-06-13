"""Check benchmark progress on Colab VM."""
import os
import subprocess

LOGFILE = os.environ.get("CHECK_LOG", "/content/train.log")
OUTPUT_DIR = os.environ.get("CHECK_OUTPUT", "/content/torch-compile-pipeline-output")

# ── Process ─────────────────────────────────────────────────────────────────
result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
if result.stdout.strip():
    print(f"Benchmark RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Benchmark NOT RUNNING (process not found)")

# ── Log tail ────────────────────────────────────────────────────────────────
print(f"\n── Last 20 lines of {LOGFILE} ──")
if os.path.exists(LOGFILE):
    with open(LOGFILE) as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())
else:
    print("(no log file yet)")

# ── Output files ────────────────────────────────────────────────────────────
print(f"\n── Output files ({OUTPUT_DIR}) ──")
if os.path.exists(OUTPUT_DIR):
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
else:
    print("(no output directory yet)")

# ── GPU status ──────────────────────────────────────────────────────────────
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print(f"\nGPU: {result.stdout.strip()}")
