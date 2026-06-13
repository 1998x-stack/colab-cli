"""
Check CNN quantization training progress on Colab VM.
Usage: colab exec -f check_progress.py --timeout 15
Override defaults: CHECK_LOG=/content/xxx.log CHECK_METRICS=/content/xxx/metrics.csv
"""
import os
import subprocess
from pathlib import Path

LOG = os.environ.get("CHECK_LOG", "/content/cnn-quantization-output/logs/train.log")
METRICS = os.environ.get("CHECK_METRICS", "/content/cnn-quantization-output/metrics.csv")
SUMMARY = os.environ.get("CHECK_SUMMARY", "/content/cnn-quantization-output/quantization_summary.csv")
PNGS = os.environ.get("CHECK_PNGS", "/content/cnn-quantization-output/pngs")

log_path = Path(LOG)
metrics_path = Path(METRICS)
summary_path = Path(SUMMARY)
pngs_dir = Path(PNGS)

# 1. Check if training process is alive
print("=== Process Status ===")
try:
    result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5)
    if result.stdout.strip():
        print(f"ALIVE: PID(s) {result.stdout.strip()}")
    else:
        print("DEAD: train.py not running")
except Exception as e:
    print(f"pgrep failed: {e}")

# 2. GPU status
print("\n=== GPU Status ===")
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM used: {torch.cuda.memory_allocated(0)/1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("CUDA not available")
except Exception as e:
    print(f"torch check failed: {e}")

# 3. Tail of training log
print(f"\n=== Tail -12: {LOG} ===")
if log_path.exists():
    lines = log_path.read_text().splitlines()
    for line in lines[-12:]:
        print(line)
    print(f"({len(lines)} total lines)")
else:
    print("LOG NOT FOUND")

# 4. Latest metrics
print(f"\n=== Tail -5: {METRICS} ===")
if metrics_path.exists():
    lines = metrics_path.read_text().splitlines()
    if lines:
        print(lines[0])  # header
        for line in lines[-5:]:
            print(line)
    print(f"({len(lines)} total rows incl. header)")
else:
    print("METRICS NOT FOUND")

# 5. Quantization summary
print(f"\n=== {SUMMARY} ===")
if summary_path.exists():
    print(summary_path.read_text())
else:
    print("SUMMARY NOT YET — training still running")

# 6. PNGs
print(f"\n=== PNGs: {PNGS} ===")
if pngs_dir.exists():
    pngs = sorted(pngs_dir.glob("*.png"))
    if pngs:
        for p in pngs:
            sz = p.stat().st_size / 1024
            print(f"  {p.name} ({sz:.0f} KB)")
    else:
        print("  No PNGs yet")
else:
    print("  PNGs dir not found")

# 7. Elapsed time
print("\n=== Wall Clock ===")
if log_path.exists():
    import datetime
    mtime = datetime.datetime.fromtimestamp(log_path.stat().st_mtime)
    now = datetime.datetime.now()
    age = (now - mtime).total_seconds()
    print(f"Log last modified: {mtime.strftime('%H:%M:%S')} ({age:.0f}s ago)")
