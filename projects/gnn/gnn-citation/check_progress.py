#!/usr/bin/env python3
"""Check GCN citation training progress on Colab VM."""
import os
import subprocess
import time

ROOT = "/content/gnn-citation-output"
DATASETS = ["Cora", "CiteSeer", "PubMed"]

print("=== GCN Training Progress Check ===")

# Process check
try:
    result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
    if result.stdout.strip():
        pids = result.stdout.strip().split()
        print(f"train.py PID(s): {', '.join(pids)} (alive)")
        try:
            subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                          "--format=csv,noheader"], timeout=5)
        except Exception:
            pass
    else:
        print("WARNING: train.py not found running!")
except FileNotFoundError:
    print("pgrep not available")
    try:
        subprocess.run(["ps", "aux"], capture_output=True, text=True)
    except Exception:
        pass

print()

all_done = True
for name in DATASETS:
    out_dir = f"{ROOT}/{name}"
    log_path = f"{out_dir}/train.log"
    csv_path = f"{out_dir}/metrics.csv"
    png_path = f"{out_dir}/training_curves.png"

    print(f"--- {name} ---")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print(f"  log: {len(lines)} lines")
        for line in lines[-3:
            ]:
            print(f"    {line.rstrip()}")
        if not any("DONE" in l for l in lines[-5:
            ]):
            all_done = False
        else:
            print("  STATUS: COMPLETED")
    else:
        print("  log: not yet created")
        all_done = False

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            n_rows = sum(1 for _ in f) - 1
        print(f"  csv: {n_rows} epochs recorded")

    if os.path.exists(png_path):
        mtime = os.path.getmtime(png_path)
        age = time.time() - mtime if hasattr(time, "time") else 0
        print(f"  png: exists (updated {age:.0f}s ago)")

    print()

import time as _time
print("=== Status Summary ===")
for name in DATASETS:
    log_path = f"{ROOT}/{name}/train.log"
    status = "PENDING"
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        if "DONE" in content:
            status = "DONE"
        else:
            status = "RUNNING"
    print(f"  {name:<10} {status}")

comparison_png = f"{ROOT}/comparison/comparison_dashboard.png"
if os.path.exists(comparison_png):
    mtime = os.path.getmtime(comparison_png)
    age = _time.time() - mtime
    print(f"\n  Comparison dashboard: exists (updated {age:.0f}s ago)")
