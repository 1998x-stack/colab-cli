#!/usr/bin/env python3
"""Check DDPG vs TD3 MuJoCo training progress on Colab VM."""
import subprocess
import os

ROOT = "/content/ddpg-td3-mujoco-output"
ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
ALGOS = ["DDPG", "TD3"]

print("=== Training Progress Check ===")

# Process check
try:
    result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
    if result.stdout.strip():
        pids = result.stdout.strip().split()
        print(f"train.py PID(s): {', '.join(pids)} (alive)")
        # nvidia-smi
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

for env_name in ENVS:
    for algo in ALGOS:
        out_dir = f"{ROOT}/{env_name}/{algo}"
        log_path = f"{out_dir}/train.log"
        csv_path = f"{out_dir}/metrics.csv"
        png_path = f"{out_dir}/training_curves.png"

        print(f"--- {algo} {env_name} ---")
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            print(f"  log: {len(lines)} lines")
            # Show last 3 log lines
            for line in lines[-3:
                ]:
                print(f"    {line.rstrip()}")
            # Check if DONE
            if any("DONE" in l for l in lines[-5:
                ]):
                print("  STATUS: COMPLETED")
        else:
            print("  log: not yet created")

        if os.path.exists(csv_path):
            with open(csv_path) as f:
                n_rows = sum(1 for _ in f) - 1  # minus header
            print(f"  csv: {n_rows} episodes recorded")

        if os.path.exists(png_path):
            import time
            mtime = os.path.getmtime(png_path)
            age = time.time() - mtime
            print(f"  png: exists (updated {age:.0f}s ago)")

        print()

# Summary: which are done, which are in progress
print("=== Status Summary ===")
for env_name in ENVS:
    for algo in ALGOS:
        log_path = f"{ROOT}/{env_name}/{algo}/train.log"
        status = "PENDING"
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            if "DONE" in content:
                status = "DONE"
            else:
                status = "RUNNING"
        print(f"  {algo:4s} {env_name:20s} {status}")
