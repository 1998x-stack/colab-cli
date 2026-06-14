#!/usr/bin/env python3
"""Check PPO MuJoCo training progress — process, GPU, log, CSV, PNGs, summary."""
import os
import json
import subprocess
import time

OUT = "/content/ppo-mujoco-output"
LOG_PATH = f"{OUT}/logs/train.log"
CSV_PATH = f"{OUT}/metrics.csv"
SUMMARY_PATH = f"{OUT}/summary.json"
PNGS_DIR = f"{OUT}/pngs"
CKPT_DIR = f"{OUT}/checkpoints"

def check_process():
    try:
        result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        if pids:
            print(f"[PROCESS] train.py running (PIDs: {', '.join(pids)})")
            return True
        else:
            print("[PROCESS] train.py NOT running")
            return False
    except Exception as e:
        print(f"[PROCESS] check failed: {e}")
        return None

def check_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            print(f"[GPU] {result.stdout.strip()}")
        else:
            print("[GPU] nvidia-smi failed or no GPU")
    except Exception as e:
        print(f"[GPU] check failed: {e}")

def tail_log(n=15):
    if not os.path.exists(LOG_PATH):
        print(f"[LOG] No log file at {LOG_PATH}")
        return
    with open(LOG_PATH) as f:
        lines = f.readlines()
    print(f"[LOG] {len(lines)} lines, last {min(n, len(lines))}:")
    for line in lines[-n:]:
        print(f"  {line.rstrip()}")

def show_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[CSV] No CSV at {CSV_PATH}")
        return
    with open(CSV_PATH) as f:
        lines = f.readlines()
    if len(lines) >= 2:
        print(f"[CSV] {len(lines)-1} rows")
        print(f"  {lines[0].rstrip()}")
        for line in lines[-3:]:
            print(f"  {line.rstrip()}")

def show_summary():
    if not os.path.exists(SUMMARY_PATH):
        print(f"[SUMMARY] No summary at {SUMMARY_PATH}")
        return
    with open(SUMMARY_PATH) as f:
        s = json.load(f)
    print(f"[SUMMARY] {len(s)} envs:")
    for name, data in sorted(s.items()):
        print(f"  {name:30s}  iters={data.get('iterations',0):4d}  "
              f"best_eval={data.get('best_eval_reward',0):8.1f}  "
              f"final={data.get('final_avg10_reward',0):8.1f}")

def list_pngs():
    if not os.path.isdir(PNGS_DIR):
        print(f"[PNGS] No pngs dir")
        return
    files = sorted(os.listdir(PNGS_DIR))
    print(f"[PNGS] {len(files)} files:")
    for f in files:
        path = os.path.join(PNGS_DIR, f)
        age = time.time() - os.path.getmtime(path)
        print(f"  {f} ({os.path.getsize(path)/1024:.1f} KB, {age:.0f}s ago)")

if __name__ == "__main__":
    print("=== PPO MuJoCo Training Progress ===\n")
    check_process()
    check_gpu()
    print()
    tail_log(12)
    print()
    show_csv()
    print()
    show_summary()
    print()
    list_pngs()
