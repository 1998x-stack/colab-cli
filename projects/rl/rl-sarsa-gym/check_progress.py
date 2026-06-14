#!/usr/bin/env python3
"""Check SARSA training progress — process, log, CSV, PNGs."""
import os
import json
import subprocess
import time

OUT = "/content/rl-sarsa-output"
LOG_PATH = f"{OUT}/logs/train.log"
CSV_PATH = f"{OUT}/metrics.csv"
SUMMARY_PATH = f"{OUT}/summary.json"
PNGS_DIR = f"{OUT}/pngs"

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
    print(f"[SUMMARY] episodes={s.get('episodes_completed', 0)}  "
          f"best_avg100={s.get('best_avg100_reward', 0):.1f}  "
          f"eval_reward={s.get('eval_reward', 0):.1f}  "
          f"elapsed={s.get('elapsed_s', 0):.0f}s")

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
    print("=== SARSA Training Progress ===\n")
    check_process()
    print()
    tail_log(10)
    print()
    show_csv()
    print()
    show_summary()
    print()
    list_pngs()
