#!/usr/bin/env python3
"""Check DDQN vs NoisyNet training progress."""
import os
import json
import subprocess

OUT = "/content/ddqn-noisy-output"
LOG_PATH = f"{OUT}/train.log"
METRICS_PATH = f"{OUT}/metrics.json"

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
    print(f"[LOG] Last {min(n, len(lines))} of {len(lines)} lines:")
    for line in lines[-n:
        ]:
        print(f"  {line.rstrip()}")

def show_metrics():
    if not os.path.exists(METRICS_PATH):
        print("[METRICS] No metrics file")
        return
    with open(METRICS_PATH) as f:
        m = json.load(f)
    print(f"[METRICS] {len(m)} configs tracked:")
    for name, data in sorted(m.items()):
        eps = data.get("episodes", [])
        evals = data.get("evals", [])
        last_r = eps[-1]["reward"] if eps else float("nan")
        best_ev = max(e["mean_reward"] for e in evals) if evals else float("nan")
        print(f"  {name:30s}  eps={len(eps):4d}  last_r={last_r:8.1f}  best_ev={best_ev:8.1f}")

def list_outputs():
    print("[FILES]")
    for path in [LOG_PATH, METRICS_PATH]:
        if os.path.exists(path):
            print(f"  {path} ({os.path.getsize(path)/1024:.1f} KB)")
    plot_dir = f"{OUT}/plots"
    if os.path.isdir(plot_dir):
        for f in sorted(os.listdir(plot_dir)):
            fpath = os.path.join(plot_dir, f)
            if os.path.isfile(fpath):
                print(f"  {fpath} ({os.path.getsize(fpath)/1024:.1f} KB)")

if __name__ == "__main__":
    print("=== DDQN vs NoisyNet Training Progress ===\n")
    check_process()
    check_gpu()
    print()
    tail_log(15)
    print()
    show_metrics()
    print()
    list_outputs()
