#!/usr/bin/env python3
"""Check TD3 training progress — tail log, show metrics, find latest plots."""
import os
import json
import subprocess

OUT = "/content/td3-output"
LOG_PATH = f"{OUT}/train.log"
METRICS_PATH = f"{OUT}/metrics.json"
PLOTS_DIR = f"{OUT}/plots"

def check_process():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5
        )
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
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
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
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
        print(f"[LOG] Last {min(n, len(lines))} of {len(lines)} lines:")
        for line in lines[-n:]:
            print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"[LOG] read failed: {e}")

def show_metrics():
    if not os.path.exists(METRICS_PATH):
        print(f"[METRICS] No metrics file at {METRICS_PATH}")
        return
    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        eps = m.get("episodes", [])
        evals = m.get("eval_episodes", [])
        print(f"[METRICS] {len(eps)} episodes logged, {len(evals)} evals")
        if eps:
            last = eps[-1]
            print(f"  Last ep: {last['episode']} reward={last['reward']:.2f} "
                  f"a_loss={last.get('actor_loss')} c_loss={last.get('critic_loss')}")
        if evals:
            last_ev = evals[-1]
            print(f"  Last eval: ep {last_ev['episode']} mean={last_ev['mean_reward']:.2f} "
                  f"± {last_ev['std_reward']:.2f}")
        if len(eps) >= 5:
            recent = [e["reward"] for e in eps[-10:]]
            print(f"  Recent rewards (last {len(recent)}): "
                  f"min={min(recent):.1f} max={max(recent):.1f} avg={sum(recent)/len(recent):.1f}")
    except Exception as e:
        print(f"[METRICS] read failed: {e}")

def list_outputs():
    print("[FILES]")
    for path in [LOG_PATH, METRICS_PATH]:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {path} ({size_kb:.1f} KB)")
    if os.path.isdir(PLOTS_DIR):
        for f in sorted(os.listdir(PLOTS_DIR)):
            fpath = os.path.join(PLOTS_DIR, f)
            if os.path.isfile(fpath):
                print(f"  {fpath} ({os.path.getsize(fpath)/1024:.1f} KB)")

if __name__ == "__main__":
    print("=== TD3 Training Progress ===\n")
    check_process()
    check_gpu()
    print()
    tail_log(15)
    print()
    show_metrics()
    print()
    list_outputs()
