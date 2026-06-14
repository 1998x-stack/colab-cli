#!/usr/bin/env python3
"""Check AlphaGo training progress — process, GPU, log tail, metrics, checkpoints."""
import os
import json
import subprocess

OUT = "/content/alphago-output"
LOG_PATH = f"{OUT}/logs/train.log"
CSV_PATH = f"{OUT}/metrics.csv"
SUMMARY_PATH = f"{OUT}/summary.json"
CKPT_DIR = "/content/drive/MyDrive/alphago-checkpoints"

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

def tail_log(n=20):
    if not os.path.exists(LOG_PATH):
        print(f"[LOG] No log file at {LOG_PATH}")
        return
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
        print(f"[LOG] Last {min(n, len(lines))} of {len(lines)} lines:")
        for line in lines[-n:]:
            print(f"  {line.rstrip()}")
        # Detect FATAL / NaN
        for line in lines:
            if "FATAL" in line or "NaN" in line:
                print(f"  *** ALERT: {line.rstrip()} ***")
    except Exception as e:
        print(f"[LOG] read failed: {e}")

def show_metrics():
    if not os.path.exists(CSV_PATH):
        print(f"[METRICS] No CSV at {CSV_PATH}")
        return
    try:
        with open(CSV_PATH) as f:
            lines = f.readlines()
        if len(lines) >= 2:
            print(f"[METRICS] {len(lines)-1} rows")
            print(f"  Header: {lines[0].rstrip()}")
            for line in lines[-3:]:
                print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"[METRICS] read failed: {e}")

def show_summary():
    if not os.path.exists(SUMMARY_PATH):
        print(f"[SUMMARY] No summary at {SUMMARY_PATH}")
        return
    try:
        with open(SUMMARY_PATH) as f:
            s = json.load(f)
        print(f"[SUMMARY] iter={s.get('iteration','?')}  "
              f"policy_loss={s.get('train_metrics',{}).get('policy_loss','?'):.4f}  "
              f"value_loss={s.get('train_metrics',{}).get('value_loss','?'):.4f}  "
              f"win_rate={s.get('eval_metrics',{}).get('win_rate',0):.3f}  "
              f"best={s.get('is_best',False)}  elapsed={s.get('elapsed_s',0):.0f}s")
    except Exception as e:
        print(f"[SUMMARY] read failed: {e}")

def check_checkpoints():
    if not os.path.isdir(CKPT_DIR):
        print(f"[CKPT] No checkpoint dir at {CKPT_DIR} (Drive not mounted?)")
        return
    try:
        files = sorted(os.listdir(CKPT_DIR))
        print(f"[CKPT] {len(files)} files in Drive:")
        for f in files:
            path = os.path.join(CKPT_DIR, f)
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  {f} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[CKPT] check failed: {e}")

if __name__ == "__main__":
    print("=== AlphaGo Training Progress ===\n")
    check_process()
    check_gpu()
    print()
    tail_log(15)
    print()
    show_metrics()
    print()
    show_summary()
    print()
    check_checkpoints()
