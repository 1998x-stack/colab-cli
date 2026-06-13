"""Local cron progress checker — runs via 'colab exec -f check_progress.py'.

Reads /content/heartbeat.json on VM, checks process health, reports status.
Intended to be run every 5-7 min via CronCreate.
"""

import json
import subprocess
import sys
import time

HEARTBEAT_PATH = "/content/heartbeat.json"

def check():
    # 1. Read heartbeat
    hb = None
    try:
        with open(HEARTBEAT_PATH) as f:
            hb = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[check] WARNING: No heartbeat file found")
        return 1

    # 2. Process alive check
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5
        )
        proc_alive = result.returncode == 0
    except Exception:
        proc_alive = False

    # 3. Heartbeat freshness
    now = time.time()
    hb_age = now - hb.get("timestamp", 0)
    hb_stale = hb_age > 120  # 2 min

    # 4. Report
    status = hb.get("status", "unknown")
    epoch = hb.get("epoch", 0)
    val_acc = hb.get("val_acc", 0)
    elapsed = hb.get("elapsed_seconds", 0)
    flops = hb.get("flops_consumed_tflops", 0)

    print(f"[check] Status: {status} | Epoch: {epoch} | Val Acc: {val_acc} | "
          f"Elapsed: {elapsed/60:.1f}m | FLOPS: {flops:.1f} TFLOPs | "
          f"HB age: {hb_age:.0f}s | Process alive: {proc_alive}")

    # 5. Health alerts
    alerts = []
    if hb_stale and not proc_alive:
        alerts.append("CRITICAL: VM likely dead — heartbeat stale AND no train.py process")
    elif hb_stale:
        alerts.append("WARNING: Heartbeat stale >2min but process may still be alive")
    elif not proc_alive and status != "done":
        alerts.append("CRITICAL: train.py process not found but heartbeat says not done")

    train_loss = hb.get("train_loss")
    if train_loss and train_loss > 10:
        alerts.append("WARNING: Loss >10 — may be diverging")

    if elapsed > 3300 and status != "done":  # 55 min
        alerts.append("WARNING: >55 min elapsed — trigger emergency download")

    for a in alerts:
        print(f"[check] {a}")

    return 0 if not alerts else 1

if __name__ == "__main__":
    sys.exit(check())
