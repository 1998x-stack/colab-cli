"""WebSocket relay watchdog for LRxBS experiments — 7-min window.

Upload once. Run via: colab exec -f watchdog.py --timeout 480

Keeps WebSocket alive while detached launch.py runs train.py sequentially.
Monitors training progress via PID check + log tail.
"""
import glob
import os
import subprocess
import time
from datetime import datetime, timezone

OUT_DIR = "/content/lr-bs-output"
LOG = f"{OUT_DIR}/logs/watchdog.log"
COUNTER_FILE = f"{OUT_DIR}/watchdog_counter"
PID_FILE = f"{OUT_DIR}/train.pid"

DURATION = 420   # 7 minutes
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

counter = 1
if os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE) as f:
        counter = int(f.read().strip()) + 1
with open(COUNTER_FILE, "w") as f:
    f.write(str(counter))
NAME = f"ws-{counter}"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def wlog(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


wlog(f"START pid={os.getpid()} duration={DURATION}s")

train_pid = None
if os.path.exists(PID_FILE):
    with open(PID_FILE) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        wlog(f"training PID={train_pid} ALIVE")
    except OSError:
        wlog(f"training PID={train_pid} DEAD")
        train_pid = None
else:
    wlog("no PID file — monitoring via log files only")

try:
    import torch
    wlog(f"GPU={torch.cuda.get_device_name(0)}")
except Exception:
    wlog("GPU check skipped")

start_time = time.time()
for iteration in range(DURATION // INTERVAL):
    time.sleep(INTERVAL)
    elapsed = time.time() - start_time

    train_status = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            train_status = f"ALIVE(PID={train_pid})"
        except OSError:
            train_status = "DEAD"
            wlog("ALERT: training process died!")

    gpu_info = "?"
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
            shell=True, text=True, timeout=5,
        ).strip()
    except Exception:
        pass

    log_files = sorted(glob.glob(f"{OUT_DIR}/bs*_lr*/logs/train.log"))
    train_tail = "(no log)"
    if log_files:
        try:
            with open(log_files[-1]) as f:
                lines = f.readlines()
                train_tail = lines[-1].strip()[-180:] if lines else "(empty)"
        except Exception:
            pass

    wlog(f"iter={iteration+1} elapsed={elapsed:.0f}s train={train_status} gpu=[{gpu_info}] log: {train_tail}")
    print(f"[{ts()}] {NAME} heartbeat iter={iteration+1} elapsed={elapsed:.0f}s", flush=True)

    if train_status == "DEAD" and train_pid is not None:
        wlog("exiting early — training is dead")
        break

total = time.time() - start_time
wlog(f"EXIT total_elapsed={total:.0f}s")
wlog("HANDOFF: start next with: colab exec -f watchdog.py --timeout 480")
