"""WebSocket relay watchdog — keeps WS alive, monitors training, 7-min window.

Upload once. Run via: colab exec -f watchdog.py --timeout 480

Auto-names itself (ws-1, ws-2, ...) using a counter file on the VM.
Supports relay handoff: start next watchdog before this one exits.
"""
import subprocess, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/relay-test-output"
LOG = f"{OUT_DIR}/logs/watchdog.log"
COUNTER_FILE = f"{OUT_DIR}/watchdog_counter"
TRAIN_PID_FILE = f"{OUT_DIR}/train.pid"

DURATION = 420   # 7 minutes
INTERVAL = 30    # check every 30 seconds

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

# Auto-increment watchdog name
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

# Check if training exists
train_pid = None
if os.path.exists(TRAIN_PID_FILE):
    with open(TRAIN_PID_FILE) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        wlog(f"found training PID={train_pid} ALIVE")
    except OSError:
        wlog(f"training PID={train_pid} is DEAD")
        train_pid = None
else:
    wlog("no training PID file — monitor-only mode")

# GPU check
try:
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    wlog(f"GPU={gpu_name}")
except Exception:
    wlog("GPU check failed")

start_time = time.time()
iteration = 0

while time.time() - start_time < DURATION:
    iteration += 1
    elapsed = time.time() - start_time

    # Training status
    train_status = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            train_status = f"ALIVE(PID={train_pid})"
        except OSError:
            train_status = "DEAD"
            wlog(f"ALERT: training died at t={elapsed:.0f}s")

    # GPU utilization
    gpu_info = "?"
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
            shell=True, text=True, timeout=5
        ).strip()
    except Exception:
        pass

    # Read last training log line
    train_tail = "(no log)"
    train_log_path = f"{OUT_DIR}/logs/train.log"
    if os.path.exists(train_log_path):
        try:
            with open(train_log_path) as f:
                lines = f.readlines()
                train_tail = lines[-1].strip()[-150:] if lines else "(empty)"
        except Exception:
            pass

    wlog(f"iter={iteration} elapsed={elapsed:.0f}s "
         f"train={train_status} gpu=[{gpu_info}] "
         f"log_tail: {train_tail}")

    # Heartbeat to stdout — keeps WebSocket output stream active
    print(f"[{ts()}] {NAME} heartbeat iter={iteration} elapsed={elapsed:.0f}s",
          flush=True)

    if train_status == "DEAD" and train_pid is not None:
        wlog("exiting early — training dead")
        break

    time.sleep(INTERVAL)

total = time.time() - start_time
wlog(f"EXIT total_elapsed={total:.0f}s iterations={iteration}")
wlog(f"HANDOFF: start next watchdog with: colab exec -f watchdog.py --timeout 480")
