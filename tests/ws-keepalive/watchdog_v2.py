"""WebSocket keepalive watchdog v2 — supports relay handoff.

Usage: run via colab exec with --timeout set to at least (duration + 30)

Environment variables:
  WATCHDOG_DURATION — seconds to run (default 600 = 10 min)
  WATCHDOG_NAME — label for logs (default "ws-1")
  WATCHDOG_CHECK_INTERVAL — seconds between heartbeat checks (default 30)
"""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/ws-test-output"
WATCHDOG_LOG = f"{OUT_DIR}/logs/watchdog.log"

DURATION = int(os.environ.get("WATCHDOG_DURATION", "600"))
NAME = os.environ.get("WATCHDOG_NAME", "ws-1")
INTERVAL = int(os.environ.get("WATCHDOG_CHECK_INTERVAL", "30"))

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def wlog(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")

wlog(f"WATCHDOG_START duration={DURATION}s interval={INTERVAL}s pid={os.getpid()}")

# Check if training process exists
TRAIN_PID_FILE = f"{OUT_DIR}/train.pid"
train_pid = None
if os.path.exists(TRAIN_PID_FILE):
    with open(TRAIN_PID_FILE) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        wlog(f"Found training process: PID={train_pid} ALIVE")
    except OSError:
        wlog(f"Training PID={train_pid} not running")
        train_pid = None
else:
    wlog("No training PID file — running in monitor-only mode")

start_time = time.time()
iteration = 0

wlog(f"Entering watchdog loop (will exit at {ts()} + {DURATION}s)")

while time.time() - start_time < DURATION:
    iteration += 1
    elapsed = time.time() - start_time
    remaining = DURATION - elapsed

    # Check training process
    train_status = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            train_status = f"PID={train_pid}(ALIVE)"
        except OSError:
            train_status = f"PID={train_pid}(DEAD)"
            wlog(f"ALERT: Training process died at t={elapsed:.0f}s")

    # Check GPU
    try:
        import subprocess as sp
        gpu = sp.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
            shell=True, text=True, timeout=5
        ).strip()
    except Exception:
        gpu = "gpu-check-failed"

    wlog(f"iter={iteration} elapsed={elapsed:.0f}s remaining={remaining:.0f}s "
         f"train={train_status} gpu=[{gpu}]")

    # Print heartbeat to stdout (keeps colab exec output stream active)
    print(f"[{ts()}] heartbeat {NAME} iter={iteration}", flush=True)

    if remaining <= 0:
        break
    time.sleep(min(INTERVAL, remaining))

total_elapsed = time.time() - start_time
wlog(f"WATCHDOG_EXIT total_elapsed={total_elapsed:.0f}s iterations={iteration}")
