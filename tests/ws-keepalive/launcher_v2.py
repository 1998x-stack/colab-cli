"""Launcher v2 — spawns training, then runs as watchdog-1 (10 min).

This is watchdog-1. watchdog-2 will be started separately while this runs,
testing whether two simultaneous WebSocket connections keep the session alive.
"""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/ws-test-output"
WATCHDOG_LOG = f"{OUT_DIR}/logs/watchdog.log"
TRAIN_LOG = f"{OUT_DIR}/logs/train.log"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

# Truncate logs for clean test
for f in [WATCHDOG_LOG, TRAIN_LOG]:
    if os.path.exists(f):
        os.remove(f)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def wlog(msg):
    line = f"[{ts()}] LAUNCHER: {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")

wlog("=== Relay Handoff Test ===")
wlog(f"launcher PID={os.getpid()}")

# Step 1: Check GPU
import torch
wlog(f"CUDA={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

# Step 2: Launch training
wlog("Launching fake_train_v2.py (20 min)...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(TRAIN_LOG, "w") as log_f:
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/fake_train_v2.py"],
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
wlog(f"Training spawned: PID={proc.pid}")
time.sleep(3)
try:
    os.kill(proc.pid, 0)
    wlog(f"Training PID={proc.pid} ALIVE")
except OSError:
    wlog(f"WARNING: Training died immediately!")

# Step 3: Watchdog loop — 10 minutes
DURATION = 10 * 60  # 600 seconds
INTERVAL = 30
start_time = time.time()
iteration = 0

wlog(f"Watchdog-1 loop START duration={DURATION}s")
wlog(">>> Start watchdog-2 NOW with: colab exec -f watchdog_v2.py --timeout 420")

while time.time() - start_time < DURATION:
    iteration += 1
    elapsed = time.time() - start_time

    train_alive = "?"
    try:
        os.kill(proc.pid, 0)
        train_alive = "ALIVE"
    except OSError:
        train_alive = "DEAD"

    # Read last train log line
    try:
        with open(TRAIN_LOG) as f:
            lines = f.readlines()
            last = lines[-1].strip() if lines else "(empty)"
    except Exception:
        last = "(cannot read)"

    wlog(f"WS1-iter={iteration} elapsed={elapsed:.0f}s train={proc.pid}({train_alive}) "
         f"train_tail: {last[:130]}")

    print(f"[{ts()}] ws1 heartbeat iter={iteration}", flush=True)

    if train_alive == "DEAD":
        wlog("Training died — exiting watchdog early")
        break

    time.sleep(INTERVAL)

wlog(f"Watchdog-1 EXIT total_elapsed={time.time()-start_time:.0f}s")
wlog("If watchdog-2 is running, session should survive the handoff.")
