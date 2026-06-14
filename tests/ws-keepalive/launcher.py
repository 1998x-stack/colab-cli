"""WebSocket keepalive test launcher — spawns training, keeps WS open for 8 min.

This script is run via `colab exec -f launcher.py --timeout 520`.
It keeps the exec WebSocket alive for the full 8-minute test window,
serving as the "persistent connection" signal to Colab's runtime proxy.

Key measurement: does the session survive past the ~10-min mark
when the WebSocket stays open vs. closing immediately after launch?
"""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/ws-test-output"
WATCHDOG_LOG = f"{OUT_DIR}/logs/watchdog.log"
TRAIN_LOG = f"{OUT_DIR}/logs/train.log"
TRAIN_PID_FILE = f"{OUT_DIR}/train.pid"
TRAIN_SCRIPT = "/content/fake_train.py"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/pngs", exist_ok=True)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def wlog(msg):
    line = f"[{ts()}] WATCHDOG: {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")

wlog("=== WebSocket Keepalive Test Started ===")
wlog(f"launcher PID={os.getpid()}  OUT_DIR={OUT_DIR}")

# Step 1: Ensure torch works (should be pre-installed)
wlog("Step 1: Checking environment...")
try:
    import torch
    wlog(f"CUDA available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        wlog(f"GPU={torch.cuda.get_device_name(0)}")
except Exception as e:
    wlog(f"WARNING: torch check failed: {e}")

# Step 2: Launch fake_train.py as detached subprocess
wlog("Step 2: Launching fake_train.py...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(TRAIN_LOG, "w") as log_f:
    proc = subprocess.Popen(
        [sys.executable, "-u", TRAIN_SCRIPT],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

wlog(f"fake_train.py spawned: PID={proc.pid}")
# Verify it started
time.sleep(3)
try:
    os.kill(proc.pid, 0)
    wlog(f"Process {proc.pid} is alive (verified)")
except OSError:
    wlog(f"WARNING: Process {proc.pid} died immediately! Check {TRAIN_LOG}")

# Step 3: Watchdog loop — 8 minutes (480 seconds)
WATCHDOG_DURATION = 8 * 60  # 480 seconds
CHECK_INTERVAL = 30          # every 30 seconds
start_time = time.time()
iteration = 0

wlog(f"Step 3: Entering watchdog loop (duration={WATCHDOG_DURATION}s, interval={CHECK_INTERVAL}s)")
wlog(f"WATCHDOG_START_TIME={ts()} unix={start_time:.0f}")

while time.time() - start_time < WATCHDOG_DURATION:
    iteration += 1
    elapsed = time.time() - start_time
    remaining = WATCHDOG_DURATION - elapsed

    # Check if training process is alive
    try:
        os.kill(proc.pid, 0)
        proc_status = "ALIVE"
    except OSError:
        proc_status = "DEAD"
        wlog(f"ALERT: Training process {proc.pid} died at t={elapsed:.0f}s!")
        break

    # Read last line of training log
    try:
        with open(TRAIN_LOG) as f:
            lines = f.readlines()
            last_train_line = lines[-1].strip() if lines else "(empty log)"
    except Exception:
        last_train_line = "(cannot read log)"

    wlog(f"watchdog_iter={iteration} | "
         f"elapsed={elapsed:.0f}s | "
         f"remaining={remaining:.0f}s | "
         f"train_pid={proc.pid}({proc_status}) | "
         f"train_log_tail: {last_train_line[:150]}")

    # Also print to stdout so colab exec output stream shows activity
    print(f"[{ts()}] heartbeat iteration={iteration} elapsed={elapsed:.0f}s", flush=True)

    if proc_status == "DEAD":
        break

    time.sleep(CHECK_INTERVAL)

total_elapsed = time.time() - start_time
wlog(f"Step 3 COMPLETE: Watchdog loop ended. Total elapsed={total_elapsed:.0f}s")

# Step 4: Final status
try:
    os.kill(proc.pid, 0)
    final_status = "ALIVE"
    wlog(f"Training process {proc.pid} still running at exit")
except OSError:
    final_status = "DEAD"
    wlog(f"Training process {proc.pid} no longer running")

# Print final log snippet
try:
    with open(TRAIN_LOG) as f:
        all_lines = f.readlines()
        wlog(f"train.log: {len(all_lines)} lines total")
        for line in all_lines[-5:]:
            wlog(f"  TRAIN_LOG: {line.strip()[:200]}")
except Exception as e:
    wlog(f"Could not read final train.log: {e}")

wlog(f"FINAL_STATUS: launcher_exit  train_pid={proc.pid}({final_status})  total_elapsed={total_elapsed:.0f}s")
wlog("=== WebSocket Keepalive Test Complete ===")
