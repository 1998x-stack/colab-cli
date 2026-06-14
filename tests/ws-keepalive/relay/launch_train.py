"""Launch training + ws-1 watchdog (7-min WebSocket keepalive window).

This is the first watchdog in the relay chain. It:
1. Spawns fake_train_25min.py as detached subprocess (start_new_session=True)
2. Writes train.pid for subsequent watchdogs
3. Enters 7-min watchdog loop monitoring training + GPU + log tail
4. Exits — ws-2 must already be queued (launched at T+6min)

Usage: colab exec -s <name> -f launch_train.py --timeout 540
"""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/relay-test-output"
LOG = f"{OUT_DIR}/logs/watchdog.log"
COUNTER_FILE = f"{OUT_DIR}/watchdog_counter"
TRAIN_PID_FILE = f"{OUT_DIR}/train.pid"
TRAIN_SCRIPT = "/content/fake_train_25min.py"

DURATION = 420   # 7 minutes
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

with open(COUNTER_FILE, "w") as f:
    f.write("1")
NAME = "ws-1"

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def wlog(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

wlog("=" * 50)
wlog(f"RELAY_TEST_START pid={os.getpid()}")
wlog("=" * 50)

# ── GPU check ────────────────────────────────────────────
try:
    import torch
    wlog(f"GPU={torch.cuda.get_device_name(0)} cuda={torch.cuda.is_available()}")
except Exception as e:
    wlog(f"GPU check failed: {e}")

# ── Launch training as detached subprocess ───────────────
wlog(f"Launching {TRAIN_SCRIPT}...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(f"{OUT_DIR}/logs/train.log", "w") as log_f:
    proc = subprocess.Popen(
        [sys.executable, "-u", TRAIN_SCRIPT],
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )

with open(TRAIN_PID_FILE, "w") as f:
    f.write(str(proc.pid))

wlog(f"Training spawned: PID={proc.pid}")

time.sleep(3)
try:
    os.kill(proc.pid, 0)
    wlog(f"Training PID={proc.pid} verified ALIVE")
except OSError:
    wlog(f"FATAL: Training PID={proc.pid} died immediately")
    if os.path.exists(f"{OUT_DIR}/logs/train.log"):
        with open(f"{OUT_DIR}/logs/train.log") as f:
            wlog(f"train.log: {f.read()[:500]}")
    sys.exit(1)

# ── ws-1 watchdog loop (7 min) ───────────────────────────
start_time = time.time()
iteration = 0
wlog(f"Watchdog START duration={DURATION}s")

while time.time() - start_time < DURATION:
    iteration += 1
    elapsed = time.time() - start_time

    train_status = "?"
    try:
        os.kill(proc.pid, 0)
        train_status = f"ALIVE(PID={proc.pid})"
    except OSError:
        train_status = "DEAD"
        wlog(f"ALERT: training died at t={elapsed:.0f}s")

    gpu_info = "?"
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader",
            shell=True, text=True, timeout=5,
        ).strip()
    except Exception:
        pass

    train_tail = "(no log)"
    train_log_path = f"{OUT_DIR}/logs/train.log"
    if os.path.exists(train_log_path):
        try:
            with open(train_log_path) as f:
                lines = f.readlines()
                train_tail = lines[-1].strip()[-200:] if lines else "(empty)"
        except Exception:
            pass

    wlog(f"iter={iteration} elapsed={elapsed:.0f}s "
         f"train={train_status} gpu=[{gpu_info}] "
         f"tail: {train_tail}")

    print(f"[{ts()}] {NAME} heartbeat iter={iteration} elapsed={elapsed:.0f}s "
          f"train={train_status}", flush=True)

    if train_status == "DEAD":
        wlog("training died — exiting early")
        break

    time.sleep(INTERVAL)

total = time.time() - start_time
wlog(f"EXIT total_elapsed={total:.0f}s iterations={iteration}")
wlog("HANDOFF → ws-2 should already be queued")
wlog("=" * 50)
