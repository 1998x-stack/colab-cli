"""Single watchdog test: spawn 12-min fake training, keep WS alive for 8 min.

Piped to colab exec. Tests whether a single WebSocket extends session life
past the ~10 minute default death point.
"""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/single-test-output"
TRAIN_DURATION = 12 * 60  # 12 min training
WATCHDOG_DURATION = 8 * 60  # 8 min watchdog (safe inside China WS window)
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/logs/test.log", "a") as f:
        f.write(line + "\n")

log(f"TEST_START pid={os.getpid()} train_dur={TRAIN_DURATION}s watchdog_dur={WATCHDOG_DURATION}s")

# GPU check
import torch
log(f"GPU={torch.cuda.get_device_name(0)}")

# Write training script to VM
train_code = f'''
import time, os, torch
from datetime import datetime, timezone
def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
def tlog(msg):
    line = f"[{{ts()}}] {{msg}}"
    print(line, flush=True)
    with open("{OUT_DIR}/logs/train.log", "a") as f:
        f.write(line + "\\n")
tlog(f"TRAIN_START pid={{os.getpid()}}")
start = time.time()
for i in range({TRAIN_DURATION} // 30):
    it = time.time()
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = torch.mm(a, b)
    del a, b, c
    torch.cuda.empty_cache()
    elapsed = time.time() - start
    dt = time.time() - it
    tlog(f"iter={{i+1}} elapsed={{elapsed:.0f}}s gpu_ok dt={{dt:.2f}}s")
    if i < {TRAIN_DURATION} // 30 - 1:
        time.sleep(max(0, 30 - dt))
tlog(f"TRAIN_COMPLETE total={{time.time()-start:.0f}}s")
'''

with open("/content/_train_12min.py", "w") as f:
    f.write(train_code)

# Launch training as detached subprocess
log("Launching 12-min training...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(f"{OUT_DIR}/logs/train.log", "w") as log_f:
    train_proc = subprocess.Popen(
        [sys.executable, "-u", "/content/_train_12min.py"],
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
log(f"Training PID={train_proc.pid}")

time.sleep(3)
try:
    os.kill(train_proc.pid, 0)
    log("Training verified ALIVE")
except OSError:
    log("FATAL: Training died immediately")
    sys.exit(1)

# Watchdog loop — 8 minutes
log(f"Watchdog START duration={WATCHDOG_DURATION}s")
watchdog_start = time.time()
iteration = 0

while time.time() - watchdog_start < WATCHDOG_DURATION:
    iteration += 1
    elapsed = time.time() - watchdog_start

    t_alive = "?"
    try:
        os.kill(train_proc.pid, 0)
        t_alive = "ALIVE"
    except OSError:
        t_alive = "DEAD"
        log(f"Training died at t={elapsed:.0f}s!")

    gpu = "?"
    try:
        gpu = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader",
            shell=True, text=True, timeout=5).strip()
    except:
        pass

    log(f"watchdog iter={iteration} elapsed={elapsed:.0f}s train={t_alive} gpu={gpu}")
    print(f"[{ts()}] heartbeat iter={iteration} elapsed={elapsed:.0f}s train={t_alive}", flush=True)

    time.sleep(INTERVAL)

log(f"Watchdog EXIT total_elapsed={time.time()-watchdog_start:.0f}s")
log("WebSocket will close now. Training should continue independently.")
