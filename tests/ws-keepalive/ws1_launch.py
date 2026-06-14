"""ws-1: Spawn 14-min training + 7-min watchdog. First in relay chain."""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/relay2-test-output"
TRAIN_DURATION = 14 * 60
WATCHDOG_DURATION = 7 * 60
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] ws-1: {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/logs/relay.log", "a") as f:
        f.write(line + "\n")

log(f"START pid={os.getpid()}")

import torch
log(f"GPU={torch.cuda.get_device_name(0)}")

# Write and launch training
train_code = f'''
import time, os, torch
from datetime import datetime, timezone
def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
def tlog(msg):
    line = f"[{{ts()}}] TRAIN: {{msg}}"
    print(line, flush=True)
    with open("{OUT_DIR}/logs/train.log", "a") as f:
        f.write(line + "\\n")
tlog(f"START pid={{os.getpid()}} duration={TRAIN_DURATION}s")
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
tlog(f"COMPLETE total={{time.time()-start:.0f}}s")
'''

with open("/content/_train_14min.py", "w") as f:
    f.write(train_code)

# Write watchdog counter file
with open(f"{OUT_DIR}/watchdog_counter", "w") as f:
    f.write("1")
with open(f"{OUT_DIR}/train.pid", "w") as f:
    pass  # will be filled by training subprocess

log("Launching 14-min training...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(f"{OUT_DIR}/logs/train.log", "w") as log_f:
    train_proc = subprocess.Popen(
        [sys.executable, "-u", "/content/_train_14min.py"],
        stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
with open(f"{OUT_DIR}/train.pid", "w") as f:
    f.write(str(train_proc.pid))

log(f"Training PID={train_proc.pid}")
time.sleep(3)
try:
    os.kill(train_proc.pid, 0)
    log("Training ALIVE")
except OSError:
    log("FATAL: Training died")
    sys.exit(1)

# 7-min watchdog
log(f"Watchdog START duration={WATCHDOG_DURATION}s")
ws = time.time()
for i in range(WATCHDOG_DURATION // INTERVAL):
    time.sleep(INTERVAL)
    elapsed = time.time() - ws
    t_alive = "ALIVE" if (lambda: os.kill(train_proc.pid, 0) or True)() else "DEAD"
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
    log(f"iter={i+1} elapsed={elapsed:.0f}s train={t_alive} gpu={gpu}")
    print(f"[{ts()}] ws-1 heartbeat iter={i+1} elapsed={elapsed:.0f}s train={t_alive}", flush=True)

log(f"EXIT total={time.time()-ws:.0f}s — ws-2 should already be queued")
