"""ws-2: Generic 7-min watchdog. Auto-names from counter file on VM."""
import subprocess, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/relay2-test-output"
DURATION = 7 * 60
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

# Auto-name
counter = 1
cf = f"{OUT_DIR}/watchdog_counter"
if os.path.exists(cf):
    with open(cf) as f:
        counter = int(f.read().strip()) + 1
with open(cf, "w") as f:
    f.write(str(counter))
NAME = f"ws-{counter}"

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/logs/relay.log", "a") as f:
        f.write(line + "\n")

log(f"START pid={os.getpid()}")

# Find training
train_pid = None
pf = f"{OUT_DIR}/train.pid"
if os.path.exists(pf):
    with open(pf) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        log(f"Training PID={train_pid} ALIVE")
    except OSError:
        log(f"Training PID={train_pid} DEAD")
        train_pid = None
else:
    log("No train.pid found")

# 7-min watchdog
ws = time.time()
for i in range(DURATION // INTERVAL):
    time.sleep(INTERVAL)
    elapsed = time.time() - ws
    t_alive = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
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
    print(f"[{ts()}] {NAME} heartbeat iter={i+1} elapsed={elapsed:.0f}s train={t_alive}", flush=True)

log(f"EXIT total={time.time()-ws:.0f}s")
