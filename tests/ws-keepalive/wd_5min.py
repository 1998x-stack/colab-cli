"""Generic 5-min watchdog. Auto-names from counter file. Tight window, frequent heartbeat."""
import subprocess, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/relay25-output"
DUR = 5 * 60
INTERVAL = 25

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

cf = f"{OUT_DIR}/watchdog_counter"
c = 1
if os.path.exists(cf):
    with open(cf) as f:
        c = int(f.read().strip()) + 1
with open(cf, "w") as f:
    f.write(str(c))
NAME = f"ws-{c}"

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/logs/relay.log", "a") as f:
        f.write(line + "\n")

log(f"START pid={os.getpid()}")

tp = None
pf = f"{OUT_DIR}/train.pid"
if os.path.exists(pf):
    with open(pf) as f:
        tp = int(f.read().strip())
    try:
        os.kill(tp, 0)
        log(f"Training PID={tp} ALIVE")
    except OSError:
        log(f"Training PID={tp} DEAD")
        tp = None

ws = time.time()
for i in range(DUR // INTERVAL):
    time.sleep(INTERVAL)
    e = time.time() - ws
    ta = "N/A"
    if tp:
        try:
            os.kill(tp, 0)
            ta = "ALIVE"
        except OSError:
            ta = "DEAD"
            log(f"Training died at t={e:.0f}s!")
    g = "?"
    try:
        g = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader",
            shell=True, text=True, timeout=5).strip()
    except:
        pass
    log(f"iter={i+1} elapsed={e:.0f}s train={ta} gpu={g}")
    print(f"[{ts()}] {NAME} heartbeat i={i+1} t={e:.0f}s train={ta}", flush=True)
    if ta == "DEAD":
        break

log(f"EXIT total={time.time()-ws:.0f}s")
