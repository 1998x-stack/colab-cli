"""ws-1: Spawn 25-min training + 5-min watchdog. Tight window for relay chain."""
import subprocess, sys, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/relay25-output"
TRAIN_DUR = 25 * 60
WD_DUR = 5 * 60
INTERVAL = 25  # shorter interval for more TCP payload

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] ws-1: {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/logs/relay.log", "a") as f:
        f.write(line + "\n")

log(f"START pid={os.getpid()} train={TRAIN_DUR}s wd={WD_DUR}s")

import torch
log(f"GPU={torch.cuda.get_device_name(0)}")

# Write training script
code = f'''
import time, os, torch
from datetime import datetime, timezone
def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
def tlog(m):
    line = f"[{{ts()}}] TRAIN: {{m}}"
    print(line, flush=True)
    with open("{OUT_DIR}/logs/train.log", "a") as f:
        f.write(line + "\\n")
tlog(f"START pid={{os.getpid()}} dur={TRAIN_DUR}s")
start = time.time()
for i in range({TRAIN_DUR} // 30):
    it = time.time()
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = torch.mm(a, b)
    del a, b, c
    torch.cuda.empty_cache()
    e = time.time() - start
    dt = time.time() - it
    tlog(f"iter={{i+1}} elapsed={{e:.0f}}s progress={{e/{TRAIN_DUR}*100:.0f}}% dt={{dt:.2f}}s")
    if i < {TRAIN_DUR} // 30 - 1:
        time.sleep(max(0, 30 - dt))
tlog(f"COMPLETE total={{time.time()-start:.0f}}s")
'''

with open("/content/_train.py", "w") as f:
    f.write(code)

with open(f"{OUT_DIR}/watchdog_counter", "w") as f:
    f.write("1")

log("spawning training...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(f"{OUT_DIR}/logs/train.log", "w") as lf:
    tp = subprocess.Popen(
        [sys.executable, "-u", "/content/_train.py"],
        stdout=lf, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
with open(f"{OUT_DIR}/train.pid", "w") as f:
    f.write(str(tp.pid))
log(f"Training PID={tp.pid}")
time.sleep(3)
try:
    os.kill(tp.pid, 0)
    log("Training ALIVE")
except OSError:
    log("FATAL: Training died")
    sys.exit(1)

# 5-min watchdog
log(f"WD START dur={WD_DUR}s")
ws = time.time()
for i in range(WD_DUR // INTERVAL):
    time.sleep(INTERVAL)
    e = time.time() - ws
    ta = "ALIVE"
    try:
        os.kill(tp.pid, 0)
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
    print(f"[{ts()}] ws-1 heartbeat i={i+1} t={e:.0f}s train={ta}", flush=True)
    if ta == "DEAD":
        break

log(f"EXIT total={time.time()-ws:.0f}s → ws-2 should be queued")
