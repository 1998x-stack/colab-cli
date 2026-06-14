"""Emergency watchdog — minimal. Just prints heartbeat every 20s for 2 min."""
import time, os, subprocess
from datetime import datetime, timezone
OUT_DIR = "/content/relay25-output"
def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
print(f"[{ts()}] EMERGENCY-WD START pid={os.getpid()}", flush=True)
# Check training
pf = f"{OUT_DIR}/train.pid"
if os.path.exists(pf):
    with open(pf) as f:
        tp = int(f.read().strip())
    try:
        os.kill(tp, 0)
        print(f"[{ts()}] Training PID={tp} ALIVE", flush=True)
    except OSError:
        print(f"[{ts()}] Training DEAD", flush=True)
for i in range(6):
    time.sleep(20)
    g = "?"
    try:
        g = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader",
            shell=True, text=True, timeout=5).strip()
    except:
        pass
    print(f"[{ts()}] emergency heartbeat i={i+1} gpu={g}", flush=True)
print(f"[{ts()}] EMERGENCY-WD EXIT", flush=True)
