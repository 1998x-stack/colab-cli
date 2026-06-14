"""Fake GPU training v2 — longer duration for reconnection test (20 min)."""
import time, sys, os
from datetime import datetime, timezone

OUT_DIR = "/content/ws-test-output"
LOG_PATH = f"{OUT_DIR}/logs/train.log"
PID_FILE = f"{OUT_DIR}/train.pid"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

log(f"fake_train_v2 started PID={os.getpid()}")

import torch
gpu_ok = torch.cuda.is_available()
log(f"CUDA={gpu_ok} GPU={torch.cuda.get_device_name(0) if gpu_ok else 'NONE'}")

INTERVAL = 30
TOTAL_MINUTES = 20
TOTAL_ITERATIONS = (TOTAL_MINUTES * 60) // INTERVAL

log(f"Starting {TOTAL_ITERATIONS} iterations over {TOTAL_MINUTES}min")

start_time = time.time()
for i in range(TOTAL_ITERATIONS):
    iter_start = time.time()
    elapsed = iter_start - start_time

    if gpu_ok:
        try:
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            c = torch.mm(a, b)
            gpu_status = f"gpu_ok val={c[0,0].item():.3f}"
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:
            gpu_status = f"gpu_err={e}"
    else:
        x = sum(j*j for j in range(2000000))
        gpu_status = f"cpu_work"

    iter_time = time.time() - iter_start
    log(f"iter={i+1}/{TOTAL_ITERATIONS} elapsed={elapsed:.0f}s "
        f"deadline={elapsed/60:.1f}min {gpu_status} iter_time={iter_time:.2f}s")

    if i < TOTAL_ITERATIONS - 1:
        time.sleep(max(0, INTERVAL - iter_time))

log(f"COMPLETE total_elapsed={time.time()-start_time:.0f}s")
