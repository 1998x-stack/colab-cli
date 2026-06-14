"""Fake GPU training — periodic GPU work + timestamped logging for keepalive test."""
import time, sys, os, json
from datetime import datetime, timezone

OUT_DIR = "/content/ws-test-output"
LOG_PATH = f"{OUT_DIR}/logs/train.log"
PID_FILE = f"{OUT_DIR}/train.pid"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/pngs", exist_ok=True)

# Write PID for watchdog
with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

log("fake_train.py started")
log(f"PID={os.getpid()}  OUT_DIR={OUT_DIR}")

# GPU check
try:
    import torch
    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "NONE"
    log(f"CUDA available={gpu_ok}  GPU={gpu_name}")
except Exception as e:
    log(f"GPU check failed: {e}")
    gpu_ok = False

# Simple GPU work loop — 8 minutes total
INTERVAL = 30       # seconds between work units
TOTAL_MINUTES = 8   # run for 8 minutes
TOTAL_ITERATIONS = (TOTAL_MINUTES * 60) // INTERVAL

log(f"Starting {TOTAL_ITERATIONS} iterations ({INTERVAL}s interval, {TOTAL_MINUTES}min total)")

start_time = time.time()

for i in range(TOTAL_ITERATIONS):
    iter_start = time.time()
    elapsed_total = iter_start - start_time
    session_deadline = elapsed_total / 60  # minutes into the 8-min window

    # GPU work: matrix multiply
    if gpu_ok:
        try:
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            c = torch.mm(a, b)
            gpu_status = f"gpu_ok  matmul={(c[0,0].item()):.4f}"
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:
            gpu_status = f"gpu_err={e}"
    else:
        # CPU fallback
        x = sum(j * j for j in range(1000000))
        gpu_status = f"cpu_work={x}"

    iter_time = time.time() - iter_start
    log(f"iter={i+1}/{TOTAL_ITERATIONS} | "
        f"elapsed={elapsed_total:.0f}s | "
        f"deadline_at_10min={session_deadline:.1f}min | "
        f"{gpu_status} | "
        f"iter_time={iter_time:.2f}s")

    if i < TOTAL_ITERATIONS - 1:
        sleep_remaining = max(0, INTERVAL - iter_time)
        time.sleep(sleep_remaining)

total_elapsed = time.time() - start_time
log(f"fake_train.py COMPLETE. Total elapsed={total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
