"""Relay watchdog (ws-2, ws-3, ...) — 7-min WebSocket keepalive window.

Upload once to /content/watchdog.py. Run via stdin pipe:
  echo 'exec(open("/content/watchdog.py").read())' | colab exec -s <name> --timeout 540 &

Auto-names itself (ws-2, ws-3, ...) using a counter file on the VM.
Monitors training PID, GPU utilization, and training log progress.
"""
import subprocess, os, time
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────
OUT_DIR = None                          # None = scan /content/*-output/ for train.pid
DURATION = 420                          # 7 minutes
INTERVAL = 30
# ────────────────────────────────────────────────────────


def find_output_dir():
    """Find the output directory by scanning for train.pid files."""
    if OUT_DIR:
        return OUT_DIR
    for entry in os.listdir("/content/"):
        path = os.path.join("/content/", entry)
        if os.path.isdir(path) and entry.endswith("-output"):
            pid_file = os.path.join(path, "train.pid")
            if os.path.exists(pid_file):
                return path
    return "/content/output"


OUT_DIR = find_output_dir()
LOG = f"{OUT_DIR}/logs/watchdog.log"
COUNTER_FILE = f"{OUT_DIR}/watchdog_counter"
TRAIN_PID_FILE = f"{OUT_DIR}/train.pid"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

# Auto-increment watchdog name
counter = 1
if os.path.exists(COUNTER_FILE):
    with open(COUNTER_FILE) as f:
        counter = int(f.read().strip()) + 1
with open(COUNTER_FILE, "w") as f:
    f.write(str(counter))
NAME = f"ws-{counter}"


def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def wlog(msg):
    line = f"[{ts()}] {NAME}: {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


wlog(f"START pid={os.getpid()} out_dir={OUT_DIR} duration={DURATION}s")

# ── Check training ─────────────────────────────────────
train_pid = None
if os.path.exists(TRAIN_PID_FILE):
    with open(TRAIN_PID_FILE) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        wlog(f"found training PID={train_pid} ALIVE")
    except OSError:
        wlog(f"training PID={train_pid} DEAD (stale PID file)")
        train_pid = None
else:
    wlog("no train.pid — monitor-only mode")

# ── GPU check ───────────────────────────────────────────
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    wlog(f"GPU={gpu}")
except Exception:
    wlog("GPU check failed (torch not available)")

# ── Watchdog loop ───────────────────────────────────────
start_time = time.time()
iteration = 0

while time.time() - start_time < DURATION:
    iteration += 1
    elapsed = time.time() - start_time

    # Training status
    train_status = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            train_status = f"ALIVE(PID={train_pid})"
        except OSError:
            train_status = "DEAD"
            wlog(f"ALERT: training died at t={elapsed:.0f}s")

    # GPU utilization + memory
    gpu_info = "?"
    try:
        gpu_info = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader",
            shell=True, text=True, timeout=5,
        ).strip()
    except Exception:
        pass

    # Training progress (last log line)
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

    print(f"[{ts()}] {NAME} heartbeat "
          f"iter={iteration} elapsed={elapsed:.0f}s train={train_status}",
          flush=True)

    if train_status == "DEAD" and train_pid is not None:
        wlog("exiting early — training dead")
        break

    time.sleep(INTERVAL)

total = time.time() - start_time
wlog(f"EXIT total_elapsed={total:.0f}s iterations={iteration}")
next_ws = counter + 1
wlog(f"HANDOFF: start ws-{next_ws} with:")
wlog(f"  echo 'exec(open(\"/content/watchdog.py\").read())' | colab exec -s <name> --timeout 540 &")
