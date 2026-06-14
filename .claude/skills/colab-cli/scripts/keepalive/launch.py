"""Launch training + ws-1 watchdog (7-min WebSocket keepalive window).

Upload train.py + watchdog.py + log_utils.py + plot_utils.py to /content/ first.
Then: colab exec -f launch.py --timeout 540

This script:
1. pip installs DEPS
2. Spawns SCRIPT as detached subprocess (survives colab exec exit)
3. Writes train.pid for watchdog monitoring
4. Enters 7-min ws-1 watchdog loop (GPU + training health + log tail)
5. Exits with handoff message — ws-2 must already be queued at T+6
"""
import subprocess, sys, os, time
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────
SCRIPT = "train.py"
DEPS = []                               # e.g. ["gymnasium", "torchvision"]
OUT_DIR = None                          # None = auto-detect via /content/<SCRIPT stem>-output
DURATION = 420                          # 7 minutes
INTERVAL = 30                           # check every 30 seconds
# ────────────────────────────────────────────────────────

if OUT_DIR is None:
    name = os.path.splitext(SCRIPT)[0]
    OUT_DIR = f"/content/{name}-output"

LOG = f"{OUT_DIR}/logs/watchdog.log"
COUNTER_FILE = f"{OUT_DIR}/watchdog_counter"
TRAIN_PID_FILE = f"{OUT_DIR}/train.pid"

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/pngs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)

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
wlog(f"LAUNCH pid={os.getpid()} script={SCRIPT}")
wlog("=" * 50)

# ── Step 1: GPU check ──────────────────────────────────
try:
    import torch
    wlog(f"GPU={torch.cuda.get_device_name(0)} "
         f"mem={torch.cuda.get_device_properties(0).total_mem//1024**3}GB "
         f"cuda={torch.cuda.is_available()}")
except Exception as e:
    wlog(f"GPU check failed: {e}")

# ── Step 2: pip install ────────────────────────────────
if DEPS:
    wlog(f"Installing deps: {DEPS}")
    for pkg in DEPS:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"])

# ── Step 3: Launch training ────────────────────────────
wlog(f"Launching /content/{SCRIPT} ...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open(f"{OUT_DIR}/logs/train.log", "w") as log_f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}"],
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
            wlog(f"train.log tail: {f.read()[-500:]}")
    sys.exit(1)

# ── Step 4: ws-1 watchdog loop (7 min) ─────────────────
start_time = time.time()
iteration = 0
wlog(f"Watchdog START duration={DURATION}s interval={INTERVAL}s")

while time.time() - start_time < DURATION:
    iteration += 1
    elapsed = time.time() - start_time

    # Training alive?
    train_status = "?"
    try:
        os.kill(proc.pid, 0)
        train_status = f"ALIVE(PID={proc.pid})"
    except OSError:
        train_status = "DEAD"
        wlog(f"ALERT: training died at t={elapsed:.0f}s")

    # GPU utilization
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

    # Heartbeat keeps WebSocket output stream active
    print(f"[{ts()}] {NAME} heartbeat "
          f"iter={iteration} elapsed={elapsed:.0f}s train={train_status}",
          flush=True)

    if train_status == "DEAD":
        wlog("Training died — exiting early")
        break

    time.sleep(INTERVAL)

total = time.time() - start_time
wlog(f"EXIT total_elapsed={total:.0f}s iterations={iteration}")
wlog("HANDOFF: ws-2 should already be queued. If not, start now:")
wlog("  echo 'exec(open(\"/content/watchdog.py\").read())' | colab exec -s <name> --timeout 540 &")
wlog("=" * 50)
