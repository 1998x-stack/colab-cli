"""Check training progress on Colab VM — PID alive + log tail + metrics snapshot.

Env vars (all optional):
  CHECK_LOG: log file to tail (default: /content/cnn-explainer-output/logs/launch.log)
  CHECK_CSV: metrics CSV to report (default: /content/cnn-explainer-output/metrics.csv)
  CHECK_PNGS: directory to list PNGs (default: /content/cnn-explainer-output/pngs)
"""

import os
import subprocess

LOG = os.environ.get("CHECK_LOG", "/content/cnn-explainer-output/logs/launch.log")
CSV = os.environ.get("CHECK_CSV", "/content/cnn-explainer-output/metrics.csv")
PNGS = os.environ.get("CHECK_PNGS", "/content/cnn-explainer-output/pngs")


def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10).decode()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.decode() if e.output else str(e)}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


print("=== GPU Status ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'no GPU'"))

print("\n=== Python Process ===")
print(run("ps aux | grep -E 'python.*train' | grep -v grep || echo 'no train process found'"))

print("\n=== Log Tail (last 10 lines) ===")
if os.path.exists(LOG):
    lines = open(LOG).readlines()
    for line in lines[-10:
        ]:
        print(f"  {line.rstrip()}")
    print(f"  --- {len(lines)} total lines ---")
else:
    print(f"  {LOG} not found")

print("\n=== Metrics CSV (last 3 rows) ===")
if os.path.exists(CSV):
    lines = open(CSV).readlines()
    for line in lines[-4:
        ]:  # header + last 3
        print(f"  {line.rstrip()}")
else:
    print(f"  {CSV} not found")

print("\n=== Output PNGs ===")
if os.path.exists(PNGS):
    pngs = sorted([f for f in os.listdir(PNGS) if f.endswith(".png")])
    for p in pngs:
        sz = os.path.getsize(os.path.join(PNGS, p))
        print(f"  {p}  ({sz/1024:.0f} KB)")
else:
    print(f"  {PNGS} not found")

print("\n=== Done ===")
