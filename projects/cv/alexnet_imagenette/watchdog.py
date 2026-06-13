"""VM-side watchdog: writes /content/heartbeat.json every 30s.
Exits when /content/watchdog_stop exists (train.py creates this on completion).
"""

import json
import os
import time

HEARTBEAT_PATH = "/content/heartbeat.json"
STOP_PATH = "/content/watchdog_stop"
INTERVAL = 30

print(f"[watchdog] Started, writing to {HEARTBEAT_PATH} every {INTERVAL}s", flush=True)

while not os.path.exists(STOP_PATH):
    t0 = time.time()
    # Read latest heartbeat from train.py (it writes on each epoch)
    if os.path.exists(HEARTBEAT_PATH):
        try:
            with open(HEARTBEAT_PATH) as f:
                existing = json.load(f)
            existing["watchdog_seen"] = time.time()
            with open(HEARTBEAT_PATH, "w") as f:
                json.dump(existing, f)
        except (json.JSONDecodeError, IOError):
            pass

    elapsed = INTERVAL - (time.time() - t0)
    if elapsed > 0:
        time.sleep(elapsed)

# Final heartbeat
heartbeat = {"status": "done", "epoch": 0, "train_loss": None, "val_acc": None,
             "elapsed_seconds": 0.0, "flops_consumed_tflops": 0.0, "timestamp": time.time()}
with open(HEARTBEAT_PATH, "w") as f:
    json.dump(heartbeat, f)
print("[watchdog] Stopped", flush=True)
