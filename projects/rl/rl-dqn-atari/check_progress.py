"""Check DQN Atari training progress on Colab VM.

Reports: process status, latest log lines, checkpoint files, and latest metrics.
Run via: colab exec -f check_progress.py --timeout 30
"""

import os
import subprocess

LOGFILE = "/content/dqn_train.log"
OUTPUT_DIR = "/content/dqn-output"

# 1. Process status
result = subprocess.run(
    ["pgrep", "-f", "train.py"], capture_output=True, text=True,
)
if result.stdout.strip():
    print(f"Status: RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Status: NOT RUNNING (or just completed)")

# 2. Latest log lines
print(f"\n── Last 15 lines ({LOGFILE}) ──")
if os.path.exists(LOGFILE):
    lines = open(LOGFILE).readlines()
    for line in lines[-15:]:
        print(line.rstrip())
    print(f"\n(total: {len(lines)} lines)")
else:
    print("(no log file yet)")

# 3. Checkpoints
print(f"\n── Checkpoints ({OUTPUT_DIR}) ──")
if os.path.exists(OUTPUT_DIR):
    ckpts = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pt")])
    for ck in ckpts[-5:]:
        path = os.path.join(OUTPUT_DIR, ck)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {ck} ({size_kb:.0f} KB)")
    if not ckpts:
        print("  (no checkpoints yet)")

    # 4. Metrics JSON peek
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        m = json.load(open(metrics_path))
        print("\n── Metrics ──")
        print(f"  Best return: {m.get('best_return', '?')}")
        print(f"  Total episodes: {m.get('total_episodes', '?')}")
        print(f"  Total steps: {m.get('total_steps', '?')}")
        print(f"  Train time: {m.get('train_time_seconds', 0)/60:.1f}m")
        hist = m.get("episode_history", [])
        if hist:
            last = hist[-1]
            print(f"  Last ep: {last['episode']} | return: {last['return']} | "
                  f"avg100: {last.get('avg_return_100', '?')} | eps: {last['epsilon']:.3f}")
else:
    print("  (no output dir yet)")
