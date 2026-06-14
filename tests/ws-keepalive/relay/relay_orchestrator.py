"""Fire-and-forget watchdog relay orchestrator for long Colab GPU sessions.

Each watchdog is launched as an independent OS process via
subprocess.Popen(start_new_session=True). The orchestrator can be killed
without affecting running watchdogs — they survive independently.

Usage:
    python relay_orchestrator.py <session_name> [--duration 25] [--window 7]

What it does:
    1. Uploads fake_train_25min.py to VM
    2. Launches ws-1 (spawns training + 7-min watchdog)
    3. Launches ws-2 at T+6min, ws-3 at T+13min, ws-4 at T+20min, ws-5 at T+24min (buffer)
    4. Waits for all watchdogs, then downloads results
"""
import subprocess
import time
import sys
import os
import argparse
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def run(cmd, check=True):
    """Run a command, print output. Used for colab new/upload/stop."""
    print(f"[{ts()}] RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        print(f"[{ts()}] ERROR: command failed with code {result.returncode}")
        sys.exit(1)
    return result

def launch_watchdog(session, script_name):
    """Launch colab exec as an INDEPENDENT OS process. Returns Popen."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[{ts()}] FATAL: script not found: {script_path}")
        sys.exit(1)

    cmd = [
        "colab", "exec", "-s", session,
        "-f", script_path,
        "--timeout", "540"  # 9 min for 7-min watchdog
    ]
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[{ts()}] LAUNCHED {script_name} (PID={proc.pid})")
    return proc

def main():
    parser = argparse.ArgumentParser(description="Colab GPU relay orchestrator")
    parser.add_argument("session", help="Colab session name")
    parser.add_argument("--duration", type=int, default=25, help="Training duration in minutes")
    parser.add_argument("--window", type=int, default=7, help="Watchdog window in minutes")
    parser.add_argument("--account", default=None, help="Colab account (home dir suffix, e.g. 'cb')")
    args = parser.parse_args()

    SESSION = args.session
    DURATION_MIN = args.duration
    WINDOW_MIN = args.window
    OVERLAP_MIN = 1  # 1 minute overlap between watchdogs

    # Calculate watchdogs needed: ceil(duration / window) + 1 buffer
    num_watchdogs = (DURATION_MIN + WINDOW_MIN - 1) // WINDOW_MIN + 1

    print(f"[{ts()}] === Relay Orchestrator ===")
    print(f"[{ts()}] Session: {SESSION}")
    print(f"[{ts()}] Training: {DURATION_MIN} min | Window: {WINDOW_MIN} min | Overlap: {OVERLAP_MIN} min")
    print(f"[{ts()}] Watchdogs: {num_watchdogs} (ws-1 launches training, ws-2..{num_watchdogs} monitor)")

    # Build colab command prefix for account switching
    colab_prefix = []
    if args.account:
        account_home = os.path.expanduser(f"~/colab-accounts/account-{args.account}")
        colab_prefix = ["env", f"HOME={account_home}", "colab"]
    else:
        colab_prefix = ["colab"]

    # ── Step 1: Upload training script to VM ──────────────
    train_script = os.path.join(SCRIPT_DIR, "fake_train_25min.py")
    print(f"\n[{ts()}] Step 1: Upload {train_script}")
    run(colab_prefix + ["upload", train_script, "/content/fake_train_25min.py"])

    # ── Step 2: Launch ws-1 (spawns training + 7-min watchdog) ──
    print(f"\n[{ts()}] Step 2: Launch ws-1 (training spawner + watchdog)")
    procs = []
    procs.append(launch_watchdog(SESSION, "launch_train.py"))
    print(f"[{ts()}] ws-1 started — training is spawning on VM")

    # ── Step 3: Launch ws-2..N on schedule ────────────────
    total_wait = 0
    for i in range(2, num_watchdogs + 1):
        launch_at = (i - 1) * (WINDOW_MIN - OVERLAP_MIN) * 60
        wait = launch_at - total_wait
        print(f"\n[{ts()}] Waiting {wait}s until ws-{i} launch (T+{launch_at//60}min)...")
        if wait > 0:
            time.sleep(wait)
            total_wait += wait

        procs.append(launch_watchdog(SESSION, "watchdog.py"))
        print(f"[{ts()}] ws-{i} queued — WebSocket connected, code waiting in kernel queue")

    # ── Step 4: Wait for training to complete ─────────────
    total_test_time = DURATION_MIN * 60 + 60  # training + 1 min buffer
    remaining = total_test_time - total_wait
    print(f"\n[{ts()}] All {num_watchdogs} watchdogs launched.")
    print(f"[{ts()}] Waiting {remaining}s for training to complete ({DURATION_MIN} min total)...")
    time.sleep(remaining)

    # ── Step 5: Download results ──────────────────────────
    print(f"\n[{ts()}] Step 5: Downloading results...")
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Download train log
    run(colab_prefix + ["download", "-s", SESSION,
         "/content/relay-test-output/logs/train.log",
         f"{output_dir}/train.log"])

    # Download watchdog log
    run(colab_prefix + ["download", "-s", SESSION,
         "/content/relay-test-output/logs/watchdog.log",
         f"{output_dir}/watchdog.log"])

    # ── Step 6: Verify ────────────────────────────────────
    print(f"\n[{ts()}] === Results ===")
    train_log = f"{output_dir}/train.log"
    if os.path.exists(train_log):
        with open(train_log) as f:
            content = f.read()
        lines = content.strip().split("\n")
        print(f"Train log: {len(lines)} lines")
        if lines:
            print(f"  First: {lines[0][:120]}")
            print(f"  Last:  {lines[-1][:120]}")
        if "TRAIN_COMPLETE" in content:
            elapsed_line = [l for l in lines if "total_elapsed" in l]
            if elapsed_line:
                print(f"\n  *** SUCCESS: Training completed! {elapsed_line[-1].strip()}")
            else:
                print(f"\n  *** SUCCESS: Training completed!")
        else:
            print(f"\n  *** FAILED: Training did not complete")

    wd_log = f"{output_dir}/watchdog.log"
    if os.path.exists(wd_log):
        with open(wd_log) as f:
            content = f.read()
        lines = content.strip().split("\n")
        print(f"\nWatchdog log: {len(lines)} lines")
        exit_lines = [l for l in lines if "EXIT" in l]
        print(f"  Watchdog completions: {len(exit_lines)}")
        for l in exit_lines:
            print(f"  {l.strip()[:120]}")

    print(f"\n[{ts()}] Done. Output in {output_dir}/")

if __name__ == "__main__":
    main()
