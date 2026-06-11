"""Monitor Colab training progress.

Print heartbeat age, process liveness, and last 20 log lines.
Overridable via env vars: CHECK_LOG (log path), CHECK_PROC (process name filter).
"""
import os, json, subprocess, sys, time

LOG = os.environ.get("CHECK_LOG", "/content/run.log")
PROC = os.environ.get("CHECK_PROC", "run.py")


def main() -> None:
    # Heartbeat
    hb = "/content/heartbeat.json"
    try:
        mtime = os.path.getmtime(hb)
        age = time.time() - mtime
        print(f"Heartbeat: {age:.0f}s ago {'⚠️' if age > 120 else '✅'}")
    except FileNotFoundError:
        print("Heartbeat: not found ⚠️")

    # Process
    try:
        result = subprocess.run(["pgrep", "-f", PROC], capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        if pids:
            print(f"Process ({PROC}): alive ✅  PIDs: {', '.join(pids)}")
        else:
            print(f"Process ({PROC}): NOT FOUND ⚠️")
    except Exception as e:
        print(f"Process check failed: {e}")

    # Log tail
    try:
        with open(LOG) as f:
            lines = f.readlines()
        recent = lines[-20:] if len(lines) > 20 else lines
        print(f"\n── Log tail ({LOG}) ({len(recent)}/{len(lines)} lines) ──")
        for line in recent:
            print(line.rstrip())
    except FileNotFoundError:
        print(f"\nLog ({LOG}): not found")


if __name__ == "__main__":
    main()
