#!/usr/bin/env python3
"""Generate fat launch.py by embedding all source files as base64.

Usage: python build_launch.py [--first] > launch.py
       colab exec -f launch.py --timeout 120

The generated launch.py:
  1. Decodes and writes all project files to /content/
  2. Drive mounts
  3. Spawns train.py as a detached subprocess (nohup)
"""

import base64
import sys

FILES = ["config.py", "game.py", "model.py", "mcts.py", "train.py"]

LAUNCH_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated launcher — decode sources, mount Drive, spawn training."""
import os, sys, base64, subprocess, time

FILES = {filemap}

def decode_all():
    for name, b64 in FILES.items():
        path = f"/content/{{name}}"
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  Wrote {{path}} ({{os.path.getsize(path)}} bytes)")

def ensure_drive():
    for attempt in range(3):
        if os.path.exists("/content/drive/MyDrive"):
            print("  Drive already mounted")
            return True
        print(f"  Mounting Drive (attempt {{attempt+1}}/3)...")
        try:
            subprocess.run(["colab", "drivemount"], check=True, timeout=30)
            time.sleep(5)
            if os.path.exists("/content/drive/MyDrive"):
                print("  Drive mounted OK")
                return True
        except Exception as e:
            print(f"  Drive mount failed: {{e}}")
            time.sleep(3)
    print("  WARNING: Drive mount failed after 3 attempts — checkpoints will be local only")
    return False

def main():
    print("=== AlphaGo Launch ===")
    print("Decoding source files...")
    decode_all()

    print("Setting up Drive...")
    ensure_drive()

    os.makedirs("/content/drive/MyDrive/alphago-checkpoints", exist_ok=True)

    print("Spawning train.py (detached, nohup)...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Create launch log
    log = open("/content/launch.log", "w")

    cmd = [sys.executable, "-u", "/content/train.py"]
    if {first}:
        cmd.append("--first")

    proc = subprocess.Popen(
        cmd,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd="/content",
    )
    print(f"Launched: PID={{proc.pid}}")
    print(f"Monitor: tail -f /content/alphago-output/logs/train.log")
    print("Done. Training runs in background.")

if __name__ == "__main__":
    main()
'''


def build(first_session: bool = False) -> str:
    filemap = {}
    for fname in FILES:
        with open(fname, "r") as f:
            content = f.read()
        encoded = base64.b64encode(content.encode()).decode()
        filemap[fname] = encoded

    filemap_str = "{\n"
    for name, b64 in filemap.items():
        filemap_str += f'        "{name}": "{b64}",\n'
    filemap_str += "    }"

    return LAUNCH_TEMPLATE.format(filemap=filemap_str, first=str(first_session))


if __name__ == "__main__":
    first = "--first" in sys.argv
    print(build(first_session=first))
