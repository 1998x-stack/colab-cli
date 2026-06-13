#!/usr/bin/env python3
"""Launch DDPG vs TD3 MuJoCo training as a detached subprocess on Colab VM."""
import subprocess
import sys
import os
import time

DEPS = ["gymnasium[mujoco]", "matplotlib"]
SCRIPT = "train.py"
LOG = "/content/ddpg-td3-mujoco-output/launch.log"

print("=== DDPG vs TD3 MuJoCo Launcher ===")
print(f"Installing: {DEPS}")

for dep in DEPS:
    print(f"  pip install {dep} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print(f"  pip install {dep}: OK")

# Verify mujoco import works
print("  verifying mujoco import ...")
subprocess.check_call([sys.executable, "-c", "import mujoco; print(f'  mujoco {mujoco.__version__} OK')"])
print("  verifying gymnasium[mujoco] envs ...")
subprocess.check_call([sys.executable, "-c", """
import gymnasium as gym
for e in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    env = gym.make(e)
    print(f'  {e}: obs={env.observation_space.shape} act={env.action_space.shape}')
    env.close()
"""])

os.makedirs("/content/ddpg-td3-mujoco-output", exist_ok=True)

print(f"\nLaunching {SCRIPT} detached ...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["MUJOCO_GL"] = "egl"  # headless rendering

with open(LOG, "w") as f:
    proc = subprocess.Popen(
        [sys.executable, "-u", f"/content/{SCRIPT}"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"OK. PID={proc.pid}  log={LOG}")
print("Output root: /content/ddpg-td3-mujoco-output/")

time.sleep(5)
try:
    os.kill(proc.pid, 0)
    print(f"Process {proc.pid} is alive.")
except OSError:
    print(f"WARNING: Process {proc.pid} died! Check log:")
    try:
        with open(LOG) as f:
            print(f.read()[-2000:])
    except Exception:
        pass
