#!/usr/bin/env python3
"""Colab bootstrap: install deps, start vLLM, spawn agent, save results."""

import subprocess
import sys
import os
import time
import json


def main():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Step 1: Install dependencies
    print("[launch] Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "openai", "pyyaml", "jinja2", "matplotlib",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "vllm", "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ])

    # Step 2: Download model (if not cached)
    print("[launch] Starting vLLM server with Qwen2.5-7B-Instruct-AWQ...")
    vllm_log = open("/content/vllm.log", "w")
    vllm_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--dtype", "auto",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
            "--port", "8000",
        ],
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    print(f"[launch] vLLM PID={vllm_proc.pid}")

    # Step 3: Wait for vLLM to be ready
    print("[launch] Waiting for vLLM server...")
    from models import check_vllm_health
    for i in range(120):
        time.sleep(10)
        if check_vllm_health():
            print("[launch] vLLM ready!")
            break
        print(f"[launch] Waiting... ({i*10}s)")
    else:
        print("[launch] ERROR: vLLM server did not start within 20 min")
        sys.exit(1)

    # Step 4: Upload project files
    # Files are already uploaded to /content/ by colab upload

    # Step 5: Run agent
    print("[launch] Starting agent...")
    agent_log = open("/content/agent_run.log", "w")
    agent_proc = subprocess.Popen(
        [sys.executable, "-u", "/content/run.py"],
        stdout=agent_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd="/content",
    )
    print(f"[launch] Agent PID={agent_proc.pid} log=/content/agent_run.log")

    # Write heartbeat info
    with open("/content/heartbeat.json", "w") as f:
        json.dump({
            "vllm_pid": vllm_proc.pid,
            "agent_pid": agent_proc.pid,
            "start_time": time.time(),
        }, f)

    print("[launch] Done. Agent running in background.")


if __name__ == "__main__":
    main()
