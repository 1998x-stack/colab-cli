#!/usr/bin/env python3
"""Check progress of running SWE-agent on Colab."""

import json
import os
import sys
import time


def check():
    # Check heartbeat
    try:
        with open("/content/heartbeat.json") as f:
            hb = json.load(f)
        elapsed = time.time() - hb["start_time"]
        print(f"Uptime: {elapsed/60:.1f} min")
        print(f"vLLM PID: {hb['vllm_pid']}")
        print(f"Agent PID: {hb['agent_pid']}")
    except FileNotFoundError:
        print("No heartbeat file — launch may have failed.")
        return

    # Check processes alive
    for name, pid in [("vLLM", hb["vllm_pid"]), ("Agent", hb["agent_pid"])]:
        try:
            os.kill(pid, 0)
            print(f"{name}: RUNNING (PID {pid})")
        except OSError:
            print(f"{name}: DEAD")

    # Tail agent log
    log_path = "/content/agent_run.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print(f"\n--- Agent log (last 20 lines, {len(lines)} total) ---")
        for line in lines[-20:]:
            print(line.rstrip())

    # Check for output
    metrics_path = "/content/output/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"\n=== RESULTS ===")
        print(json.dumps(metrics, indent=2))
    else:
        print("\nNo metrics.json yet — agent still running.")

    # Check vLLM log for errors
    vllm_log = "/content/vllm.log"
    if os.path.exists(vllm_log):
        with open(vllm_log) as f:
            content = f.read()
        if "ERROR" in content or "CUDA out of memory" in content:
            print("\n!!! vLLM errors detected !!!")
            for line in content.split("\n")[-5:]:
                print(f"  {line}")


if __name__ == "__main__":
    check()
