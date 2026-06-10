"""Check vLLM comparison benchmark progress on Colab VM."""
import os, json, subprocess

logfile = "/content/vllm_compare.log"
results_file = "/content/results.json"

# Check if benchmark process is alive
result = subprocess.run(
    ["pgrep", "-f", "compare.py"], capture_output=True, text=True
)
if result.stdout.strip():
    print(f"Benchmark RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Benchmark NOT RUNNING (process not found)")

# Latest log lines
print("\n── Last 20 log lines ──")
if os.path.exists(logfile):
    with open(logfile) as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())
else:
    print("(no log file yet)")

# Results summary
print("\n── Results ──")
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    for name, res in data.get("models", {}).items():
        status = res.get("status", "?")
        if status == "ok":
            print(f"  {name}: TTFT={res['ttft_avg_ms']:.1f}ms  TPS={res['tps_avg']:.1f}  VRAM={res['peak_vram_gb']:.2f}GB")
        else:
            print(f"  {name}: {status} — {res.get('reason', '')}")
else:
    print("(no results yet)")
