"""One-shot bootstrap for vLLM comparison benchmark.

Does pip install + spawn compare.py, all in one detached process
so the colab exec WebSocket can disconnect without killing work.
"""
import subprocess, sys, os

logfile = "/content/bootstrap.log"
with open(logfile, "w") as log:

    def log_print(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    log_print("[bootstrap] Installing deps (vLLM CUDA 12.8 + huggingface_hub)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "vllm>=0.10,<0.11", "huggingface_hub",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ], stdout=log, stderr=subprocess.STDOUT)
    log_print("[bootstrap] Deps installed.")

    log_print("[bootstrap] Spawning compare.py...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    with open("/content/vllm_compare.log", "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", "/content/compare.py"],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    log_print(f"[bootstrap] OK. PID={proc.pid} log=/content/vllm_compare.log")

print(f"Bootstrap done. PID={proc.pid} log={logfile}")
