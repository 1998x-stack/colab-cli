"""One-shot bootstrap for vLLM RAG pipeline.

Pip install + spawn server + spawn eval, all in one detached process.
Colab exec WebSocket can disconnect without killing work.
"""
import subprocess, sys, os

logfile = "/content/bootstrap.log"
with open(logfile, "w") as log:

    def log_print(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    log_print("[bootstrap] Installing deps (vLLM CUDA 12.8 + RAG stack)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "vllm>=0.10,<0.11", "chromadb", "sentence-transformers", "datasets", "requests",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ], stdout=log, stderr=subprocess.STDOUT)
    log_print("[bootstrap] Deps installed.")

    log_print("[bootstrap] Spawning vLLM server...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    with open("/content/vllm_server.log", "w") as f:
        srv = subprocess.Popen(
            [sys.executable, "-u", "/content/server.py"],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    log_print(f"[bootstrap] Server PID={srv.pid}")

    log_print("[bootstrap] Spawning RAG eval...")
    with open("/content/rag_eval.log", "w") as f:
        evl = subprocess.Popen(
            [sys.executable, "-u", "/content/rag_eval.py"],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    log_print(f"[bootstrap] Eval PID={evl.pid}")

print(f"Bootstrap done. Server PID={srv.pid} Eval PID={evl.pid} log={logfile}")
