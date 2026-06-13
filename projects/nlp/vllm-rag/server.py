"""Start vLLM OpenAI-compatible API server on port 8000.

Serves Qwen2.5-7B-Instruct-AWQ for the RAG client to query.
Uses the vLLM CLI entrypoint which handles all engine config.
"""
import subprocess
import sys

print("[server] Starting vLLM API server...")
print("[server] Model: Qwen/Qwen2.5-7B-Instruct-AWQ (AWQ 4-bit)")
print("[server] Port: 8000")

args = [
    sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "--quantization", "awq",
    "--port", "8000",
    "--host", "0.0.0.0",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "2048",
    "--trust-remote-code",
]

subprocess.run(args)
