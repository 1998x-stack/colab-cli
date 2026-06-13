"""Test vLLM 0.9.2 on T4 Colab - might avoid transformers 5.x bug."""
import subprocess
import sys
import os

print("Installing vLLM 0.9.2...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "vllm==0.9.2",
     "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
    capture_output=True, text=True, timeout=600
)
print(f"pip rc={result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
    print("\nTrying cu124 instead...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "vllm==0.9.2",
         "--extra-index-url", "https://download.pytorch.org/whl/cu124"],
        capture_output=True, text=True, timeout=600
    )
    print(f"pip cu124 rc={result.returncode}")

import vllm
import torch
import transformers
print(f"vLLM: {vllm.__version__}")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
from vllm import LLM, SamplingParams

# Test with a simple model first
print("Testing SmolLM2-1.7B...")
try:
    llm = LLM(
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0.8, max_tokens=32),
    )
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"OK - VRAM peak={vram:.2f}GB")
    print(f"Output: {outputs[0].outputs[0].text[:80]}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {str(e)[:200]}")

print("Done.")
