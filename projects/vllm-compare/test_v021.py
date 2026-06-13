"""Test vLLM 0.21.0 on T4 — should have transformers 5.x fix (PR #30566)."""
import subprocess
import sys
import os

print("Installing vLLM 0.21.0...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "vllm==0.21.0",
     "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
    capture_output=True, text=True, timeout=600
)
print(f"pip rc={result.returncode}")
if result.returncode != 0:
    # Try without cu128
    print("Trying PyPI default...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "vllm==0.21.0"],
        capture_output=True, text=True, timeout=600
    )
    print(f"pip default rc={result.returncode}")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
        sys.exit(1)

import vllm
import torch
import transformers
print(f"vLLM: {vllm.__version__}")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
from vllm import LLM, SamplingParams

models = [
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 4096),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", 4096),
    ("Qwen2.5-7B-AWQ", "Qwen/Qwen2.5-7B-Instruct-AWQ", 4096),
]

for name, hf_id, max_len in models:
    print(f"Testing {name}...", flush=True)
    try:
        llm = LLM(
            model=hf_id,
            max_model_len=max_len,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
        outputs = llm.generate(
            ["Hello, my name is"],
            SamplingParams(temperature=0.8, max_tokens=32),
        )
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OK - VRAM peak={vram:.2f}GB")
        print(f"  Output: {outputs[0].outputs[0].text[:80]}")
        del llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:150]}")
        torch.cuda.empty_cache()

print("Done.")
