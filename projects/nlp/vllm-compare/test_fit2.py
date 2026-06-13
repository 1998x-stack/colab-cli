"""Quick test: which models fit on T4 with vLLM 0.10.2."""
import subprocess
import sys
import os

# Upgrade transformers for tokenizer compat
print("Upgrading transformers...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.52.0"],
    capture_output=True, timeout=300
)

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
from vllm import LLM, SamplingParams
import torch

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
