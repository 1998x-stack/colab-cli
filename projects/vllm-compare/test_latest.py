"""Test latest vLLM on T4 Colab."""
import subprocess, sys, os

# Install latest vLLM
print("Installing latest vLLM...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "vllm",
     "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
    capture_output=True, text=True, timeout=600
)
print(f"pip rc={result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
    sys.exit(1)

import vllm, torch, transformers
print(f"vLLM: {vllm.__version__}")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Fix torchvision if needed
try:
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
except:
    print("torchvision missing, installing...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "torchvision", "Pillow",
         "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
        capture_output=True, timeout=300
    )

# Test models
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
