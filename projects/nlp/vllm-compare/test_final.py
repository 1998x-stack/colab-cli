"""Final vLLM test on T4 — fix tokenizer + multiprocessing for Colab."""
import os

# Must be set BEFORE any CUDA/torch import
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

import subprocess
import sys

print("Installing vLLM 0.10.2...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "vllm==0.10.2",
     "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
    capture_output=True, timeout=300
)

# Monkey-patch for transformers 5.x compat
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
_orig_init = PreTrainedTokenizerBase.__init__
def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    if not hasattr(self, "all_special_tokens_extended"):
        self.all_special_tokens_extended = []
PreTrainedTokenizerBase.__init__ = _patched_init

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
from vllm import LLM, SamplingParams
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Test just one model quickly
print("Testing SmolLM2-1.7B...")
try:
    llm = LLM(
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(temperature=0.8, max_tokens=32),
    )
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"SUCCESS! VRAM peak={vram:.2f}GB")
    print(f"Output: {outputs[0].outputs[0].text[:80]}")
except Exception as e:
    import traceback
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("Done.")
