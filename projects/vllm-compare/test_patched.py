"""vLLM 0.10.2 on T4 with monkey-patch for all_special_tokens_extended."""
import subprocess, sys, os

# Install vLLM 0.10.2 with cu128
print("Installing vLLM 0.10.2...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "vllm==0.10.2",
     "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
    capture_output=True, timeout=300
)

# Monkey-patch: add all_special_tokens_extended to tokenizers that lack it
import transformers
print(f"transformers: {transformers.__version__}")

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
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"torch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

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
