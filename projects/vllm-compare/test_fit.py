"""Quick test: which models fit on T4 with vLLM 0.10.2."""
import os, time
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

from vllm import LLM, SamplingParams
import torch

MODELS = [
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 4096),
    ("SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 4096),
    ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 4096),
    ("Qwen2.5-7B-Instruct-AWQ", "Qwen/Qwen2.5-7B-Instruct-AWQ", 4096),
]

for name, hf_id, max_len in MODELS:
    print(f"Testing {name}...", flush=True)
    try:
        t0 = time.time()
        llm = LLM(
            model=hf_id,
            max_model_len=max_len,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
        )
        load_time = time.time() - t0

        outputs = llm.generate(
            ["Hello, my name is"],
            SamplingParams(temperature=0.8, max_tokens=32),
        )

        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OK - load={load_time:.1f}s, VRAM peak={vram:.2f}GB")
        print(f"  Output: {outputs[0].outputs[0].text[:100]}")

        del llm
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        torch.cuda.empty_cache()

print("Done.")
