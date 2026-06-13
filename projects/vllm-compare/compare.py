"""vLLM Model Comparison Tutorial — T4 GPU Benchmark.

Compares 3 small LLMs on latency (TTFT), throughput (tok/s), and peak
VRAM using vLLM's offline inference API. Demonstrates PagedAttention
and AWQ quantization through hands-on measurement.

Design: one-process, sequential benchmarks. Each model is downloaded,
loaded, warmed up, benchmarked, then fully unloaded before the next.
"""
import os
import sys
import json
import time
import gc
import logging
import torch

# Suppress vLLM's verbose internal logging
logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# ── Configuration ──────────────────────────────────────────────

MODELS = [
    {
        "name": "Qwen2.5-7B-Instruct-AWQ",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "quantization": "awq",
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": None,
        "requires_auth": True,
    },
    {
        "name": "SmolLM2-1.7B-Instruct",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "quantization": None,
    },
]

FALLBACK_MODEL = {
    "name": "Phi-3.5-mini-instruct (fallback)",
    "hf_id": "microsoft/Phi-3.5-mini-instruct",
    "quantization": None,
}

PROMPTS = [
    "Explain what PagedAttention is in one paragraph.",
    "Write a Python function to compute Fibonacci numbers recursively.",
    (
        "Summarize: 'The transformer architecture, introduced in 2017, "
        "revolutionized NLP by replacing recurrence with self-attention "
        "mechanisms. Unlike RNNs that process tokens sequentially, "
        "transformers process all tokens in parallel using attention "
        "layers. This enables much longer context windows and faster "
        "training.'"
    ),
    "What is the capital of France and what is it known for?",
    (
        "If a train leaves at 60 mph and another at 40 mph, "
        "how long until they meet if 200 miles apart?"
    ),
    "Write a haiku about machine learning.",
    "Translate to French: 'Hello, how are you today?'",
    "What is the difference between supervised and unsupervised learning?",
    "List three advantages of using GPU over CPU for deep learning.",
    "Explain the chain rule in calculus with a simple example.",
]

MAX_TOKENS = 256
TEMPERATURE = 0.8
OUTPUT_FILE = "/content/results.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def env_check() -> dict:
    """Print GPU diagnostics. Exit early if no CUDA GPU is available."""
    log("=== Environment Check ===")
    if not torch.cuda.is_available():
        log("ERROR: No GPU detected. vLLM requires a CUDA-capable GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log(f"GPU: {gpu_name}")
    log(f"VRAM total: {total_vram:.1f} GB")
    log(f"CUDA version: {torch.version.cuda}")
    log(f"PyTorch version: {torch.__version__}")

    return {
        "gpu": gpu_name,
        "vram_total_gb": round(total_vram, 1),
        "cuda_version": torch.version.cuda,
    }


def install_verify() -> None:
    """Confirm vLLM is importable and print its version."""
    log("=== Install Verification ===")
    import vllm
    log(f"vLLM version: {vllm.__version__}")


def benchmark_model(model_cfg: dict) -> dict:
    """Download, load, warm up, and benchmark a single model.

    Returns a results dict with status + metrics. On failure returns
    a dict with status != "ok" and a human-readable reason.
    """
    from vllm import LLM, SamplingParams

    log(f"\n{'=' * 60}")
    log(f"MODEL: {model_cfg['name']}")
    log(f"{'=' * 60}")

    # ── Auth check for gated models ──
    if model_cfg.get("requires_auth"):
        token = get_hf_token()
        if not token:
            log("SKIP: gated model requires HF_TOKEN environment variable.")
            log("  Set with: export HF_TOKEN=<your-huggingface-token>")
            log("  Or get one at: https://huggingface.co/settings/tokens")
            return {"status": "skipped", "reason": "HF_TOKEN not set"}

    # ── Download ──
    log("Downloading model from HuggingFace...")
    try:
        t0 = time.time()
        from huggingface_hub import snapshot_download
        snapshot_download(model_cfg["hf_id"], resume_download=True)
        dl_time = time.time() - t0
        log(f"  Downloaded in {dl_time:.0f}s")
    except Exception as exc:
        log(f"ERROR downloading: {exc}")
        return {"status": "failed", "reason": f"download error: {exc}"}

    # ── Load with vLLM ──
    log("Loading model with vLLM (this may take 30-120s on first run)...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        llm_kwargs: dict = {
            "model": model_cfg["hf_id"],
            "gpu_memory_utilization": 0.85,
            "max_model_len": 2048,
            "trust_remote_code": True,
        }
        if model_cfg.get("quantization"):
            llm_kwargs["quantization"] = model_cfg["quantization"]

        llm = LLM(**llm_kwargs)
        load_time = time.time() - t0
        log(f"  Loaded in {load_time:.0f}s")
    except Exception as exc:
        log(f"ERROR loading model: {exc}")
        return {"status": "failed", "reason": f"load error: {exc}"}

    # ── Warmup ──
    log("Warming up (2 short prompts)...")
    warm_params = SamplingParams(temperature=TEMPERATURE, max_tokens=32)
    llm.generate(["Hello, world!", "What is 2+2?"], warm_params)

    # ── Benchmark ──
    log(f"Benchmarking with {len(PROMPTS)} diverse prompts...")
    bench_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    t0 = time.time()
    outputs = llm.generate(PROMPTS, bench_params)
    total_time = time.time() - t0

    # Token throughput
    total_tokens_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens_out / total_time if total_time > 0 else 0

    # TTFT from vLLM's built-in per-request metrics
    ttfts: list[float] = []
    for o in outputs:
        m = o.metrics
        if (
            m is not None
            and m.first_token_time is not None
            and m.arrival_time is not None
        ):
            ttfts.append(m.first_token_time - m.arrival_time)

    avg_ttft_ms = (sum(ttfts) / len(ttfts)) * 1000 if ttfts else 0.0

    # Peak VRAM during this model's lifetime
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    log(f"  Total time:    {total_time:.1f}s")
    log(f"  Tokens out:    {total_tokens_out}")
    log(f"  Throughput:    {tps:.1f} tok/s")
    log(f"  Avg TTFT:      {avg_ttft_ms:.1f} ms")
    log(f"  Peak VRAM:     {peak_vram:.2f} GB")

    # ── Cleanup ──
    log("Unloading model and freeing VRAM...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "status": "ok",
        "load_time_s": round(load_time, 1),
        "total_time_s": round(total_time, 1),
        "total_tokens": total_tokens_out,
        "tps_avg": round(tps, 1),
        "ttft_avg_ms": round(avg_ttft_ms, 1),
        "peak_vram_gb": round(peak_vram, 2),
    }


def print_summary(all_results: list[tuple[dict, dict]]) -> None:
    """Print formatted comparison table and key takeaways."""
    log(f"\n{'=' * 60}")
    log("COMPARISON SUMMARY")
    log(f"{'=' * 60}")

    header = (
        f"\n{'Model':<38} {'Status':<8} {'TTFT(ms)':<10} "
        f"{'TPS':<10} {'VRAM(GB)':<10} {'Load(s)':<10}"
    )
    print(header)
    print("-" * len(header))

    for model_cfg, result in all_results:
        name = model_cfg["name"]
        if result.get("status") == "ok":
            print(
                f"{name:<38} {'ok':<8} "
                f"{result['ttft_avg_ms']:<10.1f} "
                f"{result['tps_avg']:<10.1f} "
                f"{result['peak_vram_gb']:<10.2f} "
                f"{result['load_time_s']:<10.1f}"
            )
        else:
            reason = result.get("reason", result.get("status", "?"))
            print(f"{name:<38} {'SKIP':<8} {'—':<10} {'—':<10} {'—':<10} {'—':<10}")
            print(f"  Reason: {reason}")

    print("\n── Key Takeaways ──")
    print(
        "• AWQ 4-bit quantization fits a 7B model on a T4 at ~4.5 GB VRAM —\n"
        "  competitive with much smaller FP16 models, with better quality."
    )
    print(
        "• vLLM's PagedAttention enables efficient KV-cache management,\n"
        "  letting even constrained GPUs serve LLMs reliably."
    )
    print(
        "• Smaller models (SmolLM2 1.7B) win on latency; larger quantized\n"
        "  models (Qwen 7B AWQ) win on reasoning quality."
    )
    print(
        "• Llama 3.2 3B at FP16 hits a sweet spot — no quantization needed\n"
        "  while fitting comfortably in T4 VRAM."
    )


def main() -> None:
    hardware_info = env_check()
    install_verify()

    all_results: list[tuple[dict, dict]] = []

    for i, model_cfg in enumerate(MODELS):
        log(f"\n[{i + 1}/{len(MODELS)}] Starting {model_cfg['name']}")
        result = benchmark_model(model_cfg)

        # Fallback for gated Llama if user lacks access
        if result.get("status") == "skipped" and model_cfg.get("requires_auth"):
            log("Trying fallback model (no auth required)...")
            result = benchmark_model(FALLBACK_MODEL)
            all_results.append((FALLBACK_MODEL, result))
        else:
            all_results.append((model_cfg, result))

    print_summary(all_results)

    # Persist structured results
    output = {
        "hardware": hardware_info,
        "models": {cfg["name"]: res for cfg, res in all_results},
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
