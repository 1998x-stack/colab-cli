# vLLM Model Comparison Tutorial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab tutorial that benchmarks 3 small LLMs with vLLM on a T4 GPU, comparing latency, throughput, and VRAM usage.

**Architecture:** Four files following existing project patterns. `launch.py` installs deps and spawns `compare.py` as a detached subprocess. `compare.py` runs the full benchmark loop — env check, per-model download/load/benchmark/unload, comparison table, JSON output. `check_progress.py` tails the log and shows results.

**Tech Stack:** vLLM (offline inference API), huggingface_hub (model download), torch (VRAM measurement), Python 3.10+

---

### Task 1: Create project directory and README.md

**Files:**
- Create: `projects/vllm-compare/README.md`

- [ ] **Step 1: Create project directory**

```bash
mkdir -p /Users/mx/Desktop/projects/colab-cli/projects/vllm-compare
```

- [ ] **Step 2: Write README.md**

Write to `projects/vllm-compare/README.md`:

```markdown
# vLLM Model Comparison Tutorial

Benchmarks 3 small LLMs with vLLM on a Colab T4 GPU (16GB VRAM). Compares latency (TTFT), throughput (tokens/sec), and peak VRAM across models with different quantization strategies.

## Models

| Model | Quantization | Approx VRAM |
|-------|-------------|-------------|
| Qwen2.5-7B-Instruct | AWQ 4-bit | ~4.5 GB |
| Llama-3.2-3B-Instruct | FP16 | ~6.5 GB |
| SmolLM2-1.7B-Instruct | FP16 | ~3.5 GB |
| Phi-3.5-mini-instruct (fallback) | FP16 | ~7.5 GB |

## Quick Start

```bash
# Provision T4 VM
colab new --gpu T4 -s vllm-compare

# Upload files
colab upload launch.py launch.py
colab upload compare.py compare.py

# Run (gated Llama model needs HF_TOKEN)
colab exec -f launch.py --timeout 120

# Check progress
colab upload check_progress.py check_progress.py
colab exec -f check_progress.py --timeout 15

# Download results
colab download /content/results.json .

# Clean up
colab stop -s vllm-compare
```

## Requirements

- Colab session with T4 GPU (free tier works)
- HuggingFace token for gated models (Llama-3.2-3B). Set via `HF_TOKEN` env var or the Colab secrets UI.
- If Llama is inaccessible, Phi-3.5-mini is used as fallback.

## Expected Results (approximate, T4)

| Model | TTFT (ms) | Tok/s | Peak VRAM (GB) |
|-------|-----------|-------|----------------|
| Qwen2.5-7B-AWQ | ~45 | ~85 | ~4.8 |
| Llama-3.2-3B | ~30 | ~120 | ~6.5 |
| SmolLM2-1.7B | ~15 | ~200 | ~3.5 |

## Tutorial Takeaways

- AWQ quantization (Qwen 7B) fits a 7B model on a T4 at 4.5GB VRAM — competitive with much smaller FP16 models.
- vLLM's PagedAttention enables efficient memory management even on constrained GPUs.
- Smaller models offer lower latency but weaker reasoning — choose based on your use case.
```

- [ ] **Step 3: Verify file exists**

```bash
ls -la /Users/mx/Desktop/projects/colab-cli/projects/vllm-compare/README.md
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli && git add projects/vllm-compare/README.md && git commit -m "docs: add README for vllm comparison tutorial"
```

---

### Task 2: Create launch.py

**Files:**
- Create: `projects/vllm-compare/launch.py`

- [ ] **Step 1: Write launch.py**

Write to `projects/vllm-compare/launch.py`:

```python
"""Launch vLLM comparison benchmark on Colab VM.

Installs vLLM + dependencies, then spawns compare.py as a detached
subprocess that survives kernel exec timeouts.
"""
import subprocess, sys, os

print("[launch] Installing Python dependencies (vLLM + huggingface_hub)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "vllm", "huggingface_hub",
])

print("[launch] Starting vLLM comparison benchmark...")
logfile = "/content/vllm_compare.log"
with open(logfile, "w") as f:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    proc = subprocess.Popen(
        [sys.executable, "-u", "/content/compare.py"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

print(f"[launch] OK. PID={proc.pid} log={logfile}")
```

- [ ] **Step 2: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli && git add projects/vllm-compare/launch.py && git commit -m "feat: add launch script for vllm comparison benchmark"
```

---

### Task 3: Create check_progress.py

**Files:**
- Create: `projects/vllm-compare/check_progress.py`

- [ ] **Step 1: Write check_progress.py**

Write to `projects/vllm-compare/check_progress.py`:

```python
"""Check vLLM comparison benchmark progress on Colab VM."""
import os, json, subprocess

logfile = "/content/vllm_compare.log"
results_file = "/content/results.json"

# Check if benchmark process is alive
result = subprocess.run(
    ["pgrep", "-f", "compare.py"], capture_output=True, text=True
)
if result.stdout.strip():
    print(f"Benchmark RUNNING — PID(s): {result.stdout.strip()}")
else:
    print("Benchmark NOT RUNNING (process not found)")

# Latest log lines
print("\n── Last 20 log lines ──")
if os.path.exists(logfile):
    with open(logfile) as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())
else:
    print("(no log file yet)")

# Results summary
print("\n── Results ──")
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    for name, res in data.get("models", {}).items():
        status = res.get("status", "?")
        if status == "ok":
            print(f"  {name}: TTFT={res['ttft_avg_ms']:.1f}ms  TPS={res['tps_avg']:.1f}  VRAM={res['peak_vram_gb']:.2f}GB")
        else:
            print(f"  {name}: {status} — {res.get('reason', '')}")
else:
    print("(no results yet)")
```

- [ ] **Step 2: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli && git add projects/vllm-compare/check_progress.py && git commit -m "feat: add progress checker for vllm comparison benchmark"
```

---

### Task 4: Create compare.py (main benchmark script)

**Files:**
- Create: `projects/vllm-compare/compare.py`

- [ ] **Step 1: Write compare.py — imports, config, and helpers**

Write to `projects/vllm-compare/compare.py`:

```python
"""vLLM Model Comparison Tutorial — T4 GPU Benchmark.

Compares 3 small LLMs on latency (TTFT), throughput (tok/s), and peak
VRAM using vLLM's offline inference API. Demonstrates PagedAttention
and AWQ quantization through hands-on measurement.

Design: one-process, sequential benchmarks. Each model is downloaded,
loaded, warmed up, benchmarked, then fully unloaded before the next.
"""
import os, sys, json, time, gc, logging
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
```

- [ ] **Step 2: Write compare.py — env_check() and install_verify()**

Append to `projects/vllm-compare/compare.py`:

```python
def env_check() -> dict:
    """Print GPU diagnostics. Exit early if no CUDA GPU is available."""
    log("=== Environment Check ===")
    if not torch.cuda.is_available():
        log("ERROR: No GPU detected. vLLM requires a CUDA-capable GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
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
```

- [ ] **Step 3: Write compare.py — benchmark_model()**

Append to `projects/vllm-compare/compare.py`:

```python
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
```

- [ ] **Step 4: Write compare.py — print_summary() and main()**

Append to `projects/vllm-compare/compare.py`:

```python
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
```

- [ ] **Step 5: Verify file is syntactically valid Python**

```bash
python3 -c "import ast; ast.parse(open('projects/vllm-compare/compare.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli && git add projects/vllm-compare/compare.py && git commit -m "feat: add vllm model comparison benchmark script"
```

---

### Task 5: End-to-end file structure verification

- [ ] **Step 1: Verify all files exist and have content**

```bash
wc -l /Users/mx/Desktop/projects/colab-cli/projects/vllm-compare/*.py /Users/mx/Desktop/projects/colab-cli/projects/vllm-compare/README.md
```

- [ ] **Step 2: Verify Python syntax for all .py files**

```bash
for f in /Users/mx/Desktop/projects/colab-cli/projects/vllm-compare/*.py; do echo "=== $f ===" && python3 -c "import ast; ast.parse(open('$f').read()); print('OK')"; done
```

Expected: OK for each file.

- [ ] **Step 3: Verify project matches existing patterns**

```bash
diff <(head -3 /Users/mx/Desktop/projects/colab-cli/projects/rl-sac/launch.py) <(head -3 /Users/mx/Desktop/projects/colab-cli/projects/vllm-compare/launch.py) && echo "launch.py pattern matches"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/mx/Desktop/projects/colab-cli && git add projects/vllm-compare/ && git commit -m "chore: verify vllm-compare project structure"
```
