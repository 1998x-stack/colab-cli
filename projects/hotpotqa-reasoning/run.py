"""Orchestrator: load data → CoT → ReAct → metrics → charts → tar results."""
import json
import os
import sys
import time
import logging

logging.getLogger("vllm").setLevel(logging.WARNING)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

OUTPUT_DIR = "/content"
DATA_PATH = f"{OUTPUT_DIR}/data.json"
COT_RESULTS_PATH = f"{OUTPUT_DIR}/cot_results.json"
REACT_RESULTS_PATH = f"{OUTPUT_DIR}/react_results.json"
METRICS_PATH = f"{OUTPUT_DIR}/metrics.json"
LOG_PATH = f"{OUTPUT_DIR}/run.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def main() -> None:
    log("=== HotpotQA Reasoning Comparison ===")

    # ── Load data ──
    log("Loading data...")
    with open(DATA_PATH) as f:
        examples = json.load(f)
    log(f"Loaded {len(examples)} examples")
    log(f"  Types: {sum(1 for e in examples if e['type']=='bridge')} bridge, "
        f"{sum(1 for e in examples if e['type']=='comparison')} comparison")

    # ── Init vLLM ──
    log("Initializing vLLM with Qwen2.5-7B-Instruct-AWQ...")
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    log("vLLM ready")

    # ── Run CoT ──
    log("\n=== CoT Strategy ===")
    from strategies.cot import run_cot

    t0 = time.time()
    cot_results = run_cot(llm, examples)
    cot_time = time.time() - t0
    log(f"CoT completed in {cot_time:.1f}s")

    with open(COT_RESULTS_PATH, "w") as f:
        json.dump(cot_results, f, indent=2, ensure_ascii=False)
    log(f"CoT results saved to {COT_RESULTS_PATH}")

    # ── Run ReAct ──
    log("\n=== ReAct Strategy ===")
    from strategies.react import run_react

    t0 = time.time()
    react_results = run_react(llm, examples)
    react_time = time.time() - t0
    log(f"ReAct completed in {react_time:.1f}s")

    with open(REACT_RESULTS_PATH, "w") as f:
        json.dump(react_results, f, indent=2, ensure_ascii=False)
    log(f"ReAct results saved to {REACT_RESULTS_PATH}")

    # ── Metrics ──
    log("\n=== Metrics ===")
    from metrics import compute_all

    metrics = compute_all(cot_results, react_results)
    log(f"CoT  EM: {metrics['cot']['exact_match']:.4f}  F1: {metrics['cot']['f1']:.4f}  "
        f"Lat: {metrics['cot']['avg_latency_s']:.2f}s  Tok: {metrics['cot']['avg_tokens_per_example']:.0f}")
    log(f"ReAct EM: {metrics['react']['exact_match']:.4f}  F1: {metrics['react']['f1']:.4f}  "
        f"Lat: {metrics['react']['avg_latency_s']:.2f}s  Tok: {metrics['react']['avg_tokens_per_example']:.0f}  "
        f"Steps: {metrics['react']['avg_steps']:.1f}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log(f"Metrics saved to {METRICS_PATH}")

    # ── Charts ──
    log("\n=== Charts ===")
    from visualize import generate_all
    generate_all(METRICS_PATH)

    # ── Package results ──
    log("\n=== Packaging ===")
    import subprocess
    tarball = f"{OUTPUT_DIR}/results.tar.gz"
    subprocess.run(
        ["tar", "-czf", tarball, "-C", OUTPUT_DIR,
         "metrics.json", "cot_results.json", "react_results.json", "charts"],
        check=True,
    )
    log(f"Results packaged: {tarball}")
    log(f"Download: colab download {tarball} .")

    # ── Cleanup ──
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    log("\n=== Done ===")


if __name__ == "__main__":
    main()
