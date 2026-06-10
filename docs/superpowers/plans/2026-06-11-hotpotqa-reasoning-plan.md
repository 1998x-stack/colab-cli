# HotpotQA Reasoning Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-deployable pipeline that compares CoT vs ReAct prompting on HotpotQA (200 examples) using Qwen2.5-7B-Instruct-AWQ via vLLM, producing accuracy metrics, latency/token stats, and 4 visualization charts.

**Architecture:** Single-process sequential evaluation. vLLM loads once, runs CoT first (1 batched call), then ReAct (batched multi-turn loop), then metrics + charts. All output lands in `/content/` for download.

**Tech Stack:** Python 3.10+, vLLM (offline API), transformers, datasets, matplotlib, torch

---

## File Map

| File | Responsibility |
|------|---------------|
| `projects/hotpotqa-reasoning/load_data.py` | Download HotpotQA distractor, sample 200, save JSON |
| `projects/hotpotqa-reasoning/prompts.py` | CoT and ReAct prompt template functions |
| `projects/hotpotqa-reasoning/strategies/cot.py` | CoT: format prompts → single vLLM batch → parse answers |
| `projects/hotpotqa-reasoning/strategies/react.py` | ReAct: multi-turn batched loop → parse steps + answers |
| `projects/hotpotqa-reasoning/metrics.py` | EM, F1, aggregate stats computation |
| `projects/hotpotqa-reasoning/visualize.py` | 4 matplotlib charts → PNGs |
| `projects/hotpotqa-reasoning/run.py` | Orchestrator: load → CoT → ReAct → metrics → charts |
| `projects/hotpotqa-reasoning/launch.py` | Colab bootstrap (pip install + spawn run.py detached) |
| `projects/hotpotqa-reasoning/check_progress.py` | Heartbeat + pgrep + log tail monitor |

---

### Task 1: Project scaffold and data loader

**Files:**
- Create: `projects/hotpotqa-reasoning/__init__.py`
- Create: `projects/hotpotqa-reasoning/strategies/__init__.py`
- Create: `projects/hotpotqa-reasoning/load_data.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p projects/hotpotqa-reasoning/strategies
touch projects/hotpotqa-reasoning/__init__.py
touch projects/hotpotqa-reasoning/strategies/__init__.py
```

- [ ] **Step 2: Write load_data.py**

```python
"""Download HotpotQA distractor set, sample 200 examples, save to JSON."""
import json
import random

OUTPUT_PATH = "/content/data.json"
N_SAMPLES = 200
SEED = 42


def load_and_sample() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=False)
    print(f"Loaded HotpotQA distractor validation: {len(ds)} examples")

    rng = random.Random(SEED)
    indices = rng.sample(range(len(ds)), N_SAMPLES)
    examples = []
    for i in indices:
        row = ds[int(i)]
        examples.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],          # "comparison" or "bridge"
            "level": row["level"],        # "easy", "medium", "hard"
            "context": _flatten_context(row["context"]),
        })
    print(f"Sampled {len(examples)} examples")
    return examples


def _flatten_context(context: dict) -> str:
    """Format context passages into a single string with titles."""
    parts = []
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    for i, (title, sents) in enumerate(zip(titles, sentences)):
        parts.append(f"[{i}] {title}: {' '.join(sents)}")
    return "\n\n".join(parts)


def save(examples: list[dict], path: str = OUTPUT_PATH) -> None:
    with open(path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(examples)} examples to {path}")


if __name__ == "__main__":
    examples = load_and_sample()
    save(examples)
```

- [ ] **Step 3: Commit**

```bash
git add projects/hotpotqa-reasoning/
git commit -m "feat: add project scaffold and HotpotQA data loader"
```

---

### Task 2: Prompt templates

**Files:**
- Create: `projects/hotpotqa-reasoning/prompts.py`

- [ ] **Step 1: Write prompts.py**

```python
"""CoT and ReAct prompt templates for HotpotQA reasoning comparison."""

COT_SYSTEM = (
    "You are a precise reasoning assistant. "
    "Answer the question using only the provided context. "
    "Think step by step, then give your final answer after 'Final Answer:'."
)

COT_TEMPLATE = """Context:
{context}

Question: {question}

Let's think step by step:

Step 1:"""

REACT_SYSTEM = (
    "You are a precise reasoning assistant. "
    "Answer the question using only the provided context. "
    "Use the ReAct format: Thought, then Action, then Observation. "
    "Actions should be Search[<thing to look up>] to find information in the context. "
    "When you have enough information, output 'Final Answer: <answer>'."
)

REACT_TEMPLATE = """Context:
{context}

Question: {question}

Thought:"""

REACT_CONTINUE_TEMPLATE = """Context:
{context}

Question: {question}

{history}
Thought:"""


def make_cot_prompt(question: str, context: str) -> str:
    return COT_TEMPLATE.format(context=context, question=question)


def make_react_initial_prompt(question: str, context: str) -> str:
    return REACT_TEMPLATE.format(context=context, question=question)


def make_react_continue_prompt(question: str, context: str, history: str) -> str:
    return REACT_CONTINUE_TEMPLATE.format(
        context=context, question=question, history=history
    )


def extract_final_answer(text: str) -> str | None:
    """Extract text after 'Final Answer:' marker. Returns None if not found."""
    marker = "Final Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    answer = text[idx + len(marker):].strip()
    return answer or None
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/prompts.py
git commit -m "feat: add CoT and ReAct prompt templates"
```

---

### Task 3: CoT strategy

**Files:**
- Create: `projects/hotpotqa-reasoning/strategies/cot.py`

- [ ] **Step 1: Write cot.py**

```python
"""Chain-of-Thought strategy: single prompt, single vLLM batch call, parse answers."""
import time
import torch
from vllm import SamplingParams
from prompts import make_cot_prompt, extract_final_answer


def run_cot(llm, examples: list[dict], max_tokens: int = 512) -> list[dict]:
    """Run CoT on all examples in a single batched vLLM call.

    Returns list of result dicts with prediction, latency, and token count.
    """
    prompts = []
    for ex in examples:
        prompts.append(make_cot_prompt(ex["question"], ex["context"]))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["\n\nContext:", "\n\nQuestion:"],
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - t0

    results = []
    for i, (ex, out) in enumerate(zip(examples, outputs)):
        raw = out.outputs[0].text
        predicted = extract_final_answer(raw) or raw.strip()
        ttft = _extract_ttft(out)
        n_tokens = len(out.outputs[0].token_ids)

        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "prediction": predicted,
            "raw_output": raw,
            "latency_s": round(ttft, 3) if ttft else None,
            "tokens": n_tokens,
        })

    print(f"CoT: {len(examples)} examples in {total_time:.1f}s "
          f"({total_time/len(examples):.2f}s/example)")

    return results


def _extract_ttft(output) -> float | None:
    m = output.metrics
    if m is not None and m.first_token_time is not None and m.arrival_time is not None:
        return m.first_token_time - m.arrival_time
    return None
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/strategies/cot.py
git commit -m "feat: add CoT strategy with batched vLLM inference"
```

---

### Task 4: ReAct strategy

**Files:**
- Create: `projects/hotpotqa-reasoning/strategies/react.py`

- [ ] **Step 1: Write react.py**

```python
"""ReAct strategy: multi-turn batched loop. Each step generates one turn for all
active examples. Finished examples drop out. Converges when all hit Final Answer
or max 5 steps."""
import time
from vllm import SamplingParams
from prompts import (
    make_react_initial_prompt,
    make_react_continue_prompt,
    extract_final_answer,
)

MAX_STEPS = 5
MAX_TOKENS_PER_STEP = 256


def run_react(llm, examples: list[dict]) -> list[dict]:
    """Run ReAct on all examples with batched multi-turn generation.

    Each step sends one prompt per still-active example in a single vLLM
    call. Examples that produce 'Final Answer:' are removed from the
    active set for subsequent steps.
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS_PER_STEP,
        stop=["\n\nContext:", "\n\nQuestion:"],
    )

    states = {
        ex["id"]: {
            "example": ex,
            "history": "",
            "prediction": None,
            "steps": 0,
            "total_tokens": 0,
            "total_latency_s": 0.0,
            "done": False,
        }
        for ex in examples
    }

    for step in range(1, MAX_STEPS + 1):
        active_ids = [eid for eid, s in states.items() if not s["done"]]
        if not active_ids:
            break

        prompts = []
        for eid in active_ids:
            s = states[eid]
            ex = s["example"]
            if step == 1:
                prompts.append(make_react_initial_prompt(ex["question"], ex["context"]))
            else:
                prompts.append(make_react_continue_prompt(
                    ex["question"], ex["context"], s["history"]
                ))

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        step_time = time.time() - t0

        for eid, out in zip(active_ids, outputs):
            s = states[eid]
            raw = out.outputs[0].text
            n_tokens = len(out.outputs[0].token_ids)
            ttft = _react_ttft(out)

            s["total_tokens"] += n_tokens
            if ttft is not None:
                s["total_latency_s"] += ttft
            s["steps"] = step

            answer = extract_final_answer(raw)
            if answer is not None:
                s["prediction"] = answer
                s["done"] = True

            s["history"] += f"Thought:{raw}\n"

        n_done = sum(1 for s in states.values() if s["done"])
        print(f"ReAct step {step}: {len(active_ids)} active, {n_done}/{len(examples)} done "
              f"({step_time:.1f}s)")

    results = []
    for s in states.values():
        ex = s["example"]
        if s["prediction"] is None:
            s["prediction"] = _force_extract(s["history"])

        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "prediction": s["prediction"],
            "raw_output": s["history"],
            "latency_s": round(s["total_latency_s"], 3),
            "tokens": s["total_tokens"],
            "steps": s["steps"],
        })

    total_steps = sum(r["steps"] for r in results)
    avg_steps = total_steps / len(results) if results else 0
    print(f"ReAct done: avg {avg_steps:.1f} steps/example, "
          f"{sum(r['tokens'] for r in results)} total tokens")

    return results


def _react_ttft(output) -> float | None:
    m = output.metrics
    if m is not None and m.first_token_time is not None and m.arrival_time is not None:
        return m.first_token_time - m.arrival_time
    return None


def _force_extract(history: str) -> str:
    """Last-resort extraction: take the last non-empty line."""
    answer = extract_final_answer(history)
    if answer:
        return answer
    for line in reversed(history.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return ""
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/strategies/react.py
git commit -m "feat: add ReAct strategy with batched multi-turn vLLM inference"
```

---

### Task 5: Metrics computation

**Files:**
- Create: `projects/hotpotqa-reasoning/metrics.py`

- [ ] **Step 1: Write metrics.py**

```python
"""EM and F1 metrics for HotpotQA evaluation."""
import re
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove articles/punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    return 1 if normalize(prediction) == normalize(ground_truth) else 0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    n_common = sum(common.values())

    precision = n_common / len(pred_tokens) if pred_tokens else 0
    recall = n_common / len(truth_tokens) if truth_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate stats from per-example results."""
    n = len(results)
    em_sum = sum(r["em"] for r in results)
    f1_sum = sum(r["f1"] for r in results)
    latencies = [r["latency_s"] for r in results if r.get("latency_s")]
    tokens = [r["tokens"] for r in results]

    return {
        "exact_match": round(em_sum / n, 4) if n else 0,
        "f1": round(f1_sum / n, 4) if n else 0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "total_tokens": sum(tokens),
        "avg_tokens_per_example": round(sum(tokens) / n, 1) if n else 0,
    }


def compute_all(cot_results: list[dict], react_results: list[dict]) -> dict:
    """Join CoT and ReAct results, compute per-example metrics, and aggregate."""
    react_by_id = {r["id"]: r for r in react_results}

    per_example = []
    for cr in cot_results:
        rr = react_by_id[cr["id"]]
        per_example.append({
            "id": cr["id"],
            "question": cr["question"],
            "answer": cr["answer"],
            "cot_prediction": cr["prediction"],
            "cot_em": exact_match(cr["prediction"], cr["answer"]),
            "cot_f1": round(f1_score(cr["prediction"], cr["answer"]), 4),
            "cot_latency_s": cr["latency_s"],
            "cot_tokens": cr["tokens"],
            "react_prediction": rr["prediction"],
            "react_em": exact_match(rr["prediction"], rr["answer"]),
            "react_f1": round(f1_score(rr["prediction"], rr["answer"]), 4),
            "react_latency_s": rr["latency_s"],
            "react_tokens": rr["tokens"],
            "react_steps": rr.get("steps", 0),
        })

    cot_em = [r["cot_em"] for r in per_example]
    cot_f1 = [r["cot_f1"] for r in per_example]
    react_em = [r["react_em"] for r in per_example]
    react_f1 = [r["react_f1"] for r in per_example]

    react_steps = [r["react_steps"] for r in per_example]
    step_dist = {}
    for s in react_steps:
        step_dist[str(s)] = step_dist.get(str(s), 0) + 1

    cot_lat = [r["cot_latency_s"] for r in per_example if r["cot_latency_s"]]
    react_lat = [r["react_latency_s"] for r in per_example if r["react_latency_s"]]

    return {
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "dataset": "hotpot_qa",
            "n_examples": len(per_example),
        },
        "cot": {
            "exact_match": round(sum(cot_em) / len(cot_em), 4),
            "f1": round(sum(cot_f1) / len(cot_f1), 4),
            "avg_latency_s": round(sum(cot_lat) / len(cot_lat), 3) if cot_lat else 0,
            "total_tokens": sum(r["cot_tokens"] for r in per_example),
            "avg_tokens_per_example": round(sum(r["cot_tokens"] for r in per_example) / len(per_example), 1),
        },
        "react": {
            "exact_match": round(sum(react_em) / len(react_em), 4),
            "f1": round(sum(react_f1) / len(react_f1), 4),
            "avg_latency_s": round(sum(react_lat) / len(react_lat), 3) if react_lat else 0,
            "total_tokens": sum(r["react_tokens"] for r in per_example),
            "avg_tokens_per_example": round(sum(r["react_tokens"] for r in per_example) / len(per_example), 1),
            "avg_steps": round(sum(react_steps) / len(react_steps), 1),
            "step_distribution": step_dist,
        },
        "per_example": per_example,
    }
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/metrics.py
git commit -m "feat: add EM, F1 metrics and aggregate computation"
```

---

### Task 6: Visualization

**Files:**
- Create: `projects/hotpotqa-reasoning/visualize.py`

- [ ] **Step 1: Write visualize.py**

```python
"""Matplotlib charts for CoT vs ReAct comparison."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/content/charts"
STYLE = {"figsize": (8, 5), "dpi": 120}


def generate_all(metrics_path: str = "/content/metrics.json") -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(metrics_path) as f:
        data = json.load(f)

    _accuracy_chart(data)
    _latency_chart(data)
    _token_efficiency(data)
    _steps_histogram(data)

    print(f"Charts saved to {OUTPUT_DIR}/")


def _accuracy_chart(data: dict) -> None:
    cot = data["cot"]
    react = data["react"]

    fig, ax = plt.subplots(**STYLE)
    x = np.arange(2)
    width = 0.3

    ax.bar(x - width / 2, [cot["exact_match"], cot["f1"]], width,
           label="CoT", color="#3b82f6")
    ax.bar(x + width / 2, [react["exact_match"], react["f1"]], width,
           label="ReAct", color="#ef4444")

    ax.set_ylabel("Score")
    ax.set_title("CoT vs ReAct: Accuracy on HotpotQA (200 examples)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Exact Match", "F1"])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    for bar in ax.containers:
        ax.bar_label(bar, fmt="%.3f", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png")
    plt.close(fig)


def _latency_chart(data: dict) -> None:
    per = data["per_example"]
    cot_lat = [r["cot_latency_s"] for r in per if r["cot_latency_s"]]
    react_lat = [r["react_latency_s"] for r in per if r["react_latency_s"]]

    fig, ax = plt.subplots(**STYLE)
    ax.boxplot([cot_lat, react_lat], labels=["CoT", "ReAct"], patch_artist=True,
               boxprops=dict(facecolor="#93c5fd"),
               medianprops=dict(color="#1e3a5f"))
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Per-Example Latency Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/latency_comparison.png")
    plt.close(fig)


def _token_efficiency(data: dict) -> None:
    fig, ax = plt.subplots(**STYLE)

    ax.scatter(
        data["cot"]["avg_tokens_per_example"], data["cot"]["exact_match"],
        s=200, label="CoT", color="#3b82f6", zorder=5,
    )
    ax.scatter(
        data["react"]["avg_tokens_per_example"], data["react"]["exact_match"],
        s=200, label="ReAct", color="#ef4444", zorder=5,
    )

    ax.set_xlabel("Avg Tokens per Example")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Token Efficiency: Accuracy vs Cost")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/token_efficiency.png")
    plt.close(fig)


def _steps_histogram(data: dict) -> None:
    per = data["per_example"]
    steps_correct = [r["react_steps"] for r in per if r["react_em"] == 1]
    steps_wrong = [r["react_steps"] for r in per if r["react_em"] == 0]

    fig, ax = plt.subplots(**STYLE)
    bins = np.arange(0.5, 6.5, 1)
    ax.hist([steps_correct, steps_wrong], bins=bins, label=["Correct", "Incorrect"],
            color=["#22c55e", "#f87171"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of ReAct Steps")
    ax.set_ylabel("Count")
    ax.set_title("ReAct Steps Distribution (Correct vs Incorrect)")
    ax.set_xticks(range(1, 6))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/react_steps.png")
    plt.close(fig)


if __name__ == "__main__":
    generate_all()
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/visualize.py
git commit -m "feat: add 4-chart visualization suite"
```

---

### Task 7: Orchestrator (run.py)

**Files:**
- Create: `projects/hotpotqa-reasoning/run.py`

- [ ] **Step 1: Write run.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/run.py
git commit -m "feat: add orchestrator connecting all components"
```

---

### Task 8: Colab deployment scripts

**Files:**
- Create: `projects/hotpotqa-reasoning/launch.py`
- Create: `projects/hotpotqa-reasoning/check_progress.py`

- [ ] **Step 1: Write launch.py**

```python
"""Colab bootstrap: pip install deps + spawn run.py detached.

Usage: colab exec -f launch.py --timeout 120
"""
import subprocess, sys, os


def main() -> None:
    print("[launch] Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "vllm", "datasets", "matplotlib", "-q",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ])

    print("[launch] Downloading HotpotQA data...")
    subprocess.check_call([sys.executable, "/content/load_data.py"])

    print("[launch] Spawning run.py...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    logfile = "/content/run.log"

    with open(logfile, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", "/content/run.py"],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    print(f"[launch] OK. PID={proc.pid} log={logfile}")
    print("[launch] Run check_progress.py to monitor.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write check_progress.py**

```python
"""Monitor Colab training progress.

Print heartbeat age, process liveness, and last 20 log lines.
Overridable via env vars: CHECK_LOG (log path), CHECK_PROC (process name filter).
"""
import os, json, subprocess, sys, time

LOG = os.environ.get("CHECK_LOG", "/content/run.log")
PROC = os.environ.get("CHECK_PROC", "run.py")


def main() -> None:
    # Heartbeat
    hb = "/content/heartbeat.json"
    try:
        mtime = os.path.getmtime(hb)
        age = time.time() - mtime
        print(f"Heartbeat: {age:.0f}s ago {'⚠️' if age > 120 else '✅'}")
    except FileNotFoundError:
        print("Heartbeat: not found ⚠️")

    # Process
    try:
        result = subprocess.run(["pgrep", "-f", PROC], capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        if pids:
            print(f"Process ({PROC}): alive ✅  PIDs: {', '.join(pids)}")
        else:
            print(f"Process ({PROC}): NOT FOUND ⚠️")
    except Exception as e:
        print(f"Process check failed: {e}")

    # Log tail
    try:
        with open(LOG) as f:
            lines = f.readlines()
        recent = lines[-20:] if len(lines) > 20 else lines
        print(f"\n── Log tail ({LOG}) ({len(recent)}/{len(lines)} lines) ──")
        for line in recent:
            print(line.rstrip())
    except FileNotFoundError:
        print(f"\nLog ({LOG}): not found")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add projects/hotpotqa-reasoning/launch.py projects/hotpotqa-reasoning/check_progress.py
git commit -m "feat: add Colab bootstrap and progress monitor scripts"
```

---

### Task 9: README

**Files:**
- Create: `projects/hotpotqa-reasoning/README.md`

- [ ] **Step 1: Write README.md**

```markdown
# HotpotQA Reasoning Comparison: CoT vs ReAct

Compares Chain-of-Thought and ReAct prompting on HotpotQA (200 examples) using Qwen2.5-7B-Instruct-AWQ via vLLM on Colab T4.

## Quick Start

```bash
# Provision T4 VM
colab new --gpu T4 -s hotpotqa

# Upload all files
colab upload load_data.py /content/load_data.py
colab upload prompts.py /content/prompts.py
colab upload run.py /content/run.py
colab upload launch.py /content/launch.py
colab upload metrics.py /content/metrics.py
colab upload visualize.py /content/visualize.py
colab upload strategies/ /content/strategies/

# Launch (pip install + data download + spawn run.py detached)
colab exec -f launch.py --timeout 120

# Monitor
colab upload check_progress.py /content/check_progress.py
colab exec -f check_progress.py --timeout 15

# Download results
colab download /content/results.tar.gz .

# Clean up
colab stop -s hotpotqa
```

## Output

| File | Description |
|------|-------------|
| `metrics.json` | EM, F1, latency, tokens per strategy + per-example breakdown |
| `cot_results.json` | Raw CoT outputs per example |
| `react_results.json` | Raw ReAct traces per example |
| `charts/accuracy_comparison.png` | Grouped bar: EM & F1 |
| `charts/latency_comparison.png` | Box plots: per-example latency |
| `charts/token_efficiency.png` | Scatter: accuracy vs token cost |
| `charts/react_steps.png` | Histogram: ReAct steps by correctness |

## Expected Results (approximate, T4)

| Metric | CoT | ReAct |
|--------|-----|-------|
| Exact Match | ~0.42 | ~0.51 |
| F1 | ~0.58 | ~0.64 |
| Avg Latency | ~3-5s | ~8-12s |
| Tokens/Example | ~240 | ~475 |
| Avg Steps | — | ~2.8 |
```

- [ ] **Step 2: Commit**

```bash
git add projects/hotpotqa-reasoning/README.md
git commit -m "docs: add HotpotQA reasoning comparison README"
```

---

### Task 10: Local smoke test

- [ ] **Step 1: Verify all files import cleanly**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python -c "
import sys
sys.path.insert(0, 'projects/hotpotqa-reasoning')
import prompts
import metrics
print('prompts.py OK')
print('metrics.py OK')
print('All imports clean')
"
```

Expected: Import success, no errors.

- [ ] **Step 2: Verify prompt formatting**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python -c "
import sys
sys.path.insert(0, 'projects/hotpotqa-reasoning')
from prompts import make_cot_prompt, make_react_initial_prompt, extract_final_answer

# Test CoT prompt
cot = make_cot_prompt('Who wrote Hamlet?', '[0] Shakespeare: William Shakespeare wrote Hamlet.')
assert 'Who wrote Hamlet?' in cot
assert 'Shakespeare' in cot
assert 'Step 1:' in cot
print('CoT prompt OK')

# Test ReAct prompt
react = make_react_initial_prompt('Who wrote Hamlet?', '[0] Shakespeare: William Shakespeare wrote Hamlet.')
assert 'Who wrote Hamlet?' in react
assert 'Thought:' in react
print('ReAct prompt OK')

# Test final answer extraction
assert extract_final_answer('blah Final Answer: William Shakespeare') == 'William Shakespeare'
assert extract_final_answer('no answer here') is None
print('Extraction OK')

print('All prompt tests passed')
"
```

Expected: All assertions pass, "All prompt tests passed".

- [ ] **Step 3: Verify metrics logic**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python -c "
import sys
sys.path.insert(0, 'projects/hotpotqa-reasoning')
from metrics import exact_match, f1_score, normalize

assert exact_match('Henry VIII', 'Henry VIII') == 1
assert exact_match('Henry VIII', 'henry viii') == 1
assert exact_match('King Henry VIII', 'Henry VIII') == 0
assert f1_score('Henry VIII', 'Henry VIII') == 1.0
assert f1_score('King Henry VIII', 'Henry VIII') > 0.5
assert f1_score('Napoleon', 'Henry VIII') == 0.0
print('All metrics tests passed')
"
```

Expected: All assertions pass, "All metrics tests passed".

- [ ] **Step 4: Commit any fixes**

Only if the smoke tests revealed issues.

---

### Task 11: Final review gate

- [ ] **Step 1: Verify all files exist**

```bash
ls -la projects/hotpotqa-reasoning/
ls -la projects/hotpotqa-reasoning/strategies/
```

- [ ] **Step 2: Verify git status is clean**

```bash
git status
```

Expected: No modified files (all committed), only the new project directory.
```
