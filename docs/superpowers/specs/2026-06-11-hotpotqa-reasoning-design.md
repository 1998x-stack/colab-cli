# HotpotQA Reasoning Comparison: CoT vs ReAct

Compare Chain-of-Thought and ReAct prompting strategies on HotpotQA using Qwen2.5-7B-Instruct-AWQ served via vLLM on a Colab T4 GPU. Measure accuracy (EM, F1), latency, token efficiency, and visualize the accuracy-cost tradeoff.

## Architecture

Single-process sequential evaluation — CoT runs first, then ReAct. vLLM loads the model once, serves both strategies, then shuts down. No concurrent GPU workloads to avoid VRAM competition on T4's 16GB.

```
projects/hotpotqa-reasoning/
├── load_data.py          # Download HotpotQA, sample 200, save to /content/data.json
├── prompts.py            # CoT and ReAct prompt templates
├── strategies/
│   ├── cot.py            # CoT: single prompt → single vLLM call per batch
│   └── react.py          # ReAct: multi-turn loop (Thought→Action→Observe) with vLLM
├── metrics.py            # EM, F1, latency, token counts
├── visualize.py          # Matplotlib charts → /content/charts/
├── run.py                # Orchestrator: load data → run CoT → run ReAct → compare
├── launch.py             # Colab bootstrap (pip install + spawn run.py detached)
├── check_progress.py     # Heartbeat + log tail
└── README.md
```

**Data flow:**
```
load_data.py → /content/data.json (200 HotpotQA examples)
                      │
              ┌───────┴───────┐
              ▼               ▼
         cot.py          react.py
              │               │
              ▼               ▼
         /content/cot_results.json
         /content/react_results.json
                      │
                      ▼
              metrics.py → /content/metrics.json
              visualize.py → /content/charts/*.png
```

## Approach: Context-Provided ReAct

HotpotQA provides supporting passages (`context`) with each question. Both strategies receive the same passages. ReAct's "Action: Search[...]" looks up facts within the provided text rather than calling external APIs. This isolates reasoning strategy as the only variable — no retrieval quality confound.

## Prompt Design

### CoT

```
You are a precise reasoning assistant. Answer the question using the provided context.

Context:
{passages}

Question: {question}

Let's think step by step:

Step 1: ...
Step 2: ...
...

Final Answer: [short answer]
```

### ReAct

```
You are a precise reasoning assistant. Answer the question using the provided context.
Use the following format:

Thought: <reason about what information you need>
Action: Search[<entity or fact to look up in the context>]
Observation: <what the context says about this>
... (repeat as needed)
Thought: I have enough information to answer.
Final Answer: [short answer]

Context:
{passages}

Question: {question}
```

| | CoT | ReAct |
|---|---|---|
| Structure | Linear chain of reasoning | Interleaved reasoning + explicit lookups |
| LLM calls per example | 1 | 2-4 (one per step) |
| Parsing | Extract after "Final Answer:" | Feed observation back as prompt prefix for next turn |
| Max tokens/step | 512 | 256 |
| Stop condition | — | "Final Answer:" or 5 steps max |

Both strategies use greedy decoding (temperature=0) for reproducibility.

## Model & Inference

- **Model:** `Qwen/Qwen2.5-7B-Instruct-AWQ` (AWQ 4-bit, ~4.5GB VRAM)
- **Inference engine:** vLLM offline API (`vllm.LLM` + `SamplingParams`)
- **GPU:** Colab T4 (16GB VRAM)
- **Max model length:** 4096 tokens (enough for context passages + generation)
- **GPU memory utilization:** 0.85

vLLM loads the model once at startup. CoT runs first (batched, 1 call), then ReAct (multi-turn per example, batched across independent examples at each turn). Model is unloaded after both complete.

## Metrics

```json
{
  "config": {
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "dataset": "hotpot_qa",
    "n_examples": 200
  },
  "cot": {
    "exact_match": 0.42,
    "f1": 0.58,
    "avg_latency_s": 3.2,
    "total_tokens": 48000,
    "avg_tokens_per_example": 240
  },
  "react": {
    "exact_match": 0.51,
    "f1": 0.64,
    "avg_latency_s": 8.7,
    "total_tokens": 95000,
    "avg_tokens_per_example": 475,
    "avg_steps": 2.8,
    "step_distribution": {"1": 12, "2": 68, "3": 74, "4": 35, "5": 11}
  },
  "per_example": [...]
}
```

**Primary metrics:** Exact Match (EM), F1 (token-level overlap).

**Secondary metrics:** Per-example latency, token count, ReAct step count.

Per-example records enable slicing by difficulty (1-hop vs 2-hop per HotpotQA `type` field) and finding cases where strategies diverge.

## Visualizations

All charts saved as PNGs to `/content/charts/`:

1. **`accuracy_comparison.png`** — grouped bar chart: EM and F1 for CoT vs ReAct
2. **`latency_comparison.png`** — box plots: per-example latency distribution for each strategy
3. **`token_efficiency.png`** — scatter plot: tokens (x) vs EM accuracy (y), one point per strategy
4. **`react_steps.png`** — histogram of ReAct step counts, split by correct/incorrect

## Colab Deployment

```bash
colab new --gpu T4 -s hotpotqa
colab upload load_data.py prompts.py run.py launch.py /content/
colab upload strategies/ /content/strategies/
colab upload metrics.py visualize.py /content/
colab exec -f launch.py --timeout 120
# Monitor with check_progress.py
# Download: colab download /content/results.tar.gz .
colab stop -s hotpotqa
```

Same detached bootstrap pattern as vllm-compare: `launch.py` spawns `run.py` via `subprocess.Popen(start_new_session=True)` after pip-installing vLLM and dependencies.

## Constraints

- **Session lifetime:** Free-tier T4 sessions last ~2-4 hours. 200 examples × (CoT ~4s + ReAct ~12s) ≈ ~53 minutes. Fits comfortably.
- **VRAM:** Qwen2.5-7B-AWQ uses ~4.5GB. T4 has 16GB. Ample headroom.
- **vLLM install:** Use `--extra-index-url https://download.pytorch.org/whl/cu128` for CUDA 12.8 compatibility on Colab.
- **HotpotQA download:** Use `datasets` library. The `hotpot_qa` dataset is available in `distractor` and `fullwiki` configs. Use `distractor` (provides context passages, the standard setting).

## Success Criteria

1. Both strategies run to completion on 200 HotpotQA examples within one T4 session
2. EM and F1 computed and reported for both strategies
3. Latency and token efficiency captured per strategy
4. Four charts generated and saved as PNGs
5. All results downloadable as a single tarball from Colab
