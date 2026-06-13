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
