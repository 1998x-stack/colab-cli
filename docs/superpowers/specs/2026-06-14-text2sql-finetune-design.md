# Text2SQL Fine-tuning on Colab T4 — Design Spec

**Date:** 2026-06-14
**Status:** Approved

## Overview

Fine-tune Qwen3-0.6B on `b-mc2/sql-create-context` for Text2SQL generation using LoRA SFT on a Colab T4 GPU. Validate via execution-match against in-memory SQLite databases. Deploy with cron watchtower monitoring.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training | SFT (cross-entropy on answer tokens) | Fits T4 VRAM, fast iteration |
| Model | `Qwen/Qwen3-0.6B` (bfloat16) | 1.41 GB FP16, ~2.5 GB with LoRA — leaves 13 GB free on T4 |
| Dataset | `b-mc2/sql-create-context` (78K examples) | WikiSQL + Spider merged, includes CREATE TABLE context |
| Eval | Execution-match (SQLite in-memory, 5s timeout) | Ground truth feedback without needing a real database |
| Subset | 500 train / 100 test (1 epoch) | Fits Colab ~8 min GPU window |
| LoRA | rank=8, alpha=16, target q_proj,v_proj,k_proj,o_proj | Standard config for causal LM, small adapter size |
| Deploy | Colab T4, single `colab exec --timeout 540` | Within 10-min keepalive window |

## Architecture

```
dataset.py          train.py            evaluate.py
load from HF ──▶    LoRA SFT      ──▶   generate SQL
format prompt       save adapter         execute both (GT + gen)
subset              log loss             compare result sets
                                        output eval_report.json

launch.py           fetch.sh
Colab bootstrap     cron watchtower (tar → download → tail)
```

Each module independently runnable. `evaluate.py` can be re-run without re-training.

## Component specs

### dataset.py

- Loads `b-mc2/sql-create-context` via `datasets.load_dataset`
- Formats each example into Qwen3 chat template:
  - System: "You are a SQL expert..."
  - User: "Schema:\n<CREATE TABLE>\n\nQuestion: <question>"
  - Assistant: `<SQL query>`
- Loss mask: compute on assistant tokens only
- CLI: `--max_examples N`, `--split train|test`, `--output path.pt`
- Output: PyTorch tensor dict `{input_ids, attention_mask, labels}` saved as `.pt`

### train.py

- Loads `Qwen/Qwen3-0.6B` in bfloat16
- Applies LoRA via peft: r=8, alpha=16, target_modules=["q_proj","v_proj","k_proj","o_proj"]
- Training loop: batch_size=4, grad_accum=2 (effective batch 8), max_seq_len=1024
- Optimizer: AdamW, lr=2e-4, cosine schedule, 1 epoch
- Logs every 10 steps to `logs/train.log` and `metrics.csv`
- Saves LoRA adapter to `lora_weights/` via `model.save_pretrained()`
- CLI: `--data_path path.pt --output_dir lora_weights/ --max_steps N`

### evaluate.py

- Loads base model + LoRA adapter
- For each test example:
  1. Format prompt with schema + question
  2. Generate SQL via `model.generate()` with greedy decoding
  3. Extract SQL from between `<|im_start|>assistant` and `<|im_end|>` tags
  4. Parse CREATE TABLE → build `sqlite3 :memory:` database
  5. Execute ground truth SQL → result A
  6. Execute generated SQL in ThreadPoolExecutor with 5s timeout → result B
  7. Compare result sets for execution match
- Outputs `eval_report.json`:
  ```json
  {
    "execution_accuracy": 0.72,
    "exact_match_accuracy": 0.65,
    "total": 100,
    "errors": {"syntax": 8, "timeout": 2, "wrong_result": 18},
    "per_example": [{"question": "...", "generated_sql": "...", "ground_truth": "...", "exec_match": true, "error": null}]
  }
  ```
- Also writes `logs/eval.log` with per-example details
- CLI: `--data_path path.pt --lora_path lora_weights/ --output eval_report.json`

### launch.py

- Single Colab entry point
- Sequence: `pip install` dependencies → download model weights → run dataset.py → run train.py → run evaluate.py → tar outputs
- Writes all outputs to `/content/text2sql-finetune-output/`
- Handles proxy setup (Config B as default: HTTP CONNECT)
- Uses `colab exec -f launch.py` pattern

### fetch.sh

- Cron watchtower payload, fires every 2 minutes
- Steps:
  1. Check session alive: `colab sessions | grep <session>`
  2. Tar on VM: `colab exec -s <name> -f tar_outputs.py`
  3. Download: `colab download -s <name> /content/text2sql-finetune-output.tar.gz ./output.tar.gz`
  4. Extract: `tar -xzf output.tar.gz -C /tmp/text2sql-output/`
  5. Report: `tail logs/train.log`, `tail metrics.csv`
  6. If `eval_report.json` exists → print accuracy + stop cron (training complete)

## Logging & metrics

| Artifact | Format | Frequency |
|----------|--------|-----------|
| `logs/train.log` | `[HH:MM:SS] step 40/500 \| loss=1.23 \| lr=1.8e-4 \| elapsed=42s` | Every 10 steps |
| `metrics.csv` | `step,loss,lr,elapsed_s` | Every 10 steps, append |
| `logs/eval.log` | Per-example: question, generated SQL, ground truth, pass/fail, error type | Once after training |
| `eval_report.json` | Summary + per-example breakdown | Once after training |

## Testing

```
tests/
  test_dataset.py     — format produces valid chat template, truncation works
  test_evaluate.py    — execution-match on 3 hand-crafted SQL pairs (correct, wrong syntax, timeout)
  test_train.py       — forward pass on 1 batch doesn't NaN, loss decreases on 2nd step
```

Dataset and eval tests run locally without GPU. Train test needs a tiny model stub or runs on Colab.

## Session time budget

| Phase | Est. time |
|-------|----------|
| pip install + model download | ~2 min |
| dataset load + format | ~30s |
| Training (500 ex, 1 epoch, ~2-3 ex/s) | ~3-4 min |
| Eval (100 ex, generation + execution) | ~2 min |
| **Total** | **~7-8 min** |

Fits within single `colab exec --timeout 540` window.

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| First-session overhead (CUDA JIT, data download) eats ~7 min | Warmup session first to cache model + dataset |
| SQL execution hangs (cartesian product, infinite loop) | 5s ThreadPoolExecutor timeout per query |
| Colab WebSocket drops mid-training | fetch.sh cron continues downloading outputs via REST |
| Model generates non-SQL text | Extract between assistant tags; syntax error → count as wrong |
| HF dataset download slow on Colab | Pre-download locally, upload as .pt file to VM |
