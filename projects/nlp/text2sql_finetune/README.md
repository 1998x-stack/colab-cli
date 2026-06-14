# text2sql_finetune

LoRA fine-tune Qwen3-0.6B on `b-mc2/sql-create-context` for text-to-SQL generation. Deployed on Colab T4 (free tier).

## Quick start

```bash
cd /Users/mx/Desktop/projects/colab-cli/projects/nlp/text2sql_finetune
bash deploy.sh text2sql          # default account
bash deploy.sh text2sql cb       # account-b
```

This provisions a GPU session, uploads all files, spawns detached training, and launches redundant eval+watchdog processes to keep the session alive.

## Architecture

```
Local                              Colab T4 VM
─────                              ──────────
deploy.sh ──upload──→              /content/text2sql_finetune/
  ├── dataset.py                        ├── bg_launch.py (spawns training detached)
  ├── train.py                          ├── train.py (62 steps, ~84s)
  ├── evaluate.py                       ├── eval_and_watch.py (eval + WS keepalive)
  ├── bg_launch.py                      ├── watchdog.py (relay chain support)
  ├── eval_and_watch.py                 └── lora_weights/ (~2 MB)
  ├── watchdog.py
  └── tar_outputs.py               /content/text2sql-finetune-output/
                                        ├── logs/train.log
                                        ├── logs/eval.log
                                        ├── logs/watchdog.log
                                        ├── eval_report.json
                                        ├── metrics.csv
                                        └── train.pid

cron watchtower ──REST──→          fetch.sh fetches output tar every 2 min
```

## Model

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-0.6B (621M params, bfloat16, ~1.4 GB) |
| Fine-tuning | LoRA rank=8, alpha=16, dropout=0.05 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Trainable params | ~2M |
| VRAM usage | ~5.9 GB (T4 has 15.6 GB) |

## Dataset

`b-mc2/sql-create-context` — 78K examples from WikiSQL + Spider.

- Context: `CREATE TABLE name (cols)` — DDL statement(s)
- Question: natural language query
- Answer: SQL SELECT statement

**Gotcha:** Only `train` split exists (no `test`/`validation`). Use `dataset.py --split auto` to auto-split.

## Training

| Parameter | Value |
|-----------|-------|
| Examples | 500 train / 100 test |
| Batch size | 4 (effective 8 with grad_accum=2) |
| Optimizer | AdamW (lr=2e-4), cosine annealing |
| Max seq length | 1024 tokens |
| Steps | 62 (1 epoch) |
| Training time | ~84s on T4 |

## Evaluation

Execution-match accuracy: generated SQL is run against in-memory SQLite databases built from the parsed CREATE TABLE statements. Result sets are compared to ground truth.

| Metric | Value |
|--------|-------|
| Examples | 100 |
| Eval time | ~204s |
| Metrics | exec_acc (primary), exact_match (secondary) |

## Session survival strategy

Free-tier Colab GPU sessions die 2-5 min after the last WebSocket closes (keep-alive daemon is permanently broken — IAM deadlock at T+61s). The fix:

1. **`bg_launch.py`** — spawns training as detached subprocess (`start_new_session=True`). Returns in <1 min. Training survives all WebSocket drops.

2. **`eval_and_watch.py`** — combined eval + WebSocket heartbeat. Launched immediately after `bg_launch.py` (no gap). Prints heartbeats every 15s during model loading, every 5 examples during eval. Keeps WebSocket alive for full eval duration.

3. **Redundant launch** — 2 `eval_and_watch.py` instances, 30s apart. Primary + backup. With ~60% WebSocket connection success rate from China, at least one connects 84% of the time.

## Experiment history

| Date | Session | Strategy | Session life | Eval completed | exec_acc | Key finding |
|------|---------|----------|-------------|----------------|----------|-------------|
| 2026-06-14 #1 | hackxie1998 | ws-1 watchdog → queued eval | ~13 min | No (80/100) | 0% | WS queuing burns NAT budget |
| 2026-06-14 #2 | hackxie1998 | ws-1 watchdog → queued eval | ~12 min | No (80/100) | 0% | Same pattern — gap fatal |
| 2026-06-14 #3 | xieminghack | Clean WS eval, no watchdog | ~8 min | **Yes (100/100)** | 0% (100 syntax) | `parse_create_table` regex over-matches without `;` |
| 2026-06-14 #4 | hackxie1998 | eval_and_watch.py + redundant launch | **10+ min** | **Yes (100/100)** | 0% (76 "other", 24 syntax) | Session survived; model generates valid SQL but wrong logic |

## Bugs found & fixed

| Bug | Symptom | Root cause | Fix |
|-----|---------|-----------|-----|
| CREATE TABLE parse over-match | 100 syntax errors | `[^;]+;?` without semicolon matched through question text | `CREATE\s+TABLE\s+\w+\s*\([^)]+\)` |
| Think tags eat token budget | SQL not generated | `max_new_tokens=256` too small for Qwen3 think content | Increase to 512 |
| PeftModel path error | `ValueError: Can't find adapter_config.json` | Passed file path instead of directory | Use `adapter_dir` for `PeftModel.from_pretrained()` |
| WebSocket queuing kills eval | Session dies at 80/100 | Queued WS burns NAT budget while idle | Launch eval with fresh WS, no prior watchdog |

## Remaining: 0% exec_acc

The model generates syntactically valid SQL but with logical errors (wrong table aliases, swapped table references, ambiguous column names). This is a **training quality** issue — 500 examples × 1 epoch × LoRA rank=8 is insufficient. Next steps:

- Increase training data to 2000-5000 examples
- Increase to 3-5 epochs
- Or use larger base model (Qwen3-1.8B still fits T4 15.6 GB)

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | Load, tokenize, save as .pt (auto-split support) |
| `train.py` | LoRA SFT training loop (standalone, subprocess-safe) |
| `evaluate.py` | Execution-match evaluation (standalone) |
| `bg_launch.py` | Detached training launcher (pip install + dataset + spawn) |
| `eval_and_watch.py` | Combined eval + WebSocket watchdog heartbeats |
| `watchdog.py` | Relay chain watchdog (5-min window, auto-naming) |
| `launch.py` | Legacy inline launcher (use bg_launch.py instead) |
| `check_and_eval.py` | Legacy eval (use eval_and_watch.py instead) |
| `tar_outputs.py` | Create output tarball for cron download |
| `fetch.sh` | Cron watchtower payload (session-aware, REST fallback) |
| `deploy.sh` | One-shot deployment (provision → upload → launch → monitor) |
| `gotchas.md` | Project-specific gotchas (14 items, field-tested) |
