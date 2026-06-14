# text2sql_finetune gotchas

Project-specific surprises from the design→implementation journey (2026-06-14).

## Dataset

### b-mc2/sql-create-context only has a 'train' split

There is no `test` or `validation` split. Calling `load_dataset("b-mc2/sql-create-context", split="test")` raises `Unknown split "test"`. Use `dataset.py --split auto` to auto-split from available data, or manually split after loading `train`.

**Observed:** The design spec assumed separate `train` and `test` splits. Fixed by adding `--split auto` mode and manual split logic in `bg_launch.py`.

### Some examples are skipped due to context truncation

If a schema + question is very long (>1024 tokens), the answer tokens are completely truncated. `dataset.py` skips these (labels are all -100). With the default 600 examples loaded, typically 5-15 get skipped. Monitor the "Saved N examples" output to verify.

## Colab deployment

### GPU session death: WebSocket is the ONLY liveness signal

Colab free-tier GPU sessions die 2-5 minutes (typically ~3 min) after the last WebSocket closes. The keep-alive daemon (`KeepAliveAssignment` RPC) is permanently broken — IAM deadlock kills it at T+61s every session. There is NO backup mechanism.

**Kill chain:**
```
bg_launch.py returns → WS closes → 2-3 min grace period → session reclaimed
```

Training runs detached (survives WS drops), but the VM is reclaimed regardless.

**Fix: `eval_and_watch.py`** — a combined eval + watchdog script that:
- Launches immediately after `bg_launch.py` (no gap)
- Prints heartbeats every 15s during model loading
- Prints heartbeats every 5 examples during eval
- Keeps the WebSocket alive for the full eval duration (3-5 min)
- Uses the fixed `parse_create_table` regex and `max_new_tokens=512`

**Redundancy:** Launch 2 `eval_and_watch.py` instances (30s apart). If the first WebSocket fails (~40% chance from China), the second covers it. Probability of at least one connecting: 84%.

### bg_launch.py is the primary launcher — not launch.py

`launch.py` inlines all training/eval code to avoid `torchao` import issues with subprocess. But this duplicates logic from `train.py` and `evaluate.py`. `bg_launch.py` spawns `train.py` as a detached subprocess — survives WebSocket drops, simpler, one source of truth. Use `bg_launch.py` unless you hit the torchao issue.

### Training fits in single WebSocket window (~8 min)

With 500 examples, batch_size=4, grad_accum=2: ~125 steps, ~3-4 minutes training. Plus eval (~2 min) and setup (~2 min) = ~8 min total. This fits in a single `colab exec --timeout 540` window. The watchdog relay is only needed for hyperparameter sweeps or larger datasets.

### WebSocket connection is ~60% reliable from China

If `colab exec -f bg_launch.py` fails with `TimeoutError` at the chdir step, the WebSocket connection failed (not the session). Retry immediately — the session is still alive. If it fails 3 times in a row, stop and re-provision.

### warmup session saves ~7 minutes

First Colab session on a project downloads model weights (~1.4 GB), tokenizer, and datasets. Combined with CUDA JIT compilation: ~7-10 min overhead. The documented pattern: create a short warmup session → `colab exec -f bg_launch.py` → wait for training to start → `colab stop` → re-provision for the real run. Second session has everything cached in Colab's backend storage.

## Model / Training

### LoRA weights are ~2 MB — no checkpoint download issues

Unlike full model checkpoints (600MB+), LoRA adapter weights are tiny. The proxy download limit (~624MB) is irrelevant for this project. `lora_weights/adapter_config.json` + `adapter_model.safetensors` ≈ 2 MB total.

### Qwen3-0.6B uses ~2.5 GB VRAM with LoRA

Base model in bfloat16: ~1.4 GB. LoRA adapters + optimizer states + activations: ~1.1 GB. Total: ~2.5 GB. T4 has 15.6 GB — plenty of headroom. Could increase batch size to 8 or use a larger model (Qwen3-1.8B).

### Evaluation needs both the base model AND LoRA weights in memory

`evaluate.py` loads the full base model THEN applies LoRA adapters. Peak memory: ~2.8 GB (base + LoRA + eval batch). This is fine for T4 but would OOM on a smaller GPU.

### SQL execution match is the primary metric, not exact string match

Two SQL queries can produce identical result sets with different syntax (e.g., `SELECT a, b` vs `SELECT b, a` with different column ordering — same rows). The exec_match metric captures this; exact_match is a useful secondary metric but overly strict.

## Eval bugs (field-tested, 2026-06-14)

### parse_create_table over-matches without semicolon

The original regex `CREATE\s+TABLE\s+[^;]+;?` with optional semicolon causes `[^;]+` to match everything from the table name to end-of-string (or next `;`) — including question text, newlines, and special tokens. This produces invalid CREATE TABLE statements like:
```
CREATE TABLE head (age INTEGER)\n\nQuestion: How many heads are older than 56?<|im_end|>
```

SQLite rejects these, and all subsequent queries fail with "no such table." Fix: use parenthesis-bounded regex `CREATE\s+TABLE\s+\w+\s*\([^)]+\)`.

### max_new_tokens=256 too small for Qwen3 think tags

Qwen3 generates `<think>...</think>` content before SQL. With only 256 new tokens, the think block can consume the budget, leaving no room for SQL. Fix: `max_new_tokens=512`.

### 100 syntax errors had two root causes

The over-matching regex (no valid CREATE TABLE → all queries fail) + think tags eating token budget (SQL not generated). Both fixed in `evaluate.py`, `check_and_eval.py`, `launch.py`, and `eval_and_watch.py`.

## Infrastructure

### Output dir uses hyphens: text2sql-finetune-output

The output directory is `/content/text2sql-finetune-output` (hyphens). Don't confuse with the project name `text2sql_finetune` (underscore). This is the directory name on the VM, not the Python package name.

### fetch.sh exits 0 on missing session (doesn't kill cron)

Unlike the old version, `fetch.sh` no longer exits 1 when the session is gone. It reports the issue and exits 0 so the cron job survives — you can re-provision and the cron continues monitoring.

### tar_outputs.py handles empty output dirs

If no training output exists yet (training hasn't started), `tar_outputs.py` creates an empty tar instead of failing. This prevents the fetch.sh fallback path from triggering unnecessarily.

## Package / Version

### torchao import can fail in subprocess

The comment in `launch.py` about "avoids torchao version issues" refers to a real problem: `import torchao` can fail in a subprocess on Colab due to CUDA library path issues. If `bg_launch.py` fails with an `import torchao` error during dataset prep, switch to `launch.py` (inline) or remove `torchao` from the DEPS list (it's not actually used by this project).
