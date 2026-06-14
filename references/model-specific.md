# Model-Specific Gotchas

Project-specific issues from Colab T4 deployments.

## nanoGPT (karpathy/nanoGPT)

2026-06-10 | T4 | Free tier | 10.75M params

- **`configure_optimizers` missing from stripped-down GPT.** Copy `configure_optimizers()`, `get_num_params()`, and `estimate_mfu()` when bundling model into a single script. Requires `import inspect` for fused AdamW check.
- **`torch.cuda.amp.GradScaler` deprecated in PyTorch 2.11.** Use `torch.amp.GradScaler("cuda", ...)`.
- **T4 no bf16 compilation.** `torch.compile` with bfloat16 emits warning and falls back to eager. Use float16 or skip compile.
- **`estimate_loss()` returns tensors, not floats.** Use `.item()` before JSON serialization.
- **Free-tier sessions die in ~10-12 min.** Budget ~7 min actual training. 500 iters (7 min) safe; 800 iters (10.5 min) died twice.
- **Checkpoint bloats with torch.compile.** 126 MB (compiled) vs 42 MB (eager) for 10.75M model.

## nanochat (karpathy/nanochat)

2026-06-10 | T4 | Free tier | depth=6 (73.5M params)

- **Auto-computed batch size kills T4 speed.** nanochat computes `total_batch_size` from scaling laws (262K tokens). Override with `--total-batch-size=16384` for T4.
- **`--window-pattern=L` required.** T4 no Flash Attention 3 → SDPA fallback, no sliding window support.
- **`NANOCHAT_DTYPE=float16`.** T4 SM 7.5 < 8.0 → no bf16. Override or training runs in float32 (3x slower).
- **Checkpoint bloat kills proxy downloads.** Final checkpoint ~700MB, proxy breaks at ~624MB. Skip checkpoints in output tarballs.
- **`colab run` auto-terminates.** Use persistent workflow: `new → exec → download → stop`.
- **`uv sync` installs separate torch.** Colab has PyTorch 2.11.0 pre-installed but nanochat pins 2.9.1. Relax version pin or accept ~85s extra setup.
- **MFU reports 0.00% on T4 with fp16.** MFU calculation uses bf16 peak as reference. Use tok/sec instead.

## Transformer IWSLT

2026-06-11 | T4 | Free tier | 61M params | 206K pairs

- **IWSLT data access: 5 failed approaches.** Final working: canonical uppercase org + correct ZIP path + `urllib.request.urlretrieve`. HF CDN uses 302 for canonical URLs (urllib follows), 307 for case mismatches (doesn't follow).
- **DataLoader num_workers>0 hangs on Colab.** Use `num_workers=0`. Pre-tokenization eliminates throughput concern.
- **First epoch overhead: 7-10 min.** ZIP download 30s + BPE training 60s + pre-tokenization 30s + CUDA JIT 2-3 min + first epoch 2-5 min.
- **AMP on T4: 2-3× speedup.** Per-epoch drops from ~5 min to ~2 min.
- **Pre-tokenization eliminates 80% of DataLoader overhead.** 206K encode calls per epoch → pre-tokenize once, save to `.pt`.
- **BLEU eval is the hidden bottleneck.** 41K val pairs × beam=4 × max_len=128 → ~3-5 hours per epoch. Use 100-sentence subset with greedy decode for training-time eval.
- **CUDA OOM during eval.** Model fits training (9.3 GB) but beam search allocates extra tensors. Use `torch.cuda.empty_cache()` before eval, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`flush=True` is essential.** Even with `python -u` and `PYTHONUNBUFFERED=1`, output to subprocess files buffers. Check `nvidia-smi` to verify training is actually running.

## text2sql_finetune

2026-06-14 | T4 | Free tier | Qwen3-0.6B + LoRA rank=8

- **b-mc2/sql-create-context only has `train` split.** Use `dataset.py --split auto` or manual split. Direct `--split test` crashes.
- **`parse_create_table` over-matches without semicolon.** Dataset CREATE TABLE lacks `;`. Regex `[^;]+;?` eats through question text. Fix: `CREATE\s+TABLE\s+\w+\s*\([^)]+\)`.
- **Qwen3 think tags eat token budget.** Chat template injects `<think>...</think>` around assistant response. With `max_new_tokens=256`, think content consumes budget before SQL. Fix: `max_new_tokens=512` + strip think tags in extraction.
- **PeftModel.from_pretrained() needs directory.** Passing `adapter_config.json` file path raises ValueError. Use the directory containing adapter files.
- **500 examples × 1 epoch insufficient.** Model generates syntactically valid SQL but with logical errors (wrong table aliases, swapped table references). Need more data + epochs.
- **WebSocket queuing kills eval.** Launching eval while watchdog runs burns NAT budget during idle queue time. Fix: combined eval+watchdog script with fresh WebSocket.
- **Output dir: hyphens.** `/content/text2sql-finetune-output` (not underscore — different from project name).
- **torchao import can fail in subprocess.** If `bg_launch.py` subprocess fails with torchao error, use `launch.py` (inline) or remove torchao from DEPS.
