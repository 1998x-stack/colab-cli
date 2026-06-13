# nanochat on Colab T4

Run [karpathy/nanochat](https://github.com/karpathy/nanochat) — a full-stack ChatGPT clone — on a free-tier Colab T4 GPU. Trains a 73M-param GPT (depth=6) from scratch: tokenizer → pretraining → plots.

**Training time:** ~4 minutes | **Cost:** $0 (free tier) | **Output:** 497 KB

## Quick start

```bash
cd projects/nanochat-colab

# 1. Provision GPU (uses cb = secondary account)
cb new --gpu T4 -s nanochat

# 2. Run end-to-end: setup → train → plot → package
cb exec -f run_all.py --timeout 900

# 3. Download results
cb download /content/nanochat-output.tar.gz ./

# 4. Clean up
cb stop -s nanochat
```

Or chain everything:

```bash
cb new --gpu T4 -s nanochat && \
  cb exec -f run_all.py --timeout 900 && \
  cb download /content/nanochat-output.tar.gz ./ && \
  cb stop -s nanochat
```

## What it does

`run_all.py` orchestrates the full nanochat pipeline:

1. **Setup (~3 min):** Installs `uv`, clones nanochat, `uv sync --extra gpu`, downloads 5 ClimbMix dataset shards, trains a BPE tokenizer (500M chars, vocab 32K)
2. **Training (~4 min):** Pretrains a d=6 GPT model (73.5M params, fp16, GradScaler) for 250 iterations with validation every 50 steps
3. **Visualization:** Parses the training log, generates dashboard + loss curve plots (matplotlib)
4. **Package:** Creates a tarball with plots, training log, and tokenizer

## Results (June 2026 run)

| Metric | Value |
|--------|-------|
| Model | depth=6, head-dim=64, 73.5M params |
| Steps | 250 |
| Final loss | 5.638 |
| Min validation BPB | 1.757 |
| Throughput | ~17,000 tok/sec |
| Training time | 3.84 min |
| Peak VRAM | 1,382 MiB |
| GPU | Tesla T4 (15 GB) |

## Output structure

```
nanochat-output.tar.gz
├── plots/
│   ├── dashboard.png    # Combined: loss + val BPB + throughput + stats
│   └── loss.png         # Training loss + validation BPB curves
├── train.log            # Full 312-line training log
└── tokenizer/           # Trained BPE tokenizer (32K vocab)
```

## Files

| File | Purpose |
|------|---------|
| `run_all.py` | End-to-end script: setup → train → plot → package. Runs inline (no detached subprocess) so the VM stays alive. |
| `launch.py` | Alternative launcher that spawns training as a detached subprocess. Use when you want to check progress mid-training via `check_progress.py`. |
| `check_progress.py` | Monitors training: process status, log tail, checkpoint listing. |
| `visualize.py` | Standalone plot generator. Parses `train.log` and produces dashboard + loss PNGs. |

## T4-specific tuning

The default nanochat config targets 8×H100. On T4 (15 GB VRAM, no bf16, no FA3), these overrides are essential:

```bash
NANOCHAT_DTYPE=float16          # T4 has no bf16 — force fp16 tensor cores
--depth=6                        # Small model to fit VRAM
--max-seq-len=256                # Reduced context window
--device-batch-size=1            # Minimal VRAM per micro-batch
--total-batch-size=16384         # Override auto-computed 262K (would need 1024 grad accum steps)
--window-pattern=L               # Full attention only — SDPA fallback can't do sliding window
--num-iterations=250             # Completes in ~4 min (safe within free-tier session window)
--eval-every=50                  # Frequent enough for meaningful val curves
--core-metric-every=-1           # Skip CORE eval (too slow on T4)
--sample-every=-1                # Skip text generation samples during training
--save-every=-1                  # Only save checkpoint at end (avoids mid-training disk I/O)
```

## Gotchas

See [`docs/model-gotchas.md`](../../docs/model-gotchas.md#nanochat-karpathynanochat) for detailed write-ups:

1. **Auto-computed batch size kills speed** — always set `--total-batch-size` explicitly on T4
2. **`--window-pattern=L` required** — SDPA fallback has no sliding window support
3. **`NANOCHAT_DTYPE=float16`** — T4 has no bf16, defaults to float32 (3× slower)
4. **Checkpoints bloat downloads** — 700MB+ tarballs fail over proxy; skip checkpoints, ship log + plots only
5. **`colab run` auto-terminates** — use `colab new` + `exec` + `download` + `stop` for persistent sessions
6. **`uv sync` downloads separate torch** — Colab's pre-installed PyTorch 2.11 is ignored in favor of nanochat's pinned 2.9.1
7. **MFU reports 0.00%** — bf16 reference FLOPS doesn't apply to T4's fp16; use `tok/sec` instead
8. **Session death on proxy hiccups** — chain `exec && download` and retry on SSL errors

## Multi-account

This project uses `cb` (account: `stefaniehu929@gmail.com`) to avoid conflicting with the primary Colab account. See [`docs/multi-account-colab.md`](../../docs/multi-account-colab.md) for setup.

## References

- [nanochat repo](https://github.com/karpathy/nanochat)
- [nanochat: Beating GPT-2 for <<$100](https://github.com/karpathy/nanochat/discussions/481)
- [Colab CLI skill](../../.claude/skills/colab-cli/SKILL.md)
