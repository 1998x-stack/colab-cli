# Transformer IWSLT

Transformer (Attention Is All You Need) base model on IWSLT'17 German-to-English translation using PyTorch, with BPE tokenization, Noam LR scheduler, beam search decoding, and sacreBLEU evaluation.

## Usage

```bash
# Local training
python train.py --exp_id baseline

# Colab deployment (see ../.claude/skills/colab-cli/SKILL.md)
cb launch.py
```

Experiments: `baseline` (learned positional encodings), `fixed_pe` (sinusoidal positional encodings), `heads_1` (single-head attention).

## Key results

| Metric | Value |
|--------|-------|
| Model params | ~61M (with weight tying) |
| Dataset | IWSLT'17 De-En, ~206K pairs |
| Baseline epoch 1 train loss | 25.6 |
| Baseline epoch 1 val loss | 9.23 |
| Baseline epoch 1 BLEU (100 sents, greedy) | 0.9 |
| Baseline epoch 1 wall time | 332s (5.5 min) |
| Projected BLEU at 20 epochs | >25 |

## Gotchas

- IWSLT 2017 ZIP requires canonical HuggingFace org casing (`IWSLT`, not `iwslt2017`) for urllib to follow the 302 redirect.
- Pre-tokenization (cache to disk) is essential -- without it, `tokenizer.encode()` in every `__getitem__` dominates DataLoader time (3-5x slowdown).
- AMP mixed precision is mandatory on T4. Without it, the first epoch takes ~8 min; with it, ~2-3 min.
- `num_workers=0` on Colab -- multiprocessing hangs silently with the Rust `tokenizers` library.
- First epoch is slow due to CUDA JIT compilation (2-3 min overhead).
- Weight tying between encoder embed, decoder embed, and output projection reduces params from ~76M to ~61M.
- Noam scheduler must call `scheduler.step()` *before* `optimizer.step()` to avoid a wasted first step at LR=0.
- Beam search must lock finished beams to EOS-only to prevent score dilution.
- Full checkpoints (~1GB) fail proxy download at ~624MB. Weights-only checkpoints (~233MB) are saved alongside for reliable download.
- BLEU eval on the full validation set with beam search takes hours; limited to 100 sentences per epoch.
- Tokenizer must be saved alongside checkpoints -- checkpoints are useless without the matching vocab.
