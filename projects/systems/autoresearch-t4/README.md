# AutoResearch T4 — GPT Pretraining on a Single T4 GPU

Autonomous LLM pretraining research with a fixed 5-minute budget on an NVIDIA T4 (16 GB). Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Uses a GPT-style model with RoPE, RMSNorm, ReLU^2 MLP, and MuonAdamW optimizer.

## Usage

```bash
# Local training
python train.py

# Colab deployment (T4 GPU recommended)
cb launch.py --timeout 600
```

The launch script runs two phases:
1. `prepare.py` — Downloads TinyStories dataset, trains a BPE tokenizer (vocab=2048)
2. `train.py` — GPT training with a strict 300-second budget

## Key results

| Metric | Best (V1) |
|--------|-----------|
| val_bpb | **1.0067** |
| Model params | 4.2M |
| Depth / Embed / Heads | 4 / 256 / 4 |
| Batch tokens | 16K |
| Tokens processed | 20.1M |
| Throughput | 66K tok/s |
| Peak VRAM | 933 MB |

## Key findings

- **Depth is the bottleneck** — each additional layer slows throughput significantly. V2 (6 layers) ran at 30K tok/s vs V1 (4 layers) at 66K tok/s.
- **Width over depth** — wider embeddings scale better than deeper networks on T4.
- **Batch size is free** — V4 used batch=64 (32K total tokens) and improved throughput without hurting convergence.

## Gotchas

- The `train.py` requires a tokenizer trained first by `prepare.py`. Always run prepare.py before training.
- `torch.compile` has a one-time ~23s compilation cost; this amortizes over longer training runs.
- Time budget is hard-coded to 300 seconds; adjust `TIME_BUDGET` in `prepare.py` to change it.
- The Muon optimizer uses Newton-Schulz iterations for matrix orthogonalization.
- Outputs saved to `/content/autoresearch-output/` on Colab, including `model.pt` and `metrics.json`.
