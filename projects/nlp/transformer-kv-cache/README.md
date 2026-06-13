# Transformer with KV Cache

Character-level GPT trained on tiny Shakespeare. Demonstrates KV cache speedup for autoregressive inference.

## Architecture
- Decoder-only transformer (GPT-style)
- Pluggable attention: standard MHA (training) + cache-aware MHA (inference)
- Config-driven: all hyperparameters in config.py, overridable via CLI

## Quickstart
```
python train.py  # train on CPU
python train.py --device cuda  # train on GPU
python generate.py --checkpoint output/checkpoints/weights_epoch10.pt  # demo KV cache speedup
```

## Files
| File | Purpose |
|------|---------|
| config.py | All hyperparameters + CLI parsing |
| kv_cache.py | KVCache data structure (pluggable) |
| attention.py | MHA + CausalMHAWithCache |
| model.py | GPT transformer assembly |
| train.py | Training loop + logging + metrics |
| generate.py | Inference: with/without KV cache comparison |
| charts.py | Training curves + inference speedup charts |
| launch.py | Colab bootstrap |
| check_progress.py | Remote progress monitor |
| fetch.sh | Cron artifact pull script |
