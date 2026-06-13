# Transformer IWSLT — Project Gotchas

Field-tested issues from deploying the Transformer (Attention Is All You Need) on IWSLT'17 De->En across 3 Colab T4 accounts.

## Data pipeline

### IWSLT 2017 data access requires specific URL

5 different approaches failed. The working one: raw ZIP from HF CDN with canonical org casing.

```python
# Works: IWSLT (uppercase) → 302 redirect (urllib follows)
IWSLT_ZIP_URL = "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/de/en/de-en.zip"

# Fails: iwslt2017 (lowercase) → 307 redirect (urllib doesn't follow 307)
# Fails: wit3.fbk.eu → Google auth wall
# Fails: datasets.load_dataset("iwslt2017", ...) → version incompatibility
```

ZIP internal structure: `de-en/train.tags.de-en.de` and `de-en/train.tags.de-en.en`.

### Training files are plain text, not `<seg>` wrapped

Despite what the IWSLT dataset script suggests, the `train.tags.de-en.*` files in the de-en.zip have:
- XML meta tags in header: `<doc>`, `<url>`, `<keywords>`, `<speaker>`, `<talkid>`, `<title>`, `<description>`
- Plain text sentences (one per line) after the header

Parser logic:
```python
def _parse_iwslt_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("<"):
        return None  # skip meta tags
    return line  # plain text sentence
```

### Parallel filtering for DE/EN alignment

DE and EN files must be filtered for empty lines **in parallel** — both files must have content at the same position. Independent filtering can misalign sentences:
```python
# Correct: filter in parallel
pairs = [(d, e) for d, e in zip(de_orig, en_orig) if d and e]

# Wrong: filter independently → misaligned pairs
de_lines = [l for l in de_orig if l.strip()]
en_lines = [l for l in en_orig if l.strip()]
```

## Training performance

### Pre-tokenization is essential for viable training speed

Without pre-tokenization, `tokenizer.encode()` is called per-sample in `__getitem__`, dominating DataLoader time. Pre-tokenize once after training the tokenizer, cache to disk:
- Initial pre-tokenization: ~30s for 206K pairs
- Reload: near-instant via `torch.load()`
- Per-epoch speedup: 3-5× faster DataLoader

### AMP mixed precision: mandatory on T4

Float16 tensor cores give ~8× matmul throughput. Without AMP, first epoch takes ~8 min → barely completes within session window. With AMP: ~2-3 min per epoch. Use `torch.amp.GradScaler("cuda")` + `torch.amp.autocast("cuda")` (PyTorch 2.11 API, not deprecated `torch.cuda.amp`).

### num_workers=0 on Colab

Multiprocessing DataLoader with the Rust `tokenizers` library hangs silently. No error, no crash — just stalls after model init.

### First epoch is misleadingly slow

CUDA JIT compilation adds 2-3 min to the first epoch. Combined with data prep (download, extraction, tokenizer, pre-tokenization), the first epoch can take 7-10 min. Subsequent epochs are 2-3 min. First session rarely gets past epoch 1; second session gets 3-4 epochs.

## Model architecture

### Weight tying: encoder.embed, decoder.embed, out_proj must share the SAME tensor

```python
self.out_proj.weight = self.decoder.embed.weight  # shares Parameter object
self.encoder.embed.weight = self.decoder.embed.weight  # all three tied
```

This reduces params from ~76M to ~61M (32K vocab, d_model=512). Modern PyTorch optimizers handle shared weights correctly (deduplicate by object identity).

### Noam LR scheduler: call before optimizer.step(), not after

The initial LR of 0.0 in Adam combined with calling `scheduler.step()` *after* `optimizer.step()` means the first batch uses LR=0.0 — wasted step. Swap the order:
```python
scheduler.step()    # compute LR for THIS step
optimizer.step()    # use computed LR
```

### Beam search: finished beams must only allow EOS

When a beam hits EOS, set all other token log-probs to -inf so only EOS is selected in subsequent steps. Otherwise finished beams keep growing with random tokens, wasting GPU compute and diluting scores:
```python
log_probs[finished] = float("-inf")
log_probs[finished, eos_idx] = 0.0
```

## Session orchestration

### Checkpoint-resume must preserve scheduler state

The NoamScheduler's step counter determines the LR. On resume, restore:
- Model weights → `model.load_state_dict()`
- Optimizer moments → `optimizer.load_state_dict()`
- Scheduler step counter → `scheduler.load_state_dict()` (custom class, must implement state_dict/load_state_dict)
- Epoch number → loop starts from `start_epoch + 1`
- Token count + wall time → cumulative metrics

### Tokenizer must be saved alongside checkpoints

The BPE tokenizer is trained on the specific dataset and vocab size. Checkpoints are useless without the matching tokenizer. Save `tokenizer.json` to the same persistent location as checkpoints.

### ZIP is cached, extracted files are not

The 16MB de-en.zip is downloaded once and survives session restarts (if saved to Drive or re-downloaded each session). But extracted `train.de`, `train.en`, and `tokenizer.json` vanish with the VM. On each fresh session, re-extract from the cached ZIP and re-train the tokenizer.

## Monitoring

### REST download as exec fallback

When `colab exec` WebSocket drops (404/401), `colab download` still works — it uses the REST API. Use as fallback monitoring:
```bash
# exec check (may fail due to WebSocket)
colab exec -s <session> -f check_progress.py --timeout 15

# REST fallback (works even when exec is dead)
colab download /content/metrics.jsonl ./output/metrics.jsonl
```

### Log file appears stuck but training is running

Training prints per-epoch, not per-batch. With 2-5 min per epoch and CUDA JIT on the first batch, the log file may show no new output for 5+ minutes. Verify with `nvidia-smi` or `ps aux`.

## Data format

### IWSLT 2017 vs 2014

This project originally targeted IWSLT 2014 but uses IWSLT 2017 because:
- IWSLT 2014 has ~160K pairs; IWSLT 2017 has ~206K pairs
- IWSLT 2014 is behind Google auth wall; IWSLT 2017 is available on HF CDN
- Same task (TED talks De→En translation), comparable difficulty
- Both use the same XML-tag format in the official release, but the HF ZIP version has pre-stripped tags

## Verified training metrics (Colab T4, 50K pairs)

| Metric | Baseline (epoch 1) |
|---|---|
| Train loss | 25.6 |
| Val loss | 9.23 |
| BLEU (100 sentences, greedy) | 0.9 |
| Wall time | 332s (5.5 min) |
| GPU memory | ~9.3 GiB |
| Batch size | 32 |

Full 20-epoch training is expected to reach >25 BLEU (baseline target). Each session gets 1-3 epochs depending on data cache state.

## Post-documentation discoveries

### BLEU evaluation bottleneck

BLEU eval on the full validation set (41K sentences × beam search) takes 3-5 HOURS per epoch. Fixed by evaluating on 100 sentences with greedy decode (beam_size=1). Full BLEU with beam_size=4 only at final evaluation.

### CUDA OOM during eval

Model at batch_size=64 uses 14.5 GiB — triggers OOM during evaluation. Fixed: batch_size=32, `torch.cuda.empty_cache()` before eval, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Log buffering hides training progress

stdout redirected via Popen is buffered even with `PYTHONUNBUFFERED=1`. Training appears "stuck" after model init. Fixed with `flush=True` on all prints and per-200-batch progress indicators. Verify with `nvidia-smi` on the VM.

### Checkpoint download size limit

Full checkpoint (weights + Adam optimizer moments) = ~1GB. Proxy connection breaks at ~624MB. Gzip at level 3 only reduces to ~500MB — still fails. **Weights-only checkpoint** (~233MB) downloads reliably. Both are saved per epoch:
- `checkpoint_epochN.pt` — full (VM-local resume)
- `weights_epochN.pt` — weights only (proxy download, ~233MB)

### Smoke test before real training

A 200-pair, 3-epoch test on Colab verifies the entire pipeline in 96 seconds. Catches 80% of bugs before wasting session time. Always run before deploying real training.

### Pre-tokenization cache key collision

Multiple `TranslationDataset` instances (train, val, val_bleu) collided on `train.pt` cache path — val set loaded training data. Fixed by keying cache by dataset name (`train.pt`, `val.pt`, `val_bleu.pt`).
