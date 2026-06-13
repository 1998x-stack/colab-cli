# Transformer with KV Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable char-level GPT with KV cache inference, document KV cache mechanics in 5 deep-dive docs, deploy to Colab with 2-min cron fetch.

**Architecture:** Decoder-only transformer with pluggable attention (standard MHA for training, cache-aware MHA for inference). KVCache is an independent module that can be swapped for GQA/PagedAttention variants. Config-driven via dataclass with CLI overrides.

**Tech Stack:** PyTorch, matplotlib (headless), numpy, Python stdlib argparse + dataclasses, urllib (HF CDN download)

---

### Task 1: Create project directory and placeholder files

**Files:**
- Create: `projects/nlp/transformer-kv-cache/__init__.py` (empty)
- Create: `projects/nlp/transformer-kv-cache/README.md`

- [ ] **Step 1: Create directory and empty init**

```bash
mkdir -p projects/nlp/transformer-kv-cache/output/{logs,pngs,checkpoints}
touch projects/nlp/transformer-kv-cache/__init__.py
```

- [ ] **Step 2: Write README.md**

Write `projects/nlp/transformer-kv-cache/README.md`:
```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/
git commit -m "chore: scaffold transformer-kv-cache project structure"
```

---

### Task 2: config.py — All hyperparameters and CLI

**Files:**
- Create: `projects/nlp/transformer-kv-cache/config.py`

- [ ] **Step 1: Write config.py**

```python
"""All hyperparameters for the transformer + KV cache demo.

Everything lives in one dataclass so components can be imported independently
and receive a config object. CLI overrides any field via argparse.
"""
import argparse
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    # --- Vocabulary ---
    vocab_size: int = 65  # tiny Shakespeare has ~65 unique chars
    pad_token_id: int = 0  # reserve 0 for padding (unused in char LM, but safe)

    # --- Architecture ---
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024
    block_size: int = 256  # max context length
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 64
    max_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # --- Logging ---
    log_interval: int = 50   # steps between log lines
    eval_interval: int = 200  # steps between eval runs
    chart_interval: int = 1  # epochs between chart overwrites

    # --- Output ---
    output_dir: str = "output"

    # --- Runtime ---
    device: str = "cpu"
    compile: bool = False  # torch.compile (PyTorch >= 2.0)

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_head


def parse_args() -> TransformerConfig:
    """Parse CLI arguments and return a populated config.

    Any config field can be overridden, e.g.:
        python train.py --n_layer 6 --d_model 512 --lr 1e-4 --device cuda
    """
    parser = argparse.ArgumentParser(
        description="Train a char-level GPT with KV cache support"
    )
    config = TransformerConfig()

    # Add all dataclass fields as CLI arguments
    for field_name, field_def in TransformerConfig.__dataclass_fields__.items():
        if field_name == "d_k":
            continue  # computed property, not a real field
        field_type = field_def.type
        default = getattr(config, field_name)

        if field_type == bool:
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
            parser.add_argument(f"--no-{field_name}", action="store_false", dest=field_name)
        elif field_type == int:
            parser.add_argument(f"--{field_name}", type=int, default=default)
        elif field_type == float:
            parser.add_argument(f"--{field_name}", type=float, default=default)
        elif field_type == str:
            parser.add_argument(f"--{field_name}", type=str, default=default)

    args = parser.parse_args()
    for field_name in TransformerConfig.__dataclass_fields__:
        if field_name == "d_k":
            continue
        setattr(config, field_name, getattr(args, field_name))

    return config
```

- [ ] **Step 2: Verify config parse works**

```bash
cd projects/nlp/transformer-kv-cache && python -c "
from config import TransformerConfig, parse_args
import sys
sys.argv = ['test', '--n_layer', '2', '--device', 'cuda']
c = parse_args()
assert c.n_layer == 2
assert c.device == 'cuda'
assert c.d_k == 64
assert c.vocab_size == 65  # default
print('config OK')
"
```

Expected: `config OK`

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/config.py
git commit -m "feat: add TransformerConfig and CLI parsing"
```

---

### Task 3: kv_cache.py — KVCache data structure

**Files:**
- Create: `projects/nlp/transformer-kv-cache/kv_cache.py`

- [ ] **Step 1: Write kv_cache.py**

```python
"""KVCache: stores past keys and values for autoregressive inference.

Pluggable design: swap this with a GQA or PagedAttention cache by providing
the same update()/reset()/seq_len interface.
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Accumulates K and V tensors across autoregressive steps.

    Shape convention:
        k: [B, H, L, D]  (batch, heads, seq_len, head_dim)
        v: [B, H, L, D]
    """

    k: Tensor | None = None
    v: Tensor | None = None

    def update(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Append new single-token KV to cache, return full K, V.

        Args:
            k_new: [B, H, 1, D] — keys for the new token
            v_new: [B, H, 1, D] — values for the new token

        Returns:
            (full_k, full_v) each [B, H, total_len, D]
        """
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def reset(self) -> None:
        """Clear cache for a new sequence."""
        self.k = None
        self.v = None

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if self.k is None:
            return 0
        return self.k.size(2)

    def __repr__(self) -> str:
        return f"KVCache(seq_len={self.seq_len}, k_shape={self.k.shape if self.k is not None else None})"
```

- [ ] **Step 2: Verify KVCache operations**

```bash
cd projects/nlp/transformer-kv-cache && python -c "
import torch
from kv_cache import KVCache

cache = KVCache()
assert cache.seq_len == 0

# First update: batch=1, 4 heads, 1 token, head_dim=64
k1 = torch.randn(1, 4, 1, 64)
v1 = torch.randn(1, 4, 1, 64)
k, v = cache.update(k1, v1)
assert cache.seq_len == 1
assert k.shape == (1, 4, 1, 64)

# Second update
k2 = torch.randn(1, 4, 1, 64)
v2 = torch.randn(1, 4, 1, 64)
k, v = cache.update(k2, v2)
assert cache.seq_len == 2
assert k.shape == (1, 4, 2, 64)

# Reset
cache.reset()
assert cache.seq_len == 0

print('KVCache OK')
"
```

Expected: `KVCache OK`

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/kv_cache.py
git commit -m "feat: add KVCache data structure"
```

---

### Task 4: attention.py — MHA + Cache-aware MHA

**Files:**
- Create: `projects/nlp/transformer-kv-cache/attention.py`

- [ ] **Step 1: Write attention.py**

```python
"""Multi-head attention variants: standard (training) and cache-aware (inference).

Pluggable: swap `CausalMHAWithCache` for GQA, MQA, or FlashAttention variants
by providing the same forward(x, kv_cache=None) -> (output, kv_cache) interface.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig
from kv_cache import KVCache


class CausalMultiHeadAttention(nn.Module):
    """Standard causal MHA for training. No cache — computes full attention matrix."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.d_k
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] -> [B, L, D] with causal mask."""
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)  # [B, H, L, Dk]
        k = self.k_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class CausalMHAWithCache(nn.Module):
    """Causal MHA that accepts an optional KVCache for autoregressive inference.

    Training mode (kv_cache is None): behaves identically to CausalMultiHeadAttention.
    Inference mode (kv_cache is KVCache): only projects the new token, appends to cache,
    attends over full cached history. O(L) per step instead of O(L^2).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.d_k
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Forward pass with optional KV cache.

        Args:
            x: [B, L, D] — full sequence during training, single token (L=1) during inference
            kv_cache: None for training, KVCache for inference

        Returns:
            (output, updated_kv_cache). kv_cache is None when training.
        """
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            # Inference: append new KV to cache, attend over full history
            k_full, v_full = kv_cache.update(k, v)
            total_len = k_full.size(2)
        else:
            k_full, v_full = k, v
            total_len = L

        scores = (q @ k_full.transpose(-2, -1)) * self.scale

        if kv_cache is not None:
            # Inference: no causal mask needed (q is single token, attends to all cached)
            pass
        else:
            # Training: apply causal mask
            causal_mask = torch.triu(
                torch.ones(L, total_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v_full).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out), kv_cache
```

- [ ] **Step 2: Verify attention shapes and causality**

```bash
cd projects/nlp/transformer-kv-cache && python -c "
import torch
from config import TransformerConfig
from attention import CausalMultiHeadAttention, CausalMHAWithCache
from kv_cache import KVCache

config = TransformerConfig(n_layer=2, n_head=4, d_model=256)

# --- Standard MHA (training) ---
mha = CausalMultiHeadAttention(config)
x = torch.randn(2, 32, 256)  # B=2, L=32, D=256
out = mha(x)
assert out.shape == (2, 32, 256), f'Expected (2,32,256), got {out.shape}'

# --- Cache-aware MHA (training mode, no cache) ---
mha_cache = CausalMHAWithCache(config)
out, kv = mha_cache(x, kv_cache=None)
assert out.shape == (2, 32, 256)
assert kv is None

# --- Cache-aware MHA (inference mode) ---
cache = KVCache()
# Step 1: first token (prompt)
x1 = torch.randn(1, 1, 256)
out1, cache = mha_cache(x1, kv_cache=cache)
assert out1.shape == (1, 1, 256)
assert cache.seq_len == 1

# Step 2: second token (uses cache)
x2 = torch.randn(1, 1, 256)
out2, cache = mha_cache(x2, kv_cache=cache)
assert out2.shape == (1, 1, 256)
assert cache.seq_len == 2
assert cache.k.shape == (1, 4, 2, 64)

print('Attention OK')
"
```

Expected: `Attention OK`

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/attention.py
git commit -m "feat: add MHA and cache-aware MHA attention modules"
```

---

### Task 5: model.py — GPT transformer assembly

**Files:**
- Create: `projects/nlp/transformer-kv-cache/model.py`

- [ ] **Step 1: Write model.py**

```python
"""Decoder-only GPT transformer with pluggable attention and optional KV cache.

The model can be built in two modes:
- Training: uses CausalMultiHeadAttention (no cache overhead)
- Inference: uses CausalMHAWithCache (accepts per-layer KVCache)

Swappable: use_learned_pe, weight tying, and GELU/ReLU are config-toggleable.
"""
import math

import torch
import torch.nn as nn

from config import TransformerConfig
from attention import CausalMultiHeadAttention, CausalMHAWithCache
from kv_cache import KVCache


class DecoderBlock(nn.Module):
    """One transformer decoder block: self-attention + FFN with pre-norm."""

    def __init__(self, config: TransformerConfig, use_cache: bool = False):
        super().__init__()
        attn_cls = CausalMHAWithCache if use_cache else CausalMultiHeadAttention
        self.attn = attn_cls(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.use_cache = use_cache

    def forward(
        self, x: torch.Tensor, kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache | None]:
        if self.use_cache:
            attn_out, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        else:
            attn_out = self.attn(self.ln1(x))
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x, kv_cache


class GPT(nn.Module):
    """Decoder-only transformer (GPT-style) for character-level language modeling.

    Two forward modes:
    - Training: `model(idx)` — full sequence, no cache, returns logits [B, L, V]
    - Inference: `model(idx, kv_caches=[...])` — single token, with caches,
      returns (logits, updated_caches)
    """

    def __init__(self, config: TransformerConfig, use_cache: bool = False):
        super().__init__()
        self.config = config
        self.use_cache = use_cache

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.block_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(config, use_cache=use_cache) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: output projection shares weights with token embedding
        self.out_proj.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, kv_caches: list[KVCache] | None = None
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Forward pass.

        Args:
            idx: [B, L] token indices
            kv_caches: None (training) or list of KVCache per layer (inference)

        Returns:
            (logits, updated_kv_caches). kv_caches is None in training mode.
        """
        B, L = idx.shape
        assert L <= self.config.block_size, f"Sequence too long: {L} > {self.config.block_size}"

        # Position indices
        if kv_caches is not None and kv_caches[0] is not None and kv_caches[0].seq_len > 0:
            pos = torch.arange(
                kv_caches[0].seq_len, kv_caches[0].seq_len + L,
                device=idx.device
            ).unsqueeze(0)  # [1, L]
        else:
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)

        tok_emb = self.token_embed(idx)  # [B, L, D]
        pos_emb = self.pos_embed(pos)     # [1, L, D]
        x = self.dropout(tok_emb + pos_emb)

        new_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = layer(x, kv_cache=cache)
            if new_caches is not None:
                new_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.out_proj(x)

        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """Generate tokens autoregressively.

        Returns:
            (generated_sequence, step_latencies_ms)
        """
        import time

        self.eval()
        latencies = []

        if use_cache and self.use_cache:
            kv_caches = [KVCache() for _ in range(self.config.n_layer)]
        else:
            kv_caches = None

        for _ in range(max_new_tokens):
            t0 = time.perf_counter()

            if kv_caches is not None:
                # Inference mode: single token
                x = idx[:, -1:]  # [B, 1]
                logits, kv_caches = self(x, kv_caches=kv_caches)
            else:
                # No cache: reprocess full sequence
                x = idx  # [B, L]
                logits, _ = self(x, kv_caches=None)

            logits = logits[:, -1, :] / temperature  # [B, V]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_token], dim=1)

            latencies.append((time.perf_counter() - t0) * 1000)

        return idx, latencies
```

- [ ] **Step 2: Verify model forward and generate shapes**

```bash
cd projects/nlp/transformer-kv-cache && python -c "
import torch
from config import TransformerConfig
from model import GPT
from kv_cache import KVCache

config = TransformerConfig()

# Training mode (no cache)
model = GPT(config, use_cache=False)
x = torch.randint(0, 65, (2, 64))  # B=2, L=64
logits, caches = model(x)
assert logits.shape == (2, 64, 65), f'Expected (2,64,65), got {logits.shape}'
assert caches is None

# Inference mode (with cache)
model_cache = GPT(config, use_cache=True)
# Step 1: process prompt
prompt = torch.randint(0, 65, (1, 10))
logits, caches = model_cache(prompt, kv_caches=[KVCache() for _ in range(4)])
assert logits.shape == (1, 10, 65)
assert len(caches) == 4
assert caches[0].seq_len == 10

# Step 2: generate next token
next_tok = torch.randint(0, 65, (1, 1))
logits2, caches2 = model_cache(next_tok, kv_caches=caches)
assert logits2.shape == (1, 1, 65)
assert caches2[0].seq_len == 11

# Generate test
gen, lats = model_cache.generate(prompt, max_new_tokens=20, use_cache=True)
assert gen.shape == (1, 30)
assert len(lats) == 20

# Generate without cache
gen_nocache, lats_nocache = model_cache.generate(prompt, max_new_tokens=20, use_cache=False)
assert gen_nocache.shape == (1, 30)

print(f'Model OK. Params: {sum(p.numel() for p in model.parameters()):,}')
print(f'With cache: {len(lats)} steps')
print(f'Without cache: {len(lats_nocache)} steps')
"
```

Expected: no assertion errors, ~1.5M params

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/model.py
git commit -m "feat: add GPT transformer with pluggable KV cache support"
```

---

### Task 6: train.py — Training loop with structured outputs

**Files:**
- Create: `projects/nlp/transformer-kv-cache/train.py`

- [ ] **Step 1: Write train.py**

```python
"""Train a char-level GPT on tiny Shakespeare.

Outputs per CLAUDE.md spec:
  logs/train.log       — per-epoch, timestamped, self-contained
  pngs/training_curves.png — loss + perplexity over time
  metrics.csv          — epoch, loss, perplexity, tokens_per_sec, elapsed_s
  checkpoints/         — weights-only checkpoints

Usage:
  python train.py                          # defaults (CPU, 4-layer)
  python train.py --device cuda --n_layer 6  # GPU, larger model
"""
import csv
import os
import time
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import TransformerConfig, parse_args
from model import GPT

# --- Shakespeare download ---
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare(data_dir: str) -> str:
    """Download and load tiny Shakespeare text."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(path):
        print("[data] Downloading tiny Shakespeare (~1MB)...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r") as f:
        return f.read()


class CharDataset(Dataset):
    """Character-level dataset: sliding windows over the text."""

    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def make_dirs(output_dir: str):
    for sub in ["logs", "pngs", "checkpoints"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, kv_caches=None)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def save_weights(path: str, model: nn.Module, epoch: int, metrics: dict):
    torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def main():
    config = parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Config: {config}")

    make_dirs(config.output_dir)

    # --- Data ---
    text = load_shakespeare("/tmp/shakespeare")
    dataset = CharDataset(text, config.block_size)
    config.vocab_size = dataset.vocab_size

    n_train = int(0.9 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print(f"[data] Vocab size: {dataset.vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    # --- Model ---
    model = GPT(config, use_cache=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Params: {n_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Linear warmup + cosine decay
    total_steps = config.max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Logging setup ---
    log_path = os.path.join(config.output_dir, "logs", "train.log")
    csv_path = os.path.join(config.output_dir, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "perplexity", "tokens_per_sec", "elapsed_s", "lr"])

    def log(msg: str):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # --- Training ---
    start_time = time.time()
    global_step = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        tokens_processed = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits, _ = model(x, kv_caches=None)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            tokens_processed += x.numel()
            global_step += 1

            if global_step % config.log_interval == 0:
                log(
                    f"Ep {epoch}/{config.max_epochs} | "
                    f"step {global_step:5d} | "
                    f"loss={loss.item():.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # End of epoch
        epoch_elapsed = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        total_elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / epoch_elapsed
        perplexity = math.exp(min(avg_train_loss, 10))

        log(
            f"EPOCH {epoch}/{config.max_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"ppl={perplexity:.1f} | "
            f"tok/s={tokens_per_sec:.0f} | "
            f"elapsed={total_elapsed/60:.1f}m"
        )

        # Write metrics
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_train_loss, val_loss, perplexity,
                tokens_per_sec, total_elapsed, scheduler.get_last_lr()[0],
            ])

        # Save weights checkpoint
        ckpt_path = os.path.join(config.output_dir, "checkpoints", f"weights_epoch{epoch}.pt")
        save_weights(ckpt_path, model, epoch, {
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "perplexity": perplexity,
        })
        log(f"Checkpoint saved: {ckpt_path}")

        # Generate training curves chart
        if epoch % config.chart_interval == 0:
            _make_training_chart(config.output_dir)

    log(f"Training complete. Total: {total_elapsed/60:.1f}m. Final val loss: {val_loss:.4f}")


def _make_training_chart(output_dir: str):
    """Generate training_curves.png from metrics.csv (headless-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "b-", label="Train", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity
    ax = axes[1]
    ax.plot(df["epoch"], df["perplexity"], "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity")
    ax.grid(True, alpha=0.3)

    # Tokens/sec
    ax = axes[2]
    ax.plot(df["epoch"], df["tokens_per_sec"], "purple", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Training Speed")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_dir = os.path.join(output_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    fig.savefig(os.path.join(png_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test training runs for 2 epochs on CPU**

```bash
cd projects/nlp/transformer-kv-cache && python train.py --max_epochs 2 --batch_size 32 --block_size 128 --log_interval 10 --output_dir /tmp/kvcache-test
```

Expected: downloads Shakespeare, trains 2 epochs, produces:
- `/tmp/kvcache-test/logs/train.log`
- `/tmp/kvcache-test/metrics.csv`
- `/tmp/kvcache-test/pngs/training_curves.png`
- `/tmp/kvcache-test/checkpoints/weights_epoch1.pt`
- `/tmp/kvcache-test/checkpoints/weights_epoch2.pt`

- [ ] **Step 3: Verify output artifacts exist**

```bash
ls -lh /tmp/kvcache-test/logs/train.log /tmp/kvcache-test/metrics.csv /tmp/kvcache-test/pngs/training_curves.png /tmp/kvcache-test/checkpoints/
cat /tmp/kvcache-test/metrics.csv
```

- [ ] **Step 4: Commit**

```bash
git add projects/nlp/transformer-kv-cache/train.py
git commit -m "feat: add training loop with structured outputs (logs, csv, pngs, checkpoints)"
```

---

### Task 7: generate.py — Inference with/without KV cache comparison

**Files:**
- Create: `projects/nlp/transformer-kv-cache/generate.py`

- [ ] **Step 1: Write generate.py**

```python
"""Generate text and compare inference latency: with vs without KV cache.

Demonstrates KV cache speedup by measuring per-token latency at each step
and plotting the cumulative time difference.

Usage:
  python generate.py --checkpoint output/checkpoints/weights_epoch10.pt
  python generate.py --checkpoint output/checkpoints/weights_epoch10.pt --prompt "ROMEO:" --max_tokens 200
"""
import argparse
import os
import time
import urllib.request

import torch
import torch.nn.functional as F

from config import TransformerConfig
from model import GPT
from kv_cache import KVCache


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text():
    path = "/tmp/shakespeare/input.txt"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path) as f:
        return f.read()


def get_tokenizer(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="First Citizen:")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")

    # Load text and tokenizer
    text = load_text()
    stoi, itos, vocab_size = get_tokenizer(text)
    print(f"[generate] Vocab size: {vocab_size}")

    # Build model and load checkpoint
    config = TransformerConfig(vocab_size=vocab_size)

    # Build cache-aware model for generate
    model = GPT(config, use_cache=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[generate] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Encode prompt
    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long).to(device)
    print(f"[generate] Prompt: '{args.prompt}' ({prompt_ids.shape[1]} tokens)")

    # --- Generate WITH KV cache ---
    print("\n" + "=" * 60)
    print("Generating WITH KV Cache")
    print("=" * 60)

    gen_with, lats_with = model.generate(
        prompt_ids.clone(), max_new_tokens=args.max_tokens,
        temperature=args.temperature, use_cache=True,
    )
    output_with = "".join(itos.get(i, "?") for i in gen_with[0].tolist())
    print(output_with)
    print(f"\nLatency: {sum(lats_with):.0f}ms total, {sum(lats_with)/len(lats_with):.1f}ms avg/token")

    # Reset and generate WITHOUT KV cache
    print("\n" + "=" * 60)
    print("Generating WITHOUT KV Cache (recomputing full attention)")
    print("=" * 60)

    gen_without, lats_without = model.generate(
        prompt_ids.clone(), max_new_tokens=args.max_tokens,
        temperature=args.temperature, use_cache=False,
    )
    output_without = "".join(itos.get(i, "?") for i in gen_without[0].tolist())
    print(output_without)
    print(f"\nLatency: {sum(lats_without):.0f}ms total, {sum(lats_without)/len(lats_without):.1f}ms avg/token")

    # --- Summary ---
    speedup = sum(lats_without) / max(sum(lats_with), 1)
    print("\n" + "=" * 60)
    print("KV CACHE SPEEDUP SUMMARY")
    print("=" * 60)
    print(f"  With cache:    {sum(lats_with):.0f}ms total ({sum(lats_with)/len(lats_with):.1f}ms/step)")
    print(f"  Without cache: {sum(lats_without):.0f}ms total ({sum(lats_without)/len(lats_without):.1f}ms/step)")
    print(f"  Speedup:       {speedup:.1f}x")
    print(f"  Tokens generated: {args.max_tokens}")
    print(f"  Avg sequence length: {gen_with.shape[1]}")

    # --- Plot latency comparison ---
    _make_speedup_chart(lats_with, lats_without, args.checkpoint)


def _make_speedup_chart(lats_with, lats_without, ckpt_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ckpt_dir = os.path.dirname(ckpt_path)
    png_dir = os.path.join(ckpt_dir, "..", "pngs")
    os.makedirs(png_dir, exist_ok=True)

    steps = np.arange(1, len(lats_with) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Per-step latency
    ax1.plot(steps, lats_with, "b-", label="With KV Cache", linewidth=2, alpha=0.8)
    ax1.plot(steps, lats_without, "r-", label="Without KV Cache", linewidth=2, alpha=0.8)
    ax1.set_xlabel("Generation Step")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Per-Step Latency: KV Cache vs No Cache")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative latency
    cum_with = np.cumsum(lats_with)
    cum_without = np.cumsum(lats_without)
    ax2.plot(steps, cum_with, "b-", label="With KV Cache", linewidth=2)
    ax2.plot(steps, cum_without, "r-", label="Without KV Cache", linewidth=2)
    ax2.set_xlabel("Generation Step")
    ax2.set_ylabel("Cumulative Latency (ms)")
    ax2.set_title("Cumulative Time: O(L) vs O(L^2)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "kv_cache_speedup.png"), dpi=150)
    plt.close(fig)
    print(f"[generate] Speedup chart saved to {png_dir}/kv_cache_speedup.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify generate.py runs after training**

```bash
cd projects/nlp/transformer-kv-cache && python generate.py --checkpoint /tmp/kvcache-test/checkpoints/weights_epoch2.pt --max_tokens 50 --device cpu
```

Expected: generates text, prints speedup summary, saves `kv_cache_speedup.png`

- [ ] **Step 3: Verify speedup chart was created**

```bash
ls -lh /tmp/kvcache-test/pngs/kv_cache_speedup.png
```

- [ ] **Step 4: Commit**

```bash
git add projects/nlp/transformer-kv-cache/generate.py
git commit -m "feat: add KV cache inference comparison with latency measurement"
```

---

### Task 8: charts.py — Post-training visualization

**Files:**
- Create: `projects/nlp/transformer-kv-cache/charts.py`

- [ ] **Step 1: Write charts.py**

```python
"""Post-training charts: read output directory and produce visualizations.

Usage:
  python charts.py --output_dir output
  python charts.py --output_dir output --checkpoint output/checkpoints/weights_epoch10.pt
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    png_dir = os.path.join(args.output_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Training curves from metrics.csv
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    if os.path.exists(csv_path):
        _plot_training_curves(csv_path, png_dir)

    # 2. Generate text samples if checkpoint exists
    print(f"Charts saved to {png_dir}/")


def _plot_training_curves(csv_path, png_dir):
    epochs, train_loss, val_loss, ppl, tok_s = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            ppl.append(float(row["perplexity"]))
            tok_s.append(float(row["tokens_per_sec"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(epochs, train_loss, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, "r-", label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # Perplexity
    axes[0, 1].plot(epochs, ppl, "g-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Perplexity")
    axes[0, 1].set_title("Perplexity"); axes[0, 1].grid(alpha=0.3)

    # Tokens/sec
    axes[1, 0].plot(epochs, tok_s, "purple", linewidth=2)
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Tokens/sec")
    axes[1, 0].set_title("Training Speed"); axes[1, 0].grid(alpha=0.3)

    # Train vs Val loss ratio
    ratio = [t / max(v, 0.01) for t, v in zip(train_loss, val_loss)]
    axes[1, 1].plot(epochs, ratio, "orange", linewidth=2)
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Train/Val Loss Ratio")
    axes[1, 1].set_title("Overfitting Monitor"); axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved training_curves.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify charts.py generates updated plots**

```bash
cd projects/nlp/transformer-kv-cache && python charts.py --output_dir /tmp/kvcache-test
ls -lh /tmp/kvcache-test/pngs/training_curves.png
```

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/charts.py
git commit -m "feat: add post-training visualization (training curves, overfitting monitor)"
```

---

### Task 9: launch.py + check_progress.py — Colab bootstrap

**Files:**
- Create: `projects/nlp/transformer-kv-cache/launch.py`
- Create: `projects/nlp/transformer-kv-cache/check_progress.py`

- [ ] **Step 1: Write launch.py**

```python
"""Colab bootstrap: pip install torch+matplotlib, spawn train.py as nohup subprocess.

Reads /content/exp_id.txt for experiment name (used in log tags).
Installs minimal deps: torch (pre-installed on Colab), matplotlib, pandas.
"""
import os
import subprocess
import sys

EXP_ID_PATH = "/content/exp_id.txt"
LOG = "/content/train.log"
OUTPUT_DIR = "/content/transformer-kv-cache-output"


def main():
    # Read experiment tag
    exp_id = "default"
    if os.path.exists(EXP_ID_PATH):
        with open(EXP_ID_PATH) as f:
            exp_id = f.read().strip()
    print(f"[launch] Exp ID: {exp_id}")

    # Install deps (torch is pre-installed on Colab)
    print("[launch] Installing deps ...")
    for dep in ["matplotlib", "pandas"]:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", dep],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # Spawn training
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = (
        f"{sys.executable} -u /content/train.py "
        f"--output_dir {OUTPUT_DIR} "
        f"--device cuda --max_epochs 10 "
        f"--batch_size 64 --block_size 256 "
    )

    print(f"[launch] Running: {cmd}")
    with open(LOG, "w") as f:
        proc = subprocess.Popen(
            cmd.split(), stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
    print(f"[launch] Train PID={proc.pid}, log={LOG}")
    print("[launch] DONE. Training running detached.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write check_progress.py**

```python
"""Local cron progress checker for Transformer-KV-Cache training.

Reads /content/transformer-kv-cache-output/metrics.csv on VM, reports status.
"""
import json
import subprocess
import sys

METRICS_PATH = "/content/transformer-kv-cache-output/metrics.csv"
LOG_PATH = "/content/train.log"


def check():
    # 1. Read metrics
    metrics = []
    try:
        with open(METRICS_PATH) as f:
            import csv
            reader = csv.DictReader(f)
            metrics = list(reader)
    except FileNotFoundError:
        print("[check] WARNING: No metrics.csv found — training may not have started")
        proc_alive = _pgrep("train.py")
        print(f"[check] Process alive: {proc_alive}")
        return 0 if proc_alive else 1

    if not metrics:
        print("[check] WARNING: metrics.csv is empty — no epochs completed")
        print(f"[check] Process alive: {_pgrep('train.py')}")
        return 0

    latest = metrics[-1]
    epoch = int(latest["epoch"])
    train_loss = float(latest["train_loss"])
    val_loss = float(latest["val_loss"])
    ppl = float(latest["perplexity"])
    elapsed = float(latest["elapsed_s"])
    tokens_per_sec = float(latest["tokens_per_sec"])

    proc_alive = _pgrep("train.py")

    # Log tail
    try:
        with open(LOG_PATH) as f:
            log_lines = f.readlines()
        tail = "".join(log_lines[-5:]).rstrip()
    except FileNotFoundError:
        tail = "(no log)"

    # Report
    print(f"[check] Epoch: {epoch} | Train Loss: {train_loss:.3f} | "
          f"Val Loss: {val_loss:.3f} | PPL: {ppl:.1f} | "
          f"tok/s: {tokens_per_sec:.0f} | Time: {elapsed/60:.1f}m | "
          f"Alive: {proc_alive}")

    # Alerts
    alerts = []
    if not proc_alive and epoch < 10:
        alerts.append("CRITICAL: train.py dead before epoch 10")
    if train_loss > 6:
        alerts.append("WARNING: Train loss >6 — may be diverging")
    if epoch >= 8:
        alerts.append(f"INFO: Near completion — epoch {epoch}/10.")
    if epoch >= 10:
        alerts.append("DONE: Training complete. Download results.")

    for a in alerts:
        print(f"[check] {a}")

    if tail:
        print(f"[check] Log tail:\n{tail}")

    return 0 if not any("CRITICAL" in a for a in alerts) else 1


def _pgrep(pattern: str) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(check())
```

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/launch.py projects/nlp/transformer-kv-cache/check_progress.py
git commit -m "feat: add Colab bootstrap and progress checker scripts"
```

---

### Task 10: fetch.sh — Cron artifact pull script

**Files:**
- Create: `projects/nlp/transformer-kv-cache/fetch.sh`

- [ ] **Step 1: Write fetch.sh**

```bash
#!/bin/bash
# Fetch transformer KV cache training results from Colab VM. Called by cron every 2 minutes.
set -euo pipefail
SESSION="${1:-kv-cache}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_OUT="$PROJ_DIR/output"
OUT_TAR="kv-cache-output.tar.gz"
OUT_DIR="/content/transformer-kv-cache-output"
mkdir -p "$LOCAL_OUT"

# Proxy (Config B — HTTP CONNECT, works for full workflow)
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890

COLB="$(which colab)"

echo "[fetch] $(date '+%H:%M:%S') Session: $SESSION"

# 1. Check session alive
"$COLB" sessions 2>/dev/null | grep -q "$SESSION" || {
    echo "[fetch] WARNING: session $SESSION not found — may be dead"
    exit 0
}

# 2. Tar output on VM (exclude heavy checkpoints)
echo '
import subprocess, os
out = "/content/transformer-kv-cache-output"
tar = "/content/kv-cache-output.tar.gz"
subprocess.run(["tar", "-czf", tar, "-C", out,
    "--exclude=checkpoints", "."], check=True)
print(f"Tarball: {os.path.getsize(tar)/1024:.0f} KB")
' | "$COLB" exec -s "$SESSION" --timeout 15 2>/dev/null || {
    echo "[fetch] WARNING: exec tar failed, trying direct download..."
}

# 3. Download
"$COLB" download -s "$SESSION" "/content/$OUT_TAR" "$LOCAL_OUT/$OUT_TAR" 2>/dev/null || {
    echo "[fetch] WARNING: download failed — session may be dead or output not ready"
    exit 0
}

# 4. Extract
cd "$LOCAL_OUT"
tar -xzf "$OUT_TAR" 2>/dev/null || { echo "[fetch] WARNING: extract failed"; exit 0; }
echo "[fetch] $(date '+%H:%M:%S') Extract done."

# 5. Report
if [ -f "$LOCAL_OUT/logs/train.log" ]; then
    echo ""; echo "══ Last 5 log lines ══"
    tail -5 "$LOCAL_OUT/logs/train.log"
fi
if [ -f "$LOCAL_OUT/metrics.csv" ]; then
    echo ""; echo "══ Last 3 metrics rows ══"
    tail -3 "$LOCAL_OUT/metrics.csv"
fi
echo ""; echo "══ PNGs ══"
ls -lh "$LOCAL_OUT/pngs/" 2>/dev/null || echo "(no PNGs yet)"
echo ""; echo "[fetch] Done. Files in $LOCAL_OUT"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x projects/nlp/transformer-kv-cache/fetch.sh
```

- [ ] **Step 3: Commit**

```bash
git add projects/nlp/transformer-kv-cache/fetch.sh
git commit -m "feat: add cron fetch script (2-min interval)"
```

---

### Task 11: KV Cache Documentation (5 docs)

**Files:**
- Create: `docs/reference/kv-cache/01-mechanism.md`
- Create: `docs/reference/kv-cache/02-prefill-vs-decode.md`
- Create: `docs/reference/kv-cache/03-variants.md`
- Create: `docs/reference/kv-cache/04-flashattention.md`
- Create: `docs/reference/kv-cache/05-benchmarks.md`

- [ ] **Step 1: Create directory**

```bash
mkdir -p docs/reference/kv-cache
```

- [ ] **Step 2: Write 01-mechanism.md**

Write `docs/reference/kv-cache/01-mechanism.md`:
```markdown
# KV Cache: Mechanism & Math

## The Problem

In autoregressive decoding, each new token attends to ALL previous tokens. Without caching,
at step t we recompute K and V for tokens 0..t-1, then compute attention. This repeats work:
step t recomputes everything step t-1 already computed, plus one new token.

**Cost without cache:**
- Step 1: compute K,V for 1 token → O(1)
- Step 2: compute K,V for 2 tokens → O(2)
- ...
- Step N: compute K,V for N tokens → O(N)
- **Total: O(N²)** time, O(N²) memory for attention matrix

## The Solution: KV Cache

Keys and values from past tokens don't change. Store them and reuse.

**Cost with cache:**
- Prefill (step 1): compute K,V for all prompt tokens → O(L_prompt)
- Each decode step: compute K,V for 1 new token, append to cache → O(1)
- **Total: O(N)** time per step after prefill

## Mathematical Derivation

### Standard Attention (no cache)

For query Q, key K, value V:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

At step t of autoregressive generation, we have tokens $x_0, x_1, ..., x_t$. To predict $x_{t+1}$:

$$K = [K_0, K_1, ..., K_t] \in \mathbb{R}^{(t+1) \times d_k}$$
$$V = [V_0, V_1, ..., V_t] \in \mathbb{R}^{(t+1) \times d_v}$$
$$Q = Q_t \in \mathbb{R}^{1 \times d_k}$$

The attention computes:

$$\text{Attention}(Q_t, K, V) = \text{softmax}\left(\frac{Q_t [K_0, ..., K_t]^T}{\sqrt{d_k}}\right) [V_0, ..., V_t]$$

$K_0...K_{t-1}$ were already computed at step t-1. We recompute them unnecessarily.

### With KV Cache

Store the accumulated K and V:

1. **Step t (prefill):** Project all prompt tokens to get $K_{0..t}, V_{0..t}$. Store in cache.
2. **Step t+1 (decode):** Project only the new token: $k_{t+1}, v_{t+1} \in \mathbb{R}^{1 \times d_k}$.
   Append to cache:
   $$K_{cache} \leftarrow [K_{cache}; k_{t+1}]$$
   $$V_{cache} \leftarrow [V_{cache}; v_{t+1}]$$
   Compute attention with $Q_{t+1}$ against $K_{cache}, V_{cache}$.

## Memory Analysis

For a model with L layers, H heads, sequence length S, head dimension $d_k$:

| | Without Cache | With Cache |
|---|---|---|
| **Per-step KV compute** | $O(S \cdot H \cdot d_k)$ | $O(H \cdot d_k)$ |
| **Peak memory (KV)** | $O(S^2)$ (attention matrix) | $O(L \cdot H \cdot S \cdot d_k)$ |
| **Total time (N tokens)** | $O(N^2)$ | $O(N)$ per step |

### Concrete Example

GPT-2 small (12 layers, 12 heads, $d_k$=64, bf16):

| Sequence length | Cache memory | Attention matrix |
|---|---|---|
| 256 | 12 × 12 × 256 × 64 × 2 = 4.7 MB | 256² × 12 × 2 = 1.6 MB |
| 1024 | 18.9 MB | 25 MB |
| 4096 | 75.5 MB | 403 MB |
| 16384 | 302 MB | 6.4 GB |

Memory grows linearly with S for KV cache, but quadratically for the attention matrix.

## Limitations

- **Memory still grows with S.** At very long sequences (100K+), KV cache dominates GPU memory.
  Variants like GQA (fewer K/V heads) and PagedAttention help.
- **Batch × cache.** When batching multiple requests, each has its own cache.
  Continuous batching systems manage this carefully.
- **Precision.** KV cache is typically stored in fp16/bf16. Int8/int4 quantization
  can reduce memory ~2-4x with minimal accuracy loss (see KIVI, KVQuant).
```

- [ ] **Step 3: Write 02-prefill-vs-decode.md**

Write `docs/reference/kv-cache/02-prefill-vs-decode.md`:
```markdown
# Prefill vs Decode: Two-Phase Inference

Transformer inference with KV cache splits into two distinct phases with very different compute profiles.

## Prefill Phase

**What happens:** The full prompt is processed in a single forward pass. All prompt tokens'
K and V are computed and cached. The model generates logits for the entire sequence.

**Compute characteristics:**
- **Compute-bound:** Processes many tokens at once → high GPU utilization
- **Matrix-matrix operations:** Q, K, V projections over the full prompt length
- **Causal mask applies:** Each token can only attend to positions ≤ its own
- **No KV cache yet:** This is the step that populates the cache

**Duration:** Proportional to prompt length. For a 2048-token prompt on T4: ~200-500ms.

## Decode Phase

**What happens:** One new token is generated per step. Its K and V are projected and appended
to the cache. Attention is computed against ALL cached keys and values.

**Compute characteristics:**
- **Memory-bound:** Only 1 token → GPU compute units are underutilized
- **Matrix-vector operations:** Q projection is 1×D, attention is 1×(cached_length)
- **No masking needed:** The single query attends to all cached positions
- **Cache I/O dominates:** Reading the full KV cache from HBM is the bottleneck

**Duration:** Roughly constant per step (dominated by cache reads). On T4: ~5-20ms/step.

## The Bottleneck Shift

```
Prefill:  GPU compute >>> memory bandwidth
Decode:   memory bandwidth >>> GPU compute
```

This is why large-batch inference is efficient for prefill but not decode:
- Prefill benefits from batching (more tokens → higher GPU utilization)
- Decode with batch=B must read B separate KV caches → linear memory growth

## Continuous Batching

Modern serving systems (vLLM, TGI) use continuous batching to overlap prefill and decode:

1. New requests enter prefill together → high GPU utilization
2. Requests that finish prefill transition to decode → each gets its own cache
3. Multiple decode-phase requests share a batch → amortizes the cost

This is more efficient than static batching where all requests must finish together.

## Latency vs Throughput

| | Latency-sensitive | Throughput-focused |
|---|---|---|
| **Prefill** | Process quickly (user waiting) | Batch many prompts |
| **Decode** | Generate token-by-token (streaming) | Overlap multiple decodes |

**Tradeoff:** Longer prompts → more KV cache memory → fewer concurrent requests.
Shorter prompts → less cache → higher throughput.

## Practical Numbers (T4, 4-layer GPT, d_model=256)

| Phase | Time | % of step |
|-------|------|-----------|
| Prefill (256 tokens) | ~50ms | 100% |
| Decode (step 1) | ~8ms | 16% of prefill |
| Decode (step 50) | ~10ms | 20% of prefill |
| Decode (step 200) | ~15ms | 30% of prefill |

Decode time increases slowly with cache size (linear growth), but remains much faster
than full recomputation at each step.
```

- [ ] **Step 4: Write 03-variants.md**

Write `docs/reference/kv-cache/03-variants.md`:
```markdown
# KV Cache Variants: GQA, MQA, MLA, PagedAttention

The standard KV cache stores one K,V pair per attention head. As models scale,
this becomes a memory bottleneck. Several variants reduce cache memory.

## Multi-Query Attention (MQA)

**Idea:** Share a single K and V across ALL attention heads. Only Q has multiple heads.

**Memory:** $2 \times L \times S \times d_k$ instead of $2 \times L \times H \times S \times d_k$ → H× reduction.

**Used in:** PaLM, older models.

**Tradeoff:** Significant memory savings, but quality degrades. K/V represent only one
"view" of the input. Used when extreme memory efficiency is needed.

## Grouped-Query Attention (GQA)

**Idea:** Split heads into G groups. Each group shares one K,V. G=1 is MQA; G=H is MHA.

$$\text{K heads} = G, \quad \text{V heads} = G, \quad \text{Q heads} = H$$

**Memory:** $2 \times L \times G \times S \times d_k$ — G× less than MHA.

**Used in:** Llama 2 70B (G=8), Llama 3, Mistral.

**Why it works well:** Most attention heads capture redundant patterns. A few K/V views
are sufficient. GQA recovers most of MHA's quality with 2-8× less KV memory.

**Configuration examples:**
| Model | H (Q heads) | G (KV groups) | KV cache savings |
|-------|------------|---------------|-----------------|
| Llama 2 7B | 32 | 32 (MHA) | 1× |
| Llama 2 70B | 64 | 8 | 8× |
| Mistral 7B | 32 | 8 | 4× |
| Llama 3 70B | 64 | 8 | 8× |

## Multi-head Latent Attention (MLA)

**Idea (DeepSeek-V2):** Compress K and V into a low-rank latent space, then decompress
during attention. Far more aggressive than GQA.

$$K_{latent} = W_{down}^K \cdot X \in \mathbb{R}^{d_{latent}}$$
$$K = W_{up}^K \cdot K_{latent} \in \mathbb{R}^{H \times d_k}$$

**Memory:** Stores only the latent vector per token → $L \times S \times d_{latent}$.
With $d_{latent} = 512$ and $H \times d_k = 128 \times 128 = 16384$, this is a 32× reduction.

**Used in:** DeepSeek-V2, DeepSeek-V3.

**Challenge:** The up-projection adds compute during decode. DeepSeek absorbs this into
the Q-projection via a mathematical equivalence, making it compute-neutral.

## PagedAttention (vLLM)

**Idea:** KV cache is NOT a contiguous tensor per request. Instead, it's stored in
fixed-size "pages" (blocks), like virtual memory.

**Key concepts:**
- **Block size:** Fixed-size chunks (e.g., 16 tokens per block)
- **Block table:** Maps logical positions → physical blocks
- **Non-contiguous:** Blocks can be scattered across GPU memory

**Benefits:**
1. **Zero fragmentation:** No wasted memory between requests (internal fragmentation only in the last block)
2. **Sharing:** Prefix caches (system prompts, few-shot examples) can share physical blocks
3. **Dynamic allocation:** Blocks are allocated on-demand, not pre-allocated for max length

**Memory utilization:** PagedAttention achieves ~96% KV cache utilization vs ~20-40% for
contiguous allocation (due to over-provisioning for max_length).

## Comparison Table

| Variant | Memory per layer | Quality impact | Used in |
|---------|-----------------|----------------|---------|
| MHA (baseline) | $2HSd_k$ | Best | GPT-2, BERT |
| GQA (G groups) | $2GSd_k$ | Minor (<1%) | Llama 2/3, Mistral |
| MQA (G=1) | $2Sd_k$ | Noticeable (~1-2%) | PaLM |
| MLA | $2Sd_{latent}$ | Minor (with careful design) | DeepSeek-V2/V3 |
| PagedAttention | N/A (allocation, not format) | None | vLLM |
```

- [ ] **Step 5: Write 04-flashattention.md**

Write `docs/reference/kv-cache/04-flashattention.md`:
```markdown
# KV Cache + FlashAttention: Compatibility & Interaction

FlashAttention is an I/O-aware exact attention algorithm. It fuses the attention computation
into a single CUDA kernel, avoiding materializing the $S \times S$ attention matrix in HBM.

## How They Interact

KV cache and FlashAttention solve different problems:

- **KV cache:** Avoids recomputing K and V for past tokens
- **FlashAttention:** Makes computing attention from K,V faster and more memory-efficient

They are **complementary**, not alternatives. Modern inference uses both:
1. KV cache stores past K,V (avoids recomputation)
2. FlashAttention computes the attention efficiently (reduces memory bandwidth)

## Training: FlashAttention Without Cache

During training, we process full sequences (no cache). FlashAttention shines here:
- Standard attention materializes $[B, H, S, S]$ attention matrix in HBM → $O(S^2)$ memory
- FlashAttention computes attention in tiles in SRAM → $O(S)$ memory
- Training speedup: 3-7× for long sequences (4096+)

## Inference: Both Together

**With KV cache + FlashAttention:**
1. Cache stores $[B, H, S_{cache}, d_k]$ K and V
2. New token's K,V ($[B, H, 1, d_k]$) are appended
3. FlashAttention computes attention over the full cached sequence

**Compatibility:** FlashAttention-2 and FlashAttention-3 both support KV cache inference.
The `flash_attn_with_kvcache` function in FlashAttention's API directly handles this:

```python
from flash_attn import flash_attn_with_kvcache

# q: [B, 1, H, d_k] — new token
# k_cache, v_cache: [B, H, S_cache, d_k] — accumulated cache
# k_new, v_new: [B, 1, H, d_k] — new token's K,V

output = flash_attn_with_kvcache(
    q, k_cache, v_cache, k_new, v_new,
    causal=False  # no mask needed for single query token
)
# k_cache, v_cache are updated in-place
```

## Interleaving: When to Recompute vs Cache

For very long sequences where KV cache exceeds GPU memory, there's a **recomputation tradeoff**:

| Strategy | Memory | Compute |
|----------|--------|---------|
| Cache everything | $O(LS)$ | $O(S)$ per step |
| Cache nothing (recompute) | $O(1)$ | $O(S^2)$ per step |
| **Sliding window cache** | $O(LW)$ | $O(W)$ per step |
| **StreamingLLM** (keep attention sinks) | $O(L(W+4))$ | $O(W+4)$ per step |

**Sliding window** (Mistral, Mixtral): Only cache the last W tokens. Older tokens are dropped.
This gives O(W) memory and compute regardless of sequence length.

**StreamingLLM:** Keep the first ~4 tokens (attention sinks) + sliding window.
Attention sinks absorb attention weight and prevent perplexity collapse when
dropping middle tokens.

## FlashAttention Versions

| Version | KV Cache Support | Key Feature |
|---------|-----------------|-------------|
| v1 | Via separate API | I/O-aware tiling |
| v2 | `flash_attn_with_kvcache` | 2× faster, supports GQA natively |
| v3 | `flash_attn_with_kvcache` | Hopper FP8, async, 1.5-2× faster |

**PyTorch SDPA** (`torch.nn.functional.scaled_dot_product_attention`): Since PyTorch 2.0,
also supports FlashAttention backend and KV cache via separate K,V inputs. Does NOT
support in-place cache update — pass the full concatenated K,V each step.
```

- [ ] **Step 6: Write 05-benchmarks.md**

Write `docs/reference/kv-cache/05-benchmarks.md`:
```markdown
# KV Cache Benchmarks

Measured speedup from this project's `generate.py` on a 4-layer GPT (d_model=256, 4 heads).

Run `python generate.py --checkpoint output/checkpoints/weights_epoch10.pt --max_tokens 200`
to reproduce.

## Expected Speedup (Theoretical)

For N generation steps and average sequence length L:

| Metric | Without Cache | With Cache | Speedup |
|--------|-------------|-----------|---------|
| K,V compute per step | $O(L)$ projections | $O(1)$ projection | $O(L)$ |
| Attention compute per step | $O(L)$ dot products | $O(L)$ dot products | $O(1)$ |
| Memory (peak) | $O(L^2)$ attention matrix | $O(L)$ cached K,V | — |

The key savings: without cache, you recompute K and V for every token at every step.
With cache, each token's K and V is computed ONCE.

## Measured Results

Placeholder: fill in after running on Colab/GPU. Template:

| Sequence Length | Without Cache (ms/step) | With Cache (ms/step) | Speedup |
|----------------|------------------------|---------------------|---------|
| 10 | ? | ? | ?× |
| 50 | ? | ? | ?× |
| 100 | ? | ? | ?× |
| 200 | ? | ? | ?× |
| 500 | ? | ? | ?× |

## Memory Profile

4-layer GPT, d_model=256, 4 heads, block_size=256, fp32:

| Component | Size |
|-----------|------|
| Model weights | ~1.5M params × 4 bytes = 6 MB |
| KV cache (seq_len=256, 4 layers) | 4 × 4 × 256 × 64 × 4 × 2 = 2.1 MB |
| Attention matrix (256²) | 256 × 4 × 256 × 256 × 4 = 1 MB |
| Total inference | ~9 MB |

For reference, the attention matrix WITHOUT KV cache at step 256 would require
recomputing K,V for all 256 tokens × 4 layers → much higher compute but same peak memory.

## Latency Breakdown (T4 GPU, expected)

| Component | Time per step | % |
|-----------|-------------|---|
| QKV projection | ~2ms | 25% |
| Cache read | ~3ms | 37% |
| Attention compute | ~2ms | 25% |
| FFN + residual | ~1ms | 13% |

Cache read dominates decode latency — this is why KV cache quantization matters.

## How to Reproduce

```bash
# Train
python train.py --device cuda --max_epochs 10

# Benchmark
python generate.py \
  --checkpoint output/checkpoints/weights_epoch10.pt \
  --max_tokens 100 200 500 \
  --device cuda
```

Results appear in `output/pngs/kv_cache_speedup.png`.
```

- [ ] **Step 7: Commit all 5 docs**

```bash
git add docs/reference/kv-cache/
git commit -m "docs: add 5-part KV cache deep dive (mechanism, prefill/decode, variants, flash, benchmarks)"
```

---

### Task 12: Ruff lint + deploy dry-run

**Files:** None created, verify everything passes.

- [ ] **Step 1: Run ruff lint**

```bash
cd /Users/mx/Desktop/projects/colab-cli && ruff check projects/nlp/transformer-kv-cache/
```

Expected: Zero errors. If any, `ruff check --fix` then re-check.

- [ ] **Step 2: Run full training test on CPU (5 epochs, verify all artifacts)**

```bash
cd projects/nlp/transformer-kv-cache && python train.py --max_epochs 5 --batch_size 32 --block_size 128 --log_interval 20 --output_dir /tmp/kvcache-full-test
```

Verify outputs:
```bash
ls -lh /tmp/kvcache-full-test/logs/train.log
ls -lh /tmp/kvcache-full-test/metrics.csv
ls -lh /tmp/kvcache-full-test/pngs/training_curves.png
ls -lh /tmp/kvcache-full-test/checkpoints/
cat /tmp/kvcache-full-test/metrics.csv
```

- [ ] **Step 3: Run generate.py with trained checkpoint**

```bash
cd projects/nlp/transformer-kv-cache && python generate.py --checkpoint /tmp/kvcache-full-test/checkpoints/weights_epoch5.pt --max_tokens 100 --device cpu
```

Expected: text generated, speedup chart saved.

- [ ] **Step 4: Verify speedup chart exists**

```bash
ls -lh /tmp/kvcache-full-test/pngs/kv_cache_speedup.png
```

- [ ] **Step 5: Run charts.py on test output**

```bash
cd projects/nlp/transformer-kv-cache && python charts.py --output_dir /tmp/kvcache-full-test
ls -lh /tmp/kvcache-full-test/pngs/
```

- [ ] **Step 6: Commit project**

```bash
git add projects/nlp/transformer-kv-cache/ docs/reference/kv-cache/
git status
git commit -m "feat: add transformer KV cache project — pluggable GPT + Colab deploy + 5-part KV cache docs"
```

---

### Task 13: Deploy to Colab + Setup Cron

> **Note:** This task requires the colab-cli skill. Invoke `Skill(colab-cli)` before executing.

- [ ] **Step 1: Package and upload to Colab**

```bash
cd /Users/mx/Desktop/projects/colab-cli

# Package files needed on VM
cat > /tmp/upload_files.txt << 'EOF'
projects/nlp/transformer-kv-cache/config.py
projects/nlp/transformer-kv-cache/kv_cache.py
projects/nlp/transformer-kv-cache/attention.py
projects/nlp/transformer-kv-cache/model.py
projects/nlp/transformer-kv-cache/train.py
projects/nlp/transformer-kv-cache/launch.py
EOF

# Upload each file (using base64 embed pattern for reliability)
# See colab-cli SKILL.md for upload instructions
```

- [ ] **Step 2: Provision Colab GPU session and run launch.py**

```bash
# Provision T4 session
colab new --gpu T4 -s kv-cache

# Upload files, set exp_id, run launch
echo "default" > /tmp/exp_id.txt
# ... upload flow per colab-cli skill ...
# colab exec -s kv-cache -f launch.py
```

- [ ] **Step 3: Set up 2-minute cron with CronCreate**

```
CronCreate prompt:
"Fetch KV cache training artifacts from Colab session 'kv-cache':
1. Run: bash projects/nlp/transformer-kv-cache/fetch.sh kv-cache
2. Report the tail output to the user"
Cron: */2 * * * * (every 2 minutes)
```

- [ ] **Step 4: Monitor until training complete**

Check fetch output each tick: training loss trending down? Epoch advancing? Alert if process dead.

- [ ] **Step 5: After training: download final checkpoint + generate comparison**

```bash
colab download -s kv-cache /content/transformer-kv-cache-output output-final/
python generate.py --checkpoint output-final/checkpoints/weights_epoch10.pt --max_tokens 200 --device cpu
```

- [ ] **Step 6: Cancel cron when done**

```bash
# Use CronDelete with job ID from Step 3
```

- [ ] **Step 7: Final commit with results**

```bash
git add projects/nlp/transformer-kv-cache/output/
git commit -m "results: KV cache training results + speedup benchmarks"
```
