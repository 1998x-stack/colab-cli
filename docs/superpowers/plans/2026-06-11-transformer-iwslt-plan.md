# Transformer IWSLT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full Transformer (Vaswani et al. 2017) in PyTorch, train 3 ablation experiments on IWSLT'14 De→En across 3 Colab GPU accounts in parallel with checkpoint-resume.

**Architecture:** Six files: `model.py` (pure architecture, ~65M params), `train.py` (data pipeline, training loop, beam search eval, experiment config), `checkpoint.py` (save/load helpers), `launch.py` (Colab bootstrap + detached spawn), `check_progress.py` (local cron monitoring), `charts.py` (post-hoc chart generation from downloaded metrics.jsonl).

**Tech Stack:** PyTorch 2.11.0+cu128, `tokenizers` (HuggingFace BPE), `sacrebleu`, `matplotlib`

---

### Task 1: model.py — Transformer Architecture

**Files:**
- Create: `projects/transformer_iwslt/model.py`

- [ ] **Step 1: Write model.py with full Transformer**

```python
"""Transformer from "Attention Is All You Need" (Vaswani et al. 2017).

Paper base model: d_model=512, 6 encoder + 6 decoder layers, 8 heads, ~65M params.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding (paper Eq. 3-5).

    Returns (1, max_len, d_model) tensor.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All (batch, seq_len, d_model). mask: (batch, 1, seq_len, seq_len) or broadcastable, True = ignore."""
        B = query.size(0)

        Q = self.W_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
        d_ff: int, max_len: int, dropout: float = 0.1, use_learned_pe: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if use_learned_pe:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            self.register_buffer("pe", sinusoidal_pe(max_len, d_model))
        self.use_learned_pe = use_learned_pe
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
        d_ff: int, max_len: int, dropout: float = 0.1, use_learned_pe: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if use_learned_pe:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            self.register_buffer("pe", sinusoidal_pe(max_len, d_model))
        self.use_learned_pe = use_learned_pe
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embed(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        use_learned_pe: bool = True,
        share_embeddings: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_model, n_enc_layers, n_heads, d_ff, max_len,
            dropout, use_learned_pe,
        )
        self.decoder = Decoder(
            vocab_size, d_model, n_dec_layers, n_heads, d_ff, max_len,
            dropout, use_learned_pe,
        )
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        if share_embeddings:
            self.out_proj.weight = self.decoder.embed.weight  # tie all three
            self.encoder.embed.weight = self.decoder.embed.weight  # tie input embeddings
        self.d_model = d_model

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.out_proj(dec_out)

    @staticmethod
    def create_padding_mask(pad_idx: int, x: torch.Tensor) -> torch.Tensor:
        """(batch, 1, 1, seq_len) — True where pad token."""
        return (x == pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """(1, 1, seq_len, seq_len) — True for positions > current."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)


def build_transformer(exp_id: str, vocab_size: int = 32000) -> Transformer:
    """Factory that returns a Transformer configured per experiment."""
    base = dict(vocab_size=vocab_size, d_model=512, n_enc_layers=6, n_dec_layers=6,
                n_heads=8, d_ff=2048, max_len=512, dropout=0.1)

    if exp_id == "baseline":
        return Transformer(**base, use_learned_pe=True)
    elif exp_id == "fixed_pe":
        return Transformer(**base, use_learned_pe=False)
    elif exp_id == "heads_1":
        return Transformer(**{**base, "n_heads": 1})
    else:
        raise ValueError(f"Unknown exp_id: {exp_id}")
```

- [ ] **Step 2: Verify model forward pass locally**

```bash
cd projects/transformer_iwslt && python -c "
import torch
from model import build_transformer, Transformer

# Test all 3 experiment configs
for exp in ['baseline', 'fixed_pe', 'heads_1']:
    m = build_transformer(exp, vocab_size=1000)
    src = torch.randint(0, 1000, (2, 50))
    tgt = torch.randint(0, 1000, (2, 40))
    src_mask = Transformer.create_padding_mask(0, src)
    tgt_mask = Transformer.create_padding_mask(0, tgt[:, :-1]) | Transformer.create_causal_mask(39, src.device)
    out = m(src, tgt[:, :-1], src_mask, tgt_mask)
    assert out.shape == (2, 39, 1000), f'{exp}: expected (2,39,1000), got {out.shape}'
    loss = torch.nn.functional.cross_entropy(out.transpose(1,2), tgt[:,1:], ignore_index=0)
    loss.backward()
    params = sum(p.numel() for p in m.parameters())
    print(f'{exp}: loss={loss.item():.4f}, params={params:,}')

print('All configs pass')
"
```

Expected: All 3 configs forward+backward without error. Parameter counts: baseline ~53M (shared embeddings), heads_1 ~53M.

- [ ] **Step 3: Commit**

```bash
git add projects/transformer_iwslt/model.py
git commit -m "feat: add Transformer model (base, fixed PE, 1-head configs)"
```

---

### Task 2: checkpoint.py — Checkpoint Save/Load

**Files:**
- Create: `projects/transformer_iwslt/checkpoint.py`

- [ ] **Step 1: Write checkpoint.py**

```python
"""Checkpoint save/load helpers for training resume across Colab sessions."""
import torch
import os


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    epoch: int,
    train_loss: float,
    val_loss: float,
    bleu: float,
    tokens_processed: int,
    wall_time_s: float,
    config: dict,
):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "bleu": bleu,
        "tokens_processed": tokens_processed,
        "wall_time_s": wall_time_s,
        "config": config,
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device):
    """Returns (optimizer_state, scheduler_state, epoch, metrics_dict, config). Caller restores optimizer."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return (
        ckpt["optimizer_state"],
        ckpt.get("scheduler_state"),
        ckpt["epoch"],
        {
            "train_loss": ckpt.get("train_loss", float("inf")),
            "val_loss": ckpt.get("val_loss", float("inf")),
            "bleu": ckpt.get("bleu", 0.0),
            "tokens_processed": ckpt.get("tokens_processed", 0),
            "wall_time_s": ckpt.get("wall_time_s", 0.0),
        },
        ckpt.get("config", {}),
    )


def ensure_checkpoint_dir(base: str = "/content") -> str:
    path = os.path.join(base, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path
```

- [ ] **Step 2: Verify roundtrip locally**

```bash
cd projects/transformer_iwslt && python -c "
import torch, tempfile, os
from model import build_transformer
from checkpoint import save_checkpoint, load_checkpoint

m = build_transformer('baseline', vocab_size=1000)
opt = torch.optim.Adam(m.parameters(), lr=0.0001)
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    tmp = f.name
save_checkpoint(tmp, m, opt, None, 5, 2.3, 2.1, 22.5, 16000000, 1200.0, {'exp_id': 'baseline'})

# Load into fresh model
m2 = build_transformer('baseline', vocab_size=1000)
opt_state, sched_state, epoch, metrics, config = load_checkpoint(tmp, m2, torch.device('cpu'))
opt2 = torch.optim.Adam(m2.parameters(), lr=0.0001)
opt2.load_state_dict(opt_state)

assert epoch == 5
assert metrics['bleu'] == 22.5
assert config['exp_id'] == 'baseline'
# Verify weights match
for p1, p2 in zip(m.parameters(), m2.parameters()):
    assert torch.equal(p1, p2)
os.unlink(tmp)
print('Checkpoint roundtrip OK')
"
```

Expected: "Checkpoint roundtrip OK"

- [ ] **Step 3: Commit**

```bash
git add projects/transformer_iwslt/checkpoint.py
git commit -m "feat: add checkpoint save/load for training resume"
```

---

### Task 3: train.py — Data Pipeline, Training Loop, Evaluation

**Files:**
- Create: `projects/transformer_iwslt/train.py`

- [ ] **Step 1: Write train.py — part 1: data loading and tokenizer**

```python
"""Training loop for Transformer on IWSLT'14 De->En.

Usage:
    python train.py --exp_id baseline
    python train.py --exp_id baseline --resume /content/checkpoints/ckpt_epoch5.pt
"""
import argparse, json, math, os, sys, time, gzip, io, tarfile, urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import sacrebleu

from model import build_transformer, Transformer
from checkpoint import save_checkpoint, load_checkpoint, ensure_checkpoint_dir


# --- IWSLT Data Loading ---

IWSLT_URL = "https://wit3.fbk.eu/archive/2014-01//texts/de/en/de-en.tgz"


def download_iwslt(data_dir: str) -> tuple[str, str]:
    """Download and extract IWSLT'14 De-En. Returns (de_path, en_path)."""
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "de-en.tgz")

    if not os.path.exists(tgz_path):
        print(f"[data] Downloading IWSLT'14 De-En from {IWSLT_URL} ...")
        urllib.request.urlretrieve(IWSLT_URL, tgz_path)

    # Extract .de and .en files from the tgz
    de_path = os.path.join(data_dir, "train.de")
    en_path = os.path.join(data_dir, "train.en")

    if not os.path.exists(de_path) or not os.path.exists(en_path):
        print("[data] Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            # Find train.de and train.en in the archive
            de_member = None
            en_member = None
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if "train.tags.de-en.de" in member.name or (name.endswith(".de") and "train" in name):
                    de_member = member
                elif "train.tags.de-en.en" in member.name or (name.endswith(".en") and "train" in name):
                    en_member = member
            if de_member is None or en_member is None:
                # Fallback: list contents and find matching files
                all_names = [m.name for m in tar.getmembers()]
                print(f"[data] Archive contents: {all_names[:20]}")
                raise RuntimeError("Could not find train.de/train.en in archive")

            de_member.name = "train.de"
            en_member.name = "train.en"
            tar.extract(de_member, data_dir)
            tar.extract(en_member, data_dir)

            # Remove XML tags if present
            _clean_tags(de_path)
            _clean_tags(en_path)

    return de_path, en_path


def _clean_tags(path: str):
    """Remove XML tags like <url>, <keywords>, etc. from IWSLT files."""
    import re
    with open(path) as f:
        lines = f.readlines()
    tag_re = re.compile(r"<[^>]+>")
    cleaned = []
    for line in lines:
        line = tag_re.sub("", line).strip()
        if line:
            cleaned.append(line)
    with open(path, "w") as f:
        f.write("\n".join(cleaned))


def load_sentence_pairs(de_path: str, en_path: str) -> list[tuple[str, str]]:
    with open(de_path) as df, open(en_path) as ef:
        de_lines = [l.strip() for l in df if l.strip()]
        en_lines = [l.strip() for l in ef if l.strip()]
    assert len(de_lines) == len(en_lines), f"Mismatch: {len(de_lines)} de vs {len(en_lines)} en"
    return list(zip(de_lines, en_lines))


# --- Tokenizer ---

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]


def train_tokenizer(
    pairs: list[tuple[str, str]], vocab_size: int = 32000, save_path: str = "/content/tokenizer.json"
) -> Tokenizer:
    """Train a shared BPE tokenizer on concatenated source + target sentences."""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    # Train on both languages together for shared vocab
    all_text = [de for de, _ in pairs] + [en for _, en in pairs]
    tokenizer.train_from_iterator(all_text, trainer)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        tokenizer.save(save_path)

    return tokenizer
```

- [ ] **Step 2: Write train.py — part 2: dataset, LR scheduler, training loop**

```python
# --- Dataset ---

class TranslationDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_len: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        de, en = self.pairs[idx]
        src_ids = [SOS] + self.tokenizer.encode(de).ids[:self.max_len - 2] + [EOS]
        tgt_ids = [SOS] + self.tokenizer.encode(en).ids[:self.max_len - 2] + [EOS]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch: list, pad_idx: int = PAD) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences in batch to max length."""
    src_list, tgt_list = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# --- LR Scheduler (paper §5.3) ---

class NoamScheduler:
    """lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))"""
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._rate = 0.0

    def step(self):
        self._step += 1
        rate = self._compute_rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        arg1 = self._step ** (-0.5)
        arg2 = self._step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

    def state_dict(self):
        return {"step": self._step, "rate": self._rate}

    def load_state_dict(self, state: dict):
        self._step = state["step"]
        self._rate = state["rate"]


# --- Beam Search ---

@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    beam_size: int,
    eos_idx: int,
    device: torch.device,
) -> list[int]:
    """Greedy decode (beam=1) or beam search. Returns token list."""
    model.eval()
    enc_out = model.encoder(src, src_mask)  # (1, src_len, d_model)

    if beam_size == 1:
        return _greedy_decode(model, enc_out, src_mask, max_len, eos_idx, device)

    # Beam search
    B = beam_size
    # (1, 1, beam, d_model)
    enc_out_b = enc_out.unsqueeze(2).expand(-1, -1, B, -1)

    # Each beam: (seq_len,)
    sequences = torch.full((B, 1), SOS, dtype=torch.long, device=device)
    scores = torch.zeros(B, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        # Decode all beams in parallel
        tgt = sequences  # (B, seq_len)
        tgt_mask = (Transformer.create_padding_mask(PAD, tgt) |
                    Transformer.create_causal_mask(tgt.size(1), device))

        dec_out = model.decoder(tgt, enc_out_b[:, :, 0, :], src_mask, tgt_mask)
        logits = model.out_proj(dec_out[:, -1, :])  # (B, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # Expand: (B, vocab) candidates
        cand_scores = scores.unsqueeze(1) + log_probs  # (B, vocab)
        cand_scores = cand_scores.view(-1)  # flatten
        top_scores, top_idx = torch.topk(cand_scores, B)

        # Decode beam and token index
        beam_idx = top_idx // log_probs.size(1)
        token_idx = top_idx % log_probs.size(1)

        # Build new sequences
        new_sequences = torch.zeros(B, step + 2, dtype=torch.long, device=device)
        new_scores = torch.zeros(B, device=device)
        new_finished = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(B):
            src_beam = beam_idx[i]
            new_sequences[i, :step+1] = sequences[src_beam]
            new_sequences[i, step+1] = token_idx[i]
            new_scores[i] = top_scores[i]
            new_finished[i] = finished[src_beam] | (token_idx[i] == eos_idx)

        sequences = new_sequences
        scores = new_scores
        finished = new_finished

        if finished.all():
            break

    # Return best finished (or best unfinished)
    best_idx = scores.argmax().item()
    tokens = sequences[best_idx].tolist()
    # Truncate at first EOS
    if eos_idx in tokens:
        tokens = tokens[:tokens.index(eos_idx)]
    return tokens


def _greedy_decode(model, enc_out, src_mask, max_len, eos_idx, device):
    """Beam=1 fast path."""
    tokens = [SOS]
    for _ in range(max_len - 1):
        tgt = torch.tensor([tokens], device=device)
        tgt_mask = Transformer.create_causal_mask(len(tokens), device)
        dec_out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits = model.out_proj(dec_out[:, -1, :])
        next_tok = logits.argmax(-1).item()
        tokens.append(next_tok)
        if next_tok == eos_idx:
            break
    return tokens


# --- BLEU Evaluation ---

@torch.no_grad()
def evaluate(
    model: Transformer,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 128,
) -> float:
    """Compute sacreBLEU on validation set."""
    model.eval()
    hypotheses = []
    references = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask = Transformer.create_padding_mask(PAD, src)

        for i in range(src.size(0)):
            pred_tokens = beam_search(
                model, src[i:i+1], src_mask[i:i+1], max_len, beam_size, EOS, device
            )
            hyp = tokenizer.decode(pred_tokens[1:])  # skip SOS
            ref = tokenizer.decode([t for t in tgt[i].tolist() if t not in (PAD, SOS, EOS)])
            hypotheses.append(hyp)
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score
```

- [ ] **Step 3: Write train.py — part 3: main training function**

```python
# --- Main Training ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", required=True, choices=["baseline", "fixed_pe", "heads_1"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--data_dir", default="/content/iwslt_data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--output_dir", default="/content")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}, Exp: {args.exp_id}")

    # --- Data ---
    de_path, en_path = download_iwslt(args.data_dir)
    pairs = load_sentence_pairs(de_path, en_path)

    # Train tokenizer or load cached
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer as Tok
        tokenizer = Tok.from_file(tok_path)
        print(f"[train] Loaded tokenizer from {tok_path}")
    else:
        tokenizer = train_tokenizer(pairs, save_path=tok_path)
        print(f"[train] Trained tokenizer, vocab={tokenizer.get_vocab_size()}")

    vocab_size = tokenizer.get_vocab_size()
    print(f"[train] Vocab size: {vocab_size}, Pairs: {len(pairs)}")

    # Train/val split
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = TranslationDataset(train_pairs, tokenizer, args.max_len)
    val_ds = TranslationDataset(val_pairs, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # --- Model ---
    model = build_transformer(args.exp_id, vocab_size).to(device)
    print(f"[train] Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=model.d_model, warmup_steps=4000)

    # Resume state
    start_epoch = 0
    tokens_processed = 0
    wall_time_s = 0.0
    metrics_file = os.path.join(args.output_dir, "metrics.jsonl")

    if args.resume:
        opt_state, sched_state, start_epoch, prev_metrics, _ = load_checkpoint(
            args.resume, model, device
        )
        optimizer.load_state_dict(opt_state)
        if sched_state:
            scheduler.load_state_dict(sched_state)
        tokens_processed = prev_metrics["tokens_processed"]
        wall_time_s = prev_metrics["wall_time_s"]
        print(f"[train] Resumed from epoch {start_epoch} (loss={prev_metrics['train_loss']:.3f}, bleu={prev_metrics['bleu']:.1f})")

    # Continue metrics file if resuming
    if not args.resume:
        with open(metrics_file, "w") as f:
            pass  # create empty

    # --- Config for reproducibility ---
    config = {
        "exp_id": args.exp_id, "vocab_size": vocab_size, "d_model": model.d_model,
        "n_heads": model.encoder.layers[0].self_attn.n_heads,
        "batch_size": args.batch_size, "max_len": args.max_len,
        "beam_size": args.beam_size, "train_pairs": len(train_pairs),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    ckpt_dir = ensure_checkpoint_dir(args.output_dir)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.0)
    t0 = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing: predict tgt[1:] from tgt[:-1]
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = Transformer.create_padding_mask(PAD, src)
            tgt_mask = (Transformer.create_padding_mask(PAD, tgt_in) |
                        Transformer.create_causal_mask(tgt_in.size(1), device))

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            tokens_processed += src.numel() + tgt.numel()

        train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        val_loss = _compute_val_loss(model, val_loader, criterion, device)
        bleu = evaluate(model, val_loader, tokenizer, device, beam_size=args.beam_size, max_len=args.max_len)

        wall_time_s += time.time() - t0
        t0 = time.time()  # reset for next epoch timing (approximate)

        # --- Log ---
        lr = optimizer.param_groups[0]["lr"]
        metrics = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "bleu": round(bleu, 1),
            "lr": round(lr, 8), "tokens_processed": tokens_processed,
            "wall_time_s": round(wall_time_s, 1),
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        print(f"[train] Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.3f} | val_loss={val_loss:.3f} | "
              f"BLEU={bleu:.1f} | lr={lr:.6f} | time={wall_time_s/60:.1f}m")

        # --- Checkpoint ---
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch,
                        train_loss, val_loss, bleu, tokens_processed, wall_time_s, config)
        # Remove older checkpoint to save space (keep last 2)
        if epoch > 2:
            old = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch-2}.pt")
            if os.path.exists(old):
                os.remove(old)

        t0 = time.time()

    print(f"[train] Done. Total time: {wall_time_s/60:.1f}m")
    return 0


@torch.no_grad()
def _compute_val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = Transformer.create_padding_mask(PAD, src)
        tgt_mask = (Transformer.create_padding_mask(PAD, tgt_in) |
                    Transformer.create_causal_mask(tgt_in.size(1), device))
        logits = model(src, tgt_in, src_mask, tgt_mask)
        total += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
        n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify train.py parses args and model runs locally on CPU**

```bash
cd projects/transformer_iwslt && python -c "
from train import main
# We can't run full main() without data, but verify imports and module structure
print('train.py imports OK')
"
```

Expected: "train.py imports OK"

- [ ] **Step 5: Commit**

```bash
git add projects/transformer_iwslt/train.py
git commit -m "feat: add training loop with IWSLT data, BPE tokenizer, beam search, checkpoint-resume"
```

---

### Task 4: launch.py — Colab Bootstrap

**Files:**
- Create: `projects/transformer_iwslt/launch.py`

- [ ] **Step 1: Write launch.py**

```python
"""Colab bootstrap: pip install deps, spawn train.py as detached subprocess.

Reads /content/exp_id.txt for experiment config.
Supports --resume flag via /content/resume_path.txt if checkpoint exists.
"""
import subprocess, sys, os

EXP_ID_PATH = "/content/exp_id.txt"
RESUME_PATH_FILE = "/content/resume_path.txt"
LOG = "/content/train.log"
DEPS = ["tokenizers", "sacrebleu", "matplotlib"]

# --- Read experiment ID ---
with open(EXP_ID_PATH) as f:
    exp_id = f.read().strip()
print(f"[launch] Exp ID: {exp_id}")

# --- Check for resume checkpoint ---
resume_flag = ""
if os.path.exists(RESUME_PATH_FILE):
    with open(RESUME_PATH_FILE) as f:
        ckpt_path = f.read().strip()
    if ckpt_path and os.path.exists(ckpt_path):
        resume_flag = f"--resume {ckpt_path}"
        resume_epoch = os.path.basename(ckpt_path).replace("checkpoint_epoch", "").replace(".pt", "")
        print(f"[launch] Resuming from epoch {resume_epoch}: {ckpt_path}")
    else:
        print(f"[launch] Resume path file exists but checkpoint not found at '{ckpt_path}' — starting fresh")
else:
    print("[launch] No resume checkpoint — starting fresh")

# --- Install deps ---
print("[launch] Installing dependencies...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q"] + DEPS,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("[launch] Dependencies installed")

# --- Spawn training ---
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

cmd = f"{sys.executable} -u /content/train.py --exp_id {exp_id}"
if resume_flag:
    cmd += f" {resume_flag}"

print(f"[launch] Running: {cmd}")
with open(LOG, "w") as f:
    proc = subprocess.Popen(
        cmd.split(),
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
print(f"[launch] Train PID={proc.pid}, log={LOG}")
print(f"[launch] DONE. Training running detached.")
```

- [ ] **Step 2: Verify launch.py syntax**

```bash
python -m py_compile projects/transformer_iwslt/launch.py && echo "OK"
```

Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add projects/transformer_iwslt/launch.py
git commit -m "feat: add Colab bootstrap launcher with checkpoint-resume support"
```

---

### Task 5: check_progress.py — Local Cron Monitoring

**Files:**
- Create: `projects/transformer_iwslt/check_progress.py`

- [ ] **Step 1: Write check_progress.py**

```python
"""Local cron progress checker for Transformer training.

Reads /content/metrics.jsonl on VM, reports status, flags alerts.
"""
import json, os, subprocess, sys


METRICS_PATH = "/content/metrics.jsonl"
LOG_PATH = "/content/train.log"


def check():
    # 1. Read metrics.jsonl for latest epoch
    try:
        with open(METRICS_PATH) as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print("[check] WARNING: No metrics.jsonl found — training may not have started")
        return 1

    if not lines:
        print("[check] WARNING: metrics.jsonl is empty — no epochs completed yet")
        # Check if process is at least alive
        proc_alive = _pgrep("train.py")
        print(f"[check] Process alive: {proc_alive}")
        return 0 if proc_alive else 1

    latest = json.loads(lines[-1])
    epoch = latest.get("epoch", 0)
    train_loss = latest.get("train_loss", float("inf"))
    val_loss = latest.get("val_loss", float("inf"))
    bleu = latest.get("bleu", 0.0)
    lr = latest.get("lr", 0.0)
    wall_time = latest.get("wall_time_s", 0)

    # 2. Process alive check
    proc_alive = _pgrep("train.py")

    # 3. Log tail
    try:
        with open(LOG_PATH) as f:
            log_lines = f.readlines()
        tail = "".join(log_lines[-5:]).rstrip()
    except FileNotFoundError:
        tail = "(no log file)"

    # 4. Report
    print(f"[check] Epoch: {epoch}/20 | Train Loss: {train_loss:.3f} | "
          f"Val Loss: {val_loss:.3f} | BLEU: {bleu:.1f} | LR: {lr:.8f} | "
          f"Time: {wall_time/60:.1f}m | Process alive: {proc_alive}")

    # 5. Alerts
    alerts = []
    if not proc_alive and epoch < 20:
        alerts.append("CRITICAL: train.py process dead but training incomplete")
    if train_loss > 8:
        alerts.append("WARNING: Train loss >8 — may be diverging")
    if epoch >= 18:
        alerts.append(f"INFO: Near completion — epoch {epoch}/20. Prepare final download.")
    if epoch >= 20:
        alerts.append("DONE: Training complete (epoch 20). Download results.")

    for a in alerts:
        print(f"[check] {a}")

    # 6. Tail recent log
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

- [ ] **Step 2: Commit**

```bash
git add projects/transformer_iwslt/check_progress.py
git commit -m "feat: add progress monitoring script for cron-based Colab checks"
```

---

### Task 6: charts.py — Post-hoc Result Charts

**Files:**
- Create: `projects/transformer_iwslt/charts.py`

- [ ] **Step 1: Write charts.py**

```python
"""Post-hoc chart generation from downloaded metrics.jsonl files.

Reads output-{baseline,fixedpe,heads1}/metrics.jsonl + config.json,
produces 5 charts + results_summary.md.

Usage: python charts.py  (run locally after all experiments complete)
"""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).parent
OUTPUTS = {
    "baseline": PROJECT_DIR / "output-baseline",
    "fixed_pe": PROJECT_DIR / "output-fixedpe",
    "heads_1": PROJECT_DIR / "output-heads1",
}
CHARTS_DIR = PROJECT_DIR / "charts"
LABELS = {"baseline": "Baseline (8 heads, learned PE)", "fixed_pe": "Fixed Sinusoidal PE", "heads_1": "1 Attention Head"}


def load_metrics(exp_id: str) -> list[dict]:
    path = OUTPUTS[exp_id] / "metrics.jsonl"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {exp_id}")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    all_metrics = {}
    for exp_id in OUTPUTS:
        m = load_metrics(exp_id)
        if m:
            all_metrics[exp_id] = m

    if not all_metrics:
        print("No metrics found. Run experiments first.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for exp_id, metrics in all_metrics.items():
        epochs = [m["epoch"] for m in metrics]
        ax1.plot(epochs, [m["train_loss"] for m in metrics], label=LABELS[exp_id], linewidth=2)
        ax2.plot(epochs, [m["val_loss"] for m in metrics], label=LABELS[exp_id], linewidth=2)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.set_title("Validation Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)
    print("Saved loss_curves.png")

    # 2. BLEU curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for exp_id, metrics in all_metrics.items():
        epochs = [m["epoch"] for m in metrics]
        ax.plot(epochs, [m["bleu"] for m in metrics], label=LABELS[exp_id], linewidth=2, marker="o")
    ax.set_title("SacreBLEU on IWSLT'14 De->En"); ax.set_xlabel("Epoch"); ax.set_ylabel("BLEU")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "bleu_curves.png", dpi=150)
    plt.close(fig)
    print("Saved bleu_curves.png")

    # 3. Ablation bars
    fig, ax = plt.subplots(figsize=(6, 5))
    exp_names = []
    final_bleus = []
    for exp_id, metrics in all_metrics.items():
        exp_names.append(LABELS[exp_id])
        final_bleus.append(max(m["bleu"] for m in metrics))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(range(len(exp_names)), final_bleus, color=colors[:len(exp_names)])
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, fontsize=8, rotation=10)
    ax.set_ylabel("Best SacreBLEU")
    ax.set_title("Ablation: Final BLEU Comparison")
    for bar, bleu in zip(bars, final_bleus):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bleu:.1f}",
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ablation_bars.png", dpi=150)
    plt.close(fig)
    print("Saved ablation_bars.png")

    # 4. Attention heatmap — placeholder (requires trained model)
    attn_path = CHARTS_DIR / "attention_heads.png"
    if not attn_path.exists():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Attention map requires\ntrained model checkpoint\n(run after final download)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Attention Visualization (pending)")
        fig.savefig(attn_path, dpi=150)
        plt.close(fig)
        print("Saved attention_heads.png (placeholder)")

    # 5. Positional encoding comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Learned PE similarity
    import torch
    from model import sinusoidal_pe

    # Sinusoidal PE similarity
    sin_pe = sinusoidal_pe(128, 512)[0].numpy()
    sim = sin_pe @ sin_pe.T
    im1 = ax1.imshow(sim, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax1.set_title("Sinusoidal PE — Cosine Similarity")
    ax1.set_xlabel("Position"); ax1.set_ylabel("Position")
    plt.colorbar(im1, ax=ax1)

    # Learned PE placeholder (needs trained model)
    ax2.text(0.5, 0.5, "Learned PE requires\ntrained baseline model\n(run after final download)",
             transform=ax2.transAxes, ha="center", va="center", fontsize=12, color="gray")
    ax2.set_title("Learned PE — Cosine Similarity (pending)")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "position_encoding.png", dpi=150)
    plt.close(fig)
    print("Saved position_encoding.png (sinusoidal side done, learned side pending)")

    # 6. Results summary
    lines = ["# Transformer IWSLT'14 De->En — Results Summary\n"]
    lines.append("| Experiment | Best BLEU | Final Train Loss | Final Val Loss | Params |")
    lines.append("|---|---|---|---|---|")
    for exp_id, metrics in all_metrics.items():
        best_bleu = max(m["bleu"] for m in metrics)
        final = metrics[-1]
        lines.append(f"| {LABELS[exp_id]} | {best_bleu:.1f} | {final['train_loss']:.3f} | {final['val_loss']:.3f} | ~65M |")

    with open(CHARTS_DIR / "results_summary.md", "w") as f:
        f.write("\n".join(lines))
    print("Saved results_summary.md")

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify charts.py syntax**

```bash
python -m py_compile projects/transformer_iwslt/charts.py && echo "OK"
```

Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add projects/transformer_iwslt/charts.py
git commit -m "feat: add post-hoc chart generation (loss, BLEU, ablation, attention, PE)"
```

---

### Task 7: Integration Test — Local Forward + Backward + Checkpoint Roundtrip

**Files:**
- No new files — verify existing code

- [ ] **Step 1: Run full integration test on CPU**

```bash
cd projects/transformer_iwslt && python -c "
import torch, tempfile, os, json
from model import build_transformer, Transformer
from checkpoint import save_checkpoint, load_checkpoint

# Simulate a mini training run (2 epochs on synthetic data)
device = torch.device('cpu')
vocab_size = 2000
model = build_transformer('baseline', vocab_size=vocab_size)

# Synthetic data
src = torch.randint(1, vocab_size, (4, 30))
tgt = torch.randint(1, vocab_size, (4, 25))
tgt_in = tgt[:, :-1]
tgt_out = tgt[:, 1:]

src_mask = Transformer.create_padding_mask(0, src)
tgt_mask = (Transformer.create_padding_mask(0, tgt_in) |
            Transformer.create_causal_mask(tgt_in.size(1), device))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# Mini training loop (2 epochs)
for epoch in range(1, 3):
    model.train()
    logits = model(src, tgt_in, src_mask, tgt_mask)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: loss={loss.item():.4f}')

# Save checkpoint
tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
tmp.close()
save_checkpoint(tmp.name, model, optimizer, None, 2, loss.item(), loss.item(), 0.0, 1000, 10.0, {'exp_id': 'baseline'})

# Load into fresh model and continue
model2 = build_transformer('baseline', vocab_size=vocab_size)
opt_state, _, resumed_epoch, metrics, config = load_checkpoint(tmp.name, model2, device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optimizer2.load_state_dict(opt_state)

assert resumed_epoch == 2, f'Expected epoch 2, got {resumed_epoch}'
assert config['exp_id'] == 'baseline'

# Forward pass on resumed model — should produce same output
model2.eval()
with torch.no_grad():
    out1 = model(src, tgt_in, src_mask, tgt_mask)
    out2 = model2(src, tgt_in, src_mask, tgt_mask)
    max_diff = (out1 - out2).abs().max().item()
    assert max_diff < 1e-5, f'Outputs differ: max_diff={max_diff}'

os.unlink(tmp.name)
print('Integration test PASSED: forward, backward, checkpoint roundtrip all OK')
"
```

Expected: "Integration test PASSED: forward, backward, checkpoint roundtrip all OK"

- [ ] **Step 2: Commit (if any fixes were needed)**

```bash
git add -A && git diff --cached --stat
```

---

### Task 8: Deploy — Provision 3 Sessions and Launch

**Files:**
- Create: `projects/transformer_iwslt/exp_ids_baseline.txt`
- Create: `projects/transformer_iwslt/exp_ids_fixedpe.txt`
- Create: `projects/transformer_iwslt/exp_ids_heads1.txt`

- [ ] **Step 1: Create experiment ID files**

```bash
echo "baseline" > projects/transformer_iwslt/exp_ids_baseline.txt
echo "fixed_pe" > projects/transformer_iwslt/exp_ids_fixedpe.txt
echo "heads_1" > projects/transformer_iwslt/exp_ids_heads1.txt
```

- [ ] **Step 2: Provision all 3 sessions in parallel**

```bash
colab new --gpu T4 -s transformer-baseline
cb new --gpu T4 -s transformer-fixedpe
clb new --gpu T4 -s transformer-heads1
```

Verify: `colab sessions` + `cb sessions` + `clb sessions` all show running GPU sessions.

- [ ] **Step 3: Upload code to all 3 sessions**

```bash
# Account colab (baseline)
colab upload projects/transformer_iwslt/model.py /content/model.py
colab upload projects/transformer_iwslt/train.py /content/train.py
colab upload projects/transformer_iwslt/launch.py /content/launch.py
colab upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py
colab upload projects/transformer_iwslt/exp_ids_baseline.txt /content/exp_id.txt

# Account cb (fixed_pe)
cb upload projects/transformer_iwslt/model.py /content/model.py
cb upload projects/transformer_iwslt/train.py /content/train.py
cb upload projects/transformer_iwslt/launch.py /content/launch.py
cb upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py
cb upload projects/transformer_iwslt/exp_ids_fixedpe.txt /content/exp_id.txt

# Account clb (heads_1)
clb upload projects/transformer_iwslt/model.py /content/model.py
clb upload projects/transformer_iwslt/train.py /content/train.py
clb upload projects/transformer_iwslt/launch.py /content/launch.py
clb upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py
clb upload projects/transformer_iwslt/exp_ids_heads1.txt /content/exp_id.txt
```

- [ ] **Step 4: Launch all 3 sessions in parallel**

```bash
colab exec -s transformer-baseline -f launch.py --timeout 120
cb exec -s transformer-fixedpe -f launch.py --timeout 120
clb exec -s transformer-heads1 -f launch.py --timeout 120
```

- [ ] **Step 5: Set up 3 cron monitoring jobs**

```bash
# Cron A: colab account, baseline experiment
CronCreate cron="*/5 * * * *" prompt="Check transformer-baseline: first run 'colab sessions' to verify session alive, if dead run 'colab download /content/checkpoints/checkpoint_epoch*.pt ./projects/transformer_iwslt/output-baseline/checkpoints/ && colab download /content/metrics.jsonl ./projects/transformer_iwslt/output-baseline/' then if epoch < 20 reprovision with 'colab new --gpu T4 -s transformer-baseline && colab upload projects/transformer_iwslt/model.py /content/model.py && colab upload projects/transformer_iwslt/train.py /content/train.py && colab upload projects/transformer_iwslt/launch.py /content/launch.py && colab upload projects/transformer_iwslt/checkpoint.py /content/checkpoint.py && colab upload projects/transformer_iwslt/exp_ids_baseline.txt /content/exp_id.txt && colab upload projects/transformer_iwslt/output-baseline/checkpoints/checkpoint_epoch{N}.pt /content/checkpoint_epoch{N}.pt && echo /content/checkpoint_epoch{N}.pt > /tmp/resume_path.txt && colab upload /tmp/resume_path.txt /content/resume_path.txt && colab exec -s transformer-baseline -f launch.py --timeout 120'" durable=true recurring=true
```

Note: The actual cron prompts should use `colab exec -s transformer-baseline -f check_progress.py --timeout 15` as the check command. The full resume logic is manual — the cron detects session death and alerts you to intervene.

- [ ] **Step 6: Commit experiment ID files**

```bash
git add projects/transformer_iwslt/exp_ids_*.txt
git commit -m "chore: add experiment ID files for 3 Colab accounts"
```

---

### Task 9: Post-Training — Download Results and Generate Charts

- [ ] **Step 1: Download results from each account**

```bash
# Download baseline results
colab download /content/metrics.jsonl ./projects/transformer_iwslt/output-baseline/
colab download /content/config.json ./projects/transformer_iwslt/output-baseline/
# colab exec -s transformer-baseline -c "tar -czf /content/checkpoints.tar.gz -C /content checkpoints/"  # no -c flag!
# Instead pipe: echo 'import tarfile, os; t=tarfile.open("/content/checkpoints.tar.gz","w:gz"); t.add("/content/checkpoints","checkpoints"); t.close()' | colab exec -s transformer-baseline
colab download /content/checkpoints.tar.gz ./projects/transformer_iwslt/output-baseline/

# Download fixed_pe results
cb download /content/metrics.jsonl ./projects/transformer_iwslt/output-fixedpe/
cb download /content/config.json ./projects/transformer_iwslt/output-fixedpe/

# Download heads_1 results
clb download /content/metrics.jsonl ./projects/transformer_iwslt/output-heads1/
clb download /content/config.json ./projects/transformer_iwslt/output-heads1/
```

- [ ] **Step 2: Generate charts locally**

```bash
cd projects/transformer_iwslt && python charts.py
```

- [ ] **Step 3: Verify all charts exist**

```bash
ls projects/transformer_iwslt/charts/
```

Expected: `loss_curves.png`, `bleu_curves.png`, `ablation_bars.png`, `attention_heads.png`, `position_encoding.png`, `results_summary.md`

- [ ] **Step 4: Commit final results**

```bash
git add projects/transformer_iwslt/charts/ projects/transformer_iwslt/output-*/
git commit -m "feat: add Transformer IWSLT experiment results and charts"
```

---

### Task 10: Update README and CLAUDE.md

- [ ] **Step 1: Add project to CLAUDE.md codebase map**

Edit `CLAUDE.md` — add `transformer_iwslt/` entry to the projects table:

```
├── transformer_iwslt/    # Transformer (Attention Is All You Need) on IWSLT'14 De->En
│   ├── model.py           # Encoder-decoder Transformer, 65M params
│   ├── train.py           # IWSLT data pipeline, BPE tokenizer, training loop, beam search
│   ├── launch.py          # Colab bootstrap with checkpoint-resume
│   ├── check_progress.py  # Cron-based training monitor
│   ├── checkpoint.py      # Save/load helpers for multi-session resume
│   └── charts.py          # Post-hoc charts (loss, BLEU, ablation, attention, PE)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add transformer_iwslt to codebase map"
```
