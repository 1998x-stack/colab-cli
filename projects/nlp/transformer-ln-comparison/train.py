#!/usr/bin/env python3
"""Post-LN vs Pre-LN Transformer comparison on IWSLT2017 DE-EN translation.

Post-LN (original):  x = LayerNorm(x + Sublayer(x))   — LN after residual
Pre-LN  (modern):    x = x + Sublayer(LayerNorm(x))    — LN before sublayer

Usage:
    python train.py --ln_type post --max_steps 500
    python train.py --ln_type pre  --max_steps 500
"""
import argparse
import collections
import json
import math
import os
import time
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IWSLT_ZIP_URL = (
    "https://huggingface.co/datasets/IWSLT/iwslt2017/resolve/main/"
    "data/2017-01-trnted/texts/de/en/de-en.zip"
)
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_iwslt_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("<"):
        if line.startswith("<seg"):
            part = line.split(">", 1)[1]
            text = part.rsplit("<", 1)[0]
            return text.strip()
        return None
    return line


def load_iwslt_pairs(data_dir: str, max_pairs: int = 0) -> list[tuple[str, str]]:
    """Download IWSLT2017 DE-EN from HF CDN, return (de, en) pairs."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "de-en.zip")
    de_path = os.path.join(data_dir, "train.de")
    en_path = os.path.join(data_dir, "train.en")

    if not os.path.exists(de_path) or not os.path.exists(en_path):
        if not os.path.exists(zip_path):
            print("[data] Downloading IWSLT ZIP (~18MB)...")
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(IWSLT_ZIP_URL, zip_path)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"[data] Retry {attempt+1}/3: {e}")
                    time.sleep(2)

        print("[data] Extracting ZIP...")
        with zipfile.ZipFile(zip_path) as zf:
            de_zip = "de-en/train.tags.de-en.de"
            en_zip = "de-en/train.tags.de-en.en"
            with zf.open(de_zip) as src, open(de_path, "wb") as dst:
                dst.write(src.read())
            with zf.open(en_zip) as src, open(en_path, "wb") as dst:
                dst.write(src.read())

    with open(de_path) as df, open(en_path) as ef:
        de_lines = df.readlines()
        en_lines = ef.readlines()

    pairs = []
    for de_line, en_line in zip(de_lines, en_lines):
        de_text = _parse_iwslt_line(de_line)
        en_text = _parse_iwslt_line(en_line)
        if de_text and en_text:
            pairs.append((de_text, en_text))

    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    print(f"[data] Loaded {len(pairs)} sentence pairs")
    return pairs


# ---------------------------------------------------------------------------
# Word-level tokenizer (no external deps)
# ---------------------------------------------------------------------------

def build_word_tokenizer(
    pairs: list[tuple[str, str]], vocab_size: int
) -> tuple[dict, dict]:
    """Build word-level vocab from frequency counts. Returns (stoi, itos)."""
    counter = collections.Counter()
    for de, en in pairs:
        counter.update(de.split())
        counter.update(en.split())

    vocab = [w for w, _ in counter.most_common(vocab_size - len(SPECIALS))]
    words = SPECIALS + vocab

    stoi = {w: i for i, w in enumerate(words)}
    itos = {i: w for i, w in enumerate(words)}
    print(f"[data] Vocab size: {len(stoi)} (top-{vocab_size} words)")
    return stoi, itos


def encode(text: str, stoi: dict, max_len: int, add_sos_eos: bool = True) -> list[int]:
    tokens = [stoi.get(w, UNK) for w in text.split()]
    if add_sos_eos:
        tokens = [SOS] + tokens[: max_len - 2] + [EOS]
    else:
        tokens = tokens[:max_len]
    return tokens


def pre_tokenize(
    pairs: list[tuple[str, str]],
    stoi: dict,
    max_len: int,
    save_path: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Tokenize all pairs, cache to .pt file."""
    if os.path.exists(save_path):
        print(f"[data] Loading cached tokens from {save_path}")
        return torch.load(save_path, weights_only=False)

    print(f"[data] Tokenizing {len(pairs)} pairs (one-time)...")
    data = []
    for i, (de, en) in enumerate(pairs):
        src = torch.tensor(encode(de, stoi, max_len), dtype=torch.long)
        tgt = torch.tensor(encode(en, stoi, max_len), dtype=torch.long)
        data.append((src, tgt))
        if (i + 1) % 5000 == 0:
            print(f"[data]   {i+1}/{len(pairs)} tokenized...")

    torch.save(data, save_path)
    print(f"[data] Saved to {save_path}")
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, tokenized: list[tuple[torch.Tensor, torch.Tensor]]):
        self.data = tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: list, pad_idx: int = PAD) -> tuple[torch.Tensor, torch.Tensor]:
    src_list, tgt_list = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src, tgt


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

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

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# --- Post-LN Encoder/Decoder layers (original Transformer) ---

class PostLNEncoderLayer(nn.Module):
    """x = LayerNorm(x + Dropout(Sublayer(x)))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class PostLNDecoderLayer(nn.Module):
    """x = LayerNorm(x + Dropout(Sublayer(x))), with cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# --- Pre-LN Encoder/Decoder layers (modern stable variant) ---

class PreLNEncoderLayer(nn.Module):
    """x = x + Dropout(Sublayer(LayerNorm(x)))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PreLNDecoderLayer(nn.Module):
    """x = x + Dropout(Sublayer(LayerNorm(x))), with cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


# --- Encoder / Decoder / Transformer ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len,
                 dropout, ln_type):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pe = self._sinusoidal_pe(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        LayerCls = PreLNEncoderLayer if ln_type == "pre" else PostLNEncoderLayer
        self.layers = nn.ModuleList([
            LayerCls(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        if ln_type == "pre":
            self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    @property
    def ln_type(self):
        return "pre" if isinstance(self.layers[0], PreLNEncoderLayer) else "post"

    def forward(self, x, mask=None):
        x = self.dropout(self.embed(x) * math.sqrt(self.d_model) + self.pe[:, :x.size(1), :])
        for layer in self.layers:
            x = layer(x, mask)
        if hasattr(self, "final_norm"):
            x = self.final_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len,
                 dropout, ln_type):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pe = self._sinusoidal_pe(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        LayerCls = PreLNDecoderLayer if ln_type == "pre" else PostLNDecoderLayer
        self.layers = nn.ModuleList([
            LayerCls(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        if ln_type == "pre":
            self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    @property
    def ln_type(self):
        return "pre" if isinstance(self.layers[0], PreLNDecoderLayer) else "post"

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.embed(x) * math.sqrt(self.d_model) + self.pe[:, :x.size(1), :])
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        if hasattr(self, "final_norm"):
            x = self.final_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=4, d_ff=512,
                 max_len=128, dropout=0.1, ln_type="post"):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff,
                               max_len, dropout, ln_type)
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff,
                               max_len, dropout, ln_type)
        self.proj = nn.Linear(d_model, vocab_size)
        # Tie embedding and projection weights
        self.proj.weight = self.encoder.embed.weight
        self.ln_type = ln_type
        self._init_weights()

    def _init_weights(self):
        """Xavier init for Linear, normal for Embedding — transformer standard."""
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if "embed" in name:
                nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
            elif p.size(-1) == self.d_model or p.size(0) == self.d_model:
                nn.init.xavier_uniform_(p, gain=1 / math.sqrt(2))
            else:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.proj(dec_out)

    @staticmethod
    def create_padding_mask(pad_idx, x):
        mask = (x == pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)
        return mask

    @staticmethod
    def create_causal_mask(seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                          diagonal=1).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# LR Scheduler (Noam, paper Sec 5.3)
# ---------------------------------------------------------------------------

class NoamScheduler:
    """lr = lr_scale * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))"""

    def __init__(self, optimizer, d_model: int, warmup_steps: int, lr_scale: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr_scale = lr_scale
        self._step = 0

    def step(self):
        self._step += 1
        rate = self._compute_rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = rate

    def get_lr(self):
        return self._compute_rate()

    def _compute_rate(self):
        arg1 = self._step ** (-0.5)
        arg2 = self._step * (self.warmup_steps ** (-1.5))
        return self.lr_scale * (self.d_model ** (-0.5)) * min(arg1, arg2)


# ---------------------------------------------------------------------------
# Logging utilities (inline, minimal)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.f = open(log_path, "a")

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class MetricsCSV:
    def __init__(self, path: str, columns: list[str]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w")
        self.f.write(",".join(columns) + "\n")
        self.f.flush()
        self.columns = columns

    def write_row(self, **kwargs):
        vals = []
        for col in self.columns:
            v = kwargs.get(col, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        self.f.write(",".join(vals) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class SummaryJSON:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def write(self, data: dict):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_output_dir(project_name: str = "transformer-ln-comparison",
                      local_root: str = "./output") -> str:
    if os.path.isdir("/content"):
        return f"/content/{project_name}-output"
    elif os.path.isdir("/kaggle"):
        return f"/kaggle/working/{project_name}-output"
    return os.path.join(local_root, f"{project_name}-output")


def setup_output_dirs(out_dir: str):
    for sub in ["logs", "pngs", "checkpoints"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


# ---------------------------------------------------------------------------
# Plotting (inline, headless-safe)
# ---------------------------------------------------------------------------

def plot_training_curves(steps, losses, lr_values, out_path, title="Training",
                         window=50):
    """Single-panel: loss (raw + moving avg) + LR on twin axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if len(steps) < 2:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, losses, alpha=0.25, color="tab:blue", linewidth=0.5)
    if len(losses) >= window:
        import numpy as np
        ma = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(steps[window-1:], ma, color="tab:blue", linewidth=1.5,
                 label=f"loss (MA-{window})")
        ax1.legend(loc="upper left")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2.plot(steps, lr_values, color="tab:orange", linewidth=1, label="LR")
    ax2.set_ylabel("LR", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_val_loss(model, val_loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = Transformer.create_padding_mask(PAD, src)
            tgt_mask = (Transformer.create_padding_mask(PAD, tgt_in) |
                        Transformer.create_causal_mask(tgt_in.size(1), device))

            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                logits = model(src, tgt_in, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train(args):
    # --- Setup ---
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}, LN type: {args.ln_type}")

    out_dir = args.output_dir or detect_output_dir()
    setup_output_dirs(out_dir)
    print(f"[setup] Output: {out_dir}")

    logger = Logger(os.path.join(out_dir, "logs", "train.log"))
    logger.log(f"LN={args.ln_type} d={args.d_model} h={args.n_heads} "
               f"L={args.n_layers} bs={args.batch_size} steps={args.max_steps}")

    # --- Data ---
    pairs = load_iwslt_pairs(args.data_dir, args.max_train_pairs)

    # 90/10 train/val split
    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    stoi, itos = build_word_tokenizer(pairs, args.vocab_size)
    vocab_size = len(stoi)

    cache_dir = os.path.join(args.data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_tokenized = pre_tokenize(
        train_pairs, stoi, args.max_len,
        os.path.join(cache_dir, f"train_{args.max_train_pairs}.pt")
    )
    val_tokenized = pre_tokenize(
        val_pairs, stoi, args.max_len,
        os.path.join(cache_dir, f"val_{args.max_train_pairs}.pt")
    )

    train_ds = TranslationDataset(train_tokenized)
    val_ds = TranslationDataset(val_tokenized)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, PAD), num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, PAD), num_workers=0,
                            pin_memory=(device.type == "cuda"))

    # --- Model ---
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        ln_type=args.ln_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Params: {n_params:,}")

    # --- Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps,
                              lr_scale=args.lr_scale)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # --- Metrics tracking ---
    csv_columns = ["step", "loss", "val_loss", "lr", "grad_norm", "elapsed_s",
                   "tokens_per_sec"]
    metrics_csv = MetricsCSV(os.path.join(out_dir, "metrics.csv"), csv_columns)

    steps_log = []
    losses_log = []
    lr_log = []

    t_start = time.time()
    total_tokens = 0
    step = 0
    best_val_loss = float("inf")

    logger.log(f"Training started — {args.max_steps} steps target")

    while step < args.max_steps:
        for src, tgt in train_loader:
            if step >= args.max_steps:
                break

            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = Transformer.create_padding_mask(PAD, src)
            tgt_mask = (Transformer.create_padding_mask(PAD, tgt_in) |
                        Transformer.create_causal_mask(tgt_in.size(1), device))

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(src, tgt_in, src_mask, tgt_mask)
                    loss = criterion(logits.reshape(-1, logits.size(-1)),
                                     tgt_out.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(src, tgt_in, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 tgt_out.reshape(-1))
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            args.grad_clip)
                optimizer.step()

            scheduler.step()
            step += 1
            total_tokens += src.numel() + tgt.numel()

            # --- Logging & metrics ---
            steps_log.append(step)
            losses_log.append(loss.item())
            lr_log.append(scheduler.get_lr())

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - t_start
                tps = total_tokens / elapsed if elapsed > 0 else 0

                # Validation loss (quick, every log interval)
                val_loss = compute_val_loss(model, val_loader, criterion, device)

                metrics_csv.write_row(
                    step=step,
                    loss=loss.item(),
                    val_loss=val_loss,
                    lr=scheduler.get_lr(),
                    grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    elapsed_s=round(elapsed, 1),
                    tokens_per_sec=round(tps),
                )

                logger.log(
                    f"Step {step:5d}/{args.max_steps} | "
                    f"loss={loss.item():.4f} | val_loss={val_loss:.4f} | "
                    f"lr={scheduler.get_lr():.6f} | grad={grad_norm:.2f} | "
                    f"elapsed={elapsed:.0f}s | tps={tps:.0f}"
                )

                # Update training curves
                plot_training_curves(
                    steps_log, losses_log, lr_log,
                    os.path.join(out_dir, "pngs", "training_curves.png"),
                    title=f"Transformer {args.ln_type.upper()}-LN — IWSLT2017 DE-EN",
                )

                # Track best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

    # --- Final ---
    total_time = time.time() - t_start
    logger.log(f"Done. {step} steps in {total_time:.0f}s "
               f"({total_time/60:.1f}m) | best_val_loss={best_val_loss:.4f}")

    # Save weights checkpoint
    ckpt_path = os.path.join(out_dir, "checkpoints", "weights_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "ln_type": args.ln_type,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "config": {k: v for k, v in vars(args).items()},
    }, ckpt_path)
    logger.log(f"Checkpoint saved: {ckpt_path}")

    # Write summary
    summary = SummaryJSON(os.path.join(out_dir, "summary.json"))
    summary.write({
        "ln_type": args.ln_type,
        "n_params": n_params,
        "steps_completed": step,
        "final_train_loss": round(losses_log[-1], 4) if losses_log else None,
        "best_val_loss": round(best_val_loss, 4),
        "total_time_s": round(total_time, 1),
        "config": {k: v for k, v in vars(args).items() if k != "output_dir"},
    })

    logger.close()
    metrics_csv.close()

    return {
        "ln_type": args.ln_type,
        "steps": steps_log,
        "losses": losses_log,
        "lr_values": lr_log,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-LN vs Pre-LN Transformer comparison — IWSLT2017 DE-EN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--ln_type", type=str, default="post", choices=["post", "pre"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vocab_size", type=int, default=8000)
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="Scale factor for Noam LR (lower = more stable)")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=25)
    # Data
    p.add_argument("--max_train_pairs", type=int, default=25000)
    p.add_argument("--data_dir", type=str, default="/content/iwslt_data")
    p.add_argument("--output_dir", type=str, default="")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    # Quick arg check: if running on local (not Colab), override data_dir
    args = parse_args()
    if not os.path.isdir("/content") and args.data_dir == "/content/iwslt_data":
        args.data_dir = os.path.join(os.path.dirname(__file__) or ".", "iwslt_data")
    if not args.output_dir:
        args.output_dir = detect_output_dir()
    train(args)


if __name__ == "__main__":
    main()
