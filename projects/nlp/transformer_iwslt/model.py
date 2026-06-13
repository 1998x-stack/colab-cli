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
            self.out_proj.weight = self.decoder.embed.weight
            self.encoder.embed.weight = self.decoder.embed.weight
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
        return (x == pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
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
