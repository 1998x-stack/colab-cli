"""Decoder-only GPT transformer with pluggable attention and optional KV cache.

The model can be built in two modes:
- Training: uses CausalMultiHeadAttention (no cache overhead)
- Inference: uses CausalMHAWithCache (accepts per-layer KVCache)
"""
import time

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
    - Training: model(idx) — full sequence, no cache, returns logits [B, L, V]
    - Inference: model(idx, kv_caches=[...]) — single token, with caches,
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

        # Weight tying
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
        B, L = idx.shape
        assert L <= self.config.block_size, f"Sequence too long: {L} > {self.config.block_size}"

        if kv_caches is not None and kv_caches[0] is not None and kv_caches[0].seq_len > 0:
            pos = torch.arange(
                kv_caches[0].seq_len, kv_caches[0].seq_len + L,
                device=idx.device
            ).unsqueeze(0)
        else:
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)

        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(pos)
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
        self.eval()
        latencies = []

        if use_cache and self.use_cache:
            kv_caches = [KVCache() for _ in range(self.config.n_layer)]
        else:
            kv_caches = None

        for _ in range(max_new_tokens):
            t0 = time.perf_counter()

            if kv_caches is not None:
                x = idx[:, -1:]
                logits, kv_caches = self(x, kv_caches=kv_caches)
            else:
                x = idx
                logits, _ = self(x, kv_caches=None)

            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            latencies.append((time.perf_counter() - t0) * 1000)

        return idx, latencies
