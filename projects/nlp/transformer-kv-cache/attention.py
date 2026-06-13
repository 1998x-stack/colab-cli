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

        q = self.q_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale

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
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k, v)
            total_len = k_full.size(2)
        else:
            k_full, v_full = k, v
            total_len = L

        scores = (q @ k_full.transpose(-2, -1)) * self.scale

        if kv_cache is None:
            causal_mask = torch.triu(
                torch.ones(L, total_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v_full).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out), kv_cache
