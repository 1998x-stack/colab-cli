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
