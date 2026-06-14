"""AlphaGo Zero dual-head network: shared residual tower + policy/value heads.

Config-driven: board_size, n_residual, n_filters, n_history all tunable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN, with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaGoNet(nn.Module):
    """Dual-head network for AlphaGo Zero.

    Input:  (B, n_history*2 + 1, board_size, board_size)
    Output: (policy_logits, value)
      - policy_logits: (B, board_size² + 1)  — raw logits (no softmax)
      - value:          (B, 1)                — tanh range [-1, 1]
    """

    def __init__(
        self,
        board_size: int = 9,
        n_residual: int = 5,
        n_filters: int = 64,
        n_history: int = 4,
    ):
        super().__init__()
        self.board_size = board_size
        self.n_filters = n_filters
        self.n_history = n_history

        in_channels = n_history * 2 + 1  # black hist + white hist + color plane

        # Input convolution
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.residuals = nn.Sequential(*[
            ResidualBlock(n_filters) for _ in range(n_residual)
        ])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(n_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        policy_flat_size = 2 * board_size * board_size
        self.policy_fc = nn.Linear(policy_flat_size, board_size * board_size + 1)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(n_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value)."""
        x = self.conv_input(x)
        x = self.residuals(x)

        # Policy head
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference mode: returns (policy_probs, value) with softmax applied."""
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_actions(self) -> int:
        return self.board_size * self.board_size + 1  # +1 for pass


# --- board state → network input ---


def board_to_tensor(
    board: "GoBoard",  # type: ignore[name-defined] # noqa: F821
    history: list["torch.Tensor"] | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert GoBoard state to network input tensor (1, C, H, W).

    Maintains a rolling history buffer externally (caller-managed).
    On first call (history=None), populates all history slots with current board.
    """
    import numpy as np

    # We'll get n_history from the caller — but we need to know it from somewhere.
    # Caller should pass it. We infer from history length if provided.
    raise NotImplementedError("Use BatchEncoder instead — see below.")


class BatchEncoder:
    """Encodes board states into network input tensors with history tracking.

    Usage:
        encoder = BatchEncoder(board_size=9, n_history=4)

        # Start of game
        history_w = encoder.init_history()
        history_b = encoder.init_history()

        # After each move
        tensor = encoder.encode(board, history_w, color=WHITE)
        # ... use tensor for network forward ...
        encoder.update_history(board, history_w, color=WHITE)
    """

    def __init__(self, board_size: int = 9, n_history: int = 4):
        self.board_size = board_size
        self.n_history = n_history

    def init_history(self) -> list[torch.Tensor]:
        """Return a zero-initialized history buffer."""
        return [
            torch.zeros((self.board_size, self.board_size), dtype=torch.float32)
            for _ in range(self.n_history * 2)
        ]

    def encode(
        self,
        board: "GoBoard",  # type: ignore[name-defined] # noqa: F821
        history: list[torch.Tensor],
        color: int,
    ) -> torch.Tensor:
        """Encode board state into a (1, C, H, W) tensor.

        Args:
            board: GoBoard in current state (before the move to be predicted).
            history: list of 2*n_history tensors — first n_history are black planes,
                     next n_history are white planes.
            color: BLACK or WHITE — the player about to move.
        """
        import numpy as np

        bs = self.board_size
        n = self.n_history

        # Current board as binary planes
        black = torch.from_numpy((board.board == 1).astype(np.float32))
        white = torch.from_numpy((board.board == 2).astype(np.float32))

        # Color plane: all 1s if black to move, all 0s if white
        color_plane = torch.ones((bs, bs), dtype=torch.float32) if color == 1 else torch.zeros((bs, bs), dtype=torch.float32)

        # Build channels: [black_hist..., white_hist..., color]
        channels = history[:n] + [black] + history[n:2*n] + [white] + history[n+1:2*n] + [color_plane]
        # Wait, this is wrong. Let me rethink.

        # The history for each color should be the last n board states.
        # history[:n] = black's last n positions
        # history[n:] = white's last n positions
        # After each move, we update history for the player who moved.

        channels = history[:n] + history[n:] + [color_plane]
        return torch.stack(channels, dim=0).unsqueeze(0)  # (1, C, H, W)

    def update_history(
        self,
        board: "GoBoard",  # type: ignore[name-defined] # noqa: F821
        history: list[torch.Tensor],
        color: int,
    ):
        """Push current board's stone positions into history for `color`."""
        import numpy as np

        n = self.n_history
        offset = 0 if color == 1 else n  # black=first n slots, white=next n

        # Shift and push
        stone_plane = torch.from_numpy((board.board == color).astype(np.float32))
        for i in range(offset, offset + n - 1):
            history[i] = history[i + 1].clone()
        history[offset + n - 1] = stone_plane


# --- smoke test ---

if __name__ == "__main__":
    board_size = 9
    model = AlphaGoNet(board_size=board_size, n_residual=5, n_filters=64, n_history=4)

    print(f"AlphaGoNet ({board_size}×{board_size})")
    print(f"  Params: {model.num_params:,}")
    print(f"  Actions: {model.num_actions}")
    print(f"  Input channels: {model.n_history * 2 + 1}")

    # Forward pass with random input
    bs, c, h, w = 1, model.n_history * 2 + 1, board_size, board_size
    x = torch.randn(bs, c, h, w)
    policy_logits, value = model(x)

    print(f"\n  Input:  {tuple(x.shape)}")
    print(f"  Policy: {tuple(policy_logits.shape)}  (logits, before softmax)")
    print(f"  Value:  {tuple(value.shape)}  (tanh, [-1, 1])")

    # Batch forward
    x_batch = torch.randn(8, c, h, w)
    p_batch, v_batch = model(x_batch)
    print(f"\n  Batch-8 policy: {tuple(p_batch.shape)}")
    print(f"  Batch-8 value:  {tuple(v_batch.shape)}")

    # Inference mode
    probs, v = model.predict(x)
    print(f"\n  Policy probs sum: {probs.sum().item():.4f} (should be 1.0)")
    print(f"  Value range: [{v.min().item():.3f}, {v.max().item():.3f}]")

    # Verify weight init sanity — policy logits should not be all identical
    first_probs = probs[0, :5]
    print(f"  First 5 probs: {first_probs.tolist()}")

    print("\n  OK: Model forward pass works.")
