"""All AlphaGo hyperparameters in one place. BOARD_SIZE drives everything else."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AlphaGoConfig:
    # --- board ---
    board_size: int = 9
    komi: float = 6.5
    max_moves: int = 300

    # --- model ---
    n_residual: int = 5
    n_filters: int = 64
    n_history: int = 4

    # --- MCTS ---
    num_simulations: int = 200
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30  # first N moves use τ=1, after that τ→0

    # --- self-play ---
    n_selfplay_games: int = 50  # per session
    n_eval_games: int = 100
    eval_mcts_simulations: int = 400  # more sims for evaluation accuracy

    # --- training ---
    batch_size: int = 64
    n_epochs: int = 5
    learning_rate: float = 0.001
    l2_weight_decay: float = 1e-4
    momentum: float = 0.9  # SGD momentum (AGZ uses SGD+momentum, not Adam)

    # --- session budget (seconds) ---
    max_session_s: int = 540  # 9 min hard cutoff
    budget_selfplay: float = 0.40
    budget_train: float = 0.30
    budget_eval: float = 0.20
    budget_save: float = 0.10

    # --- checkpoint paths (on VM) ---
    drive_ckpt_dir: str = "/content/drive/MyDrive/alphago-checkpoints"
    local_output_dir: str = "/content/alphago-output"

    # --- first session (cold start) ---
    first_session: bool = True

    @property
    def num_actions(self) -> int:
        return self.board_size * self.board_size + 1

    @property
    def input_channels(self) -> int:
        return self.n_history * 2 + 1

    @property
    def num_params_estimate(self) -> int:
        """Rough parameter count for display."""
        c = self.n_filters
        bs = self.board_size
        n_res = self.n_residual
        # input conv + residual tower + policy/value heads
        in_conv = self.input_channels * c * 9
        res = n_res * (2 * c * c * 9 + 4 * c)
        policy = 2 * c + 2 * c + (2 * bs * bs) * (bs * bs + 1)
        value = c + c + bs * bs * 64 + 64 + 64
        return in_conv + res + policy + value

    def for_second_session(self) -> "AlphaGoConfig":
        """Return a copy with first_session=False (full params)."""
        c = AlphaGoConfig()
        c.__dict__.update(self.__dict__)
        c.first_session = False
        return c


# Presets
CONFIG_9X9_FAST = AlphaGoConfig(
    board_size=9,
    n_residual=5,
    n_filters=64,
    n_history=4,
    num_simulations=200,
    n_selfplay_games=50,
    n_eval_games=100,
    n_epochs=5,
)

CONFIG_9X9_FIRST = AlphaGoConfig(
    board_size=9,
    n_selfplay_games=20,
    n_eval_games=1,  # first session: eval is bottleneck (single-tree MCTS), verify pipeline only
    eval_mcts_simulations=100,
    n_epochs=3,
    num_simulations=100,
    first_session=True,
)

CONFIG_19X19 = AlphaGoConfig(
    board_size=19,
    n_residual=20,
    n_filters=256,
    n_history=8,
    num_simulations=800,
    n_selfplay_games=50,
    n_eval_games=100,
    n_epochs=5,
    max_moves=500,
)
