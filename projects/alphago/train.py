"""AlphaGo Zero training pipeline — self-play → train → eval → save.

Single-session atomic loop. Checkpoint to Google Drive. Structured outputs for cron fetch.
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F

from game import GoBoard, GameConfig, GameRunner, BLACK, WHITE, EMPTY, OPPONENT
from model import AlphaGoNet, BatchEncoder
from mcts import MCTSConfig, MCTSBatchRunner, MCTSTree
from config import AlphaGoConfig, CONFIG_9X9_FAST, CONFIG_9X9_FIRST


# --- logger (self-contained, no external dependencies) ---


class TrainLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "a")
        self.start_time = time.time()

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def close(self):
        self._file.close()


class MetricsCSV:
    def __init__(self, path: str, columns: list[str]):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        self._file = open(path, "a")
        if not exists:
            self._file.write(",".join(columns) + "\n")
            self._file.flush()

    def write_row(self, **kwargs):
        cols = list(kwargs.keys())
        vals = [f"{kwargs[c]:.6f}" if isinstance(kwargs[c], float) else str(kwargs[c]) for c in cols]
        self._file.write(",".join(vals) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


# --- helpers ---


def _openmp_workaround():
    """Suppress duplicate libomp error. Harmless on single-GPU Colab VM."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# --- policy / action conversion ---


def action_to_move(action: int, board_size: int) -> tuple[int, int] | None:
    if action == board_size * board_size:
        return None
    return (action // board_size, action % board_size)


def move_to_action(move: tuple[int, int] | None, board_size: int) -> int:
    if move is None:
        return board_size * board_size
    r, c = move
    return r * board_size + c


# --- self-play ---


def _sample_action(pi: np.ndarray, temperature: float) -> int:
    """Sample action from π with given temperature. τ→0 means argmax."""
    if temperature < 1e-6:
        return int(np.argmax(pi))
    # Apply temperature and renormalize
    pi_t = pi ** (1.0 / temperature)
    pi_t = pi_t / pi_t.sum()
    return int(np.random.choice(len(pi_t), p=pi_t))


def selfplay_games(
    model: torch.nn.Module,
    encoder: BatchEncoder,
    cfg: AlphaGoConfig,
    mcts_cfg: MCTSConfig,
    n_games: int,
    device: str,
    logger: TrainLogger,
) -> list[dict]:
    """Generate self-play games using batched MCTS.

    Returns list of positions: [{board_tensor, pi, color, action_idx, ...}, ...]
    """
    boards = [GoBoard(GameConfig(board_size=cfg.board_size)) for _ in range(n_games)]
    histories = [encoder.init_history() for _ in range(n_games)]
    runners = [GameRunner(GameConfig(board_size=cfg.board_size)) for _ in range(n_games)]
    batch_runner = MCTSBatchRunner(model, encoder, mcts_cfg, device)

    positions: list[dict] = []
    active = set(range(n_games))

    logger.log(f"Self-play: {n_games} games, {mcts_cfg.num_simulations} MCTS sims/move")

    while active:
        # Get active boards and histories
        active_idx = sorted(active)
        active_boards = [boards[i] for i in active_idx]
        active_histories = [histories[i] for i in active_idx]

        # Batch MCTS search
        pis = batch_runner.search_all(active_boards, active_histories, add_dirichlet=True)

        # Sample moves and record positions
        finished = []
        for idx, pi in zip(active_idx, pis):
            board = boards[idx]
            runner = runners[idx]

            # Record position BEFORE move
            move_num = board.move_count
            temp = mcts_cfg.temperature if move_num < mcts_cfg.temperature_threshold else 0.01

            # Encode board for replay buffer
            board_tensor = encoder.encode(board, histories[idx], board.current_player).clone()

            # Sample action
            action = _sample_action(pi, temp)
            move = action_to_move(action, cfg.board_size)

            # Record
            positions.append({
                "board_tensor": board_tensor,
                "pi": pi,
                "color": board.current_player,
                "action": action,
                "game_idx": idx,
            })

            # Apply move
            try:
                cont = runner.step(move)
            except ValueError:
                cont = False

            # Update history for the player who just moved
            encoder.update_history(board, histories[idx], board.current_player)

            if not cont:
                finished.append(idx)

        for idx in finished:
            active.discard(idx)

    # Assign game outcomes by game_idx
    for pos in positions:
        gidx = pos["game_idx"]
        winner = runners[gidx].result().winner
        if winner == 0:
            pos["z"] = 0.0
        elif winner == pos["color"]:
            pos["z"] = 1.0
        else:
            pos["z"] = -1.0

    logger.log(f"Self-play done: {len(positions)} positions from {n_games} games, "
               f"black_wins={sum(1 for r in runners if r.result().winner == BLACK)}, "
               f"white_wins={sum(1 for r in runners if r.result().winner == WHITE)}")

    return positions


# --- training ---


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    positions: list[dict],
    cfg: AlphaGoConfig,
    device: str,
) -> dict:
    """One training epoch over all positions. Returns metrics dict."""
    model.train()
    n = len(positions)
    indices = np.random.permutation(n)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    for start in range(0, n, cfg.batch_size):
        batch_idx = indices[start:start + cfg.batch_size]
        batch = [positions[i] for i in batch_idx]

        # Stack board tensors
        boards = torch.cat([p["board_tensor"] for p in batch], dim=0).to(device)

        # Policy targets (π from MCTS)
        pi_targets = torch.from_numpy(np.stack([p["pi"] for p in batch])).float().to(device)

        # Value targets (z = game outcome from current player's perspective)
        z_targets = torch.tensor([[p["z"]] for p in batch], dtype=torch.float32).to(device)

        # Forward
        policy_logits, values = model(boards)

        # Loss: cross-entropy on policy + MSE on value
        policy_loss = -torch.sum(pi_targets * F.log_softmax(policy_logits, dim=1)) / len(batch_idx)
        value_loss = F.mse_loss(values, z_targets)
        loss = policy_loss + value_loss

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Loss is NaN/Inf! policy={policy_loss.item():.4f} value={value_loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return {
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss": total_value_loss / max(n_batches, 1),
        "total_loss": (total_policy_loss + total_value_loss) / max(n_batches, 1),
    }


# --- evaluation ---


def evaluate(
    current_model: torch.nn.Module,
    best_model: torch.nn.Module,
    encoder: BatchEncoder,
    cfg: AlphaGoConfig,
    device: str,
    logger: TrainLogger,
) -> dict:
    """Play N games: current vs best. Returns {elo_delta, wins, losses, draws, ...}."""
    logger.log(f"Eval: {cfg.n_eval_games} games, {cfg.eval_mcts_simulations} MCTS sims/move")

    mcts_cfg = MCTSConfig(
        num_simulations=cfg.eval_mcts_simulations,
        c_puct=1.0,
        temperature=0.01,  # near-greedy
        temperature_threshold=0,
        dirichlet_epsilon=0.0,  # no noise in eval
    )

    # Build value_fn closures for GPU-accelerated leaf evaluation
    def make_value_fn(model):
        def fn(board, current_player):
            h = encoder.init_history()
            t = encoder.encode(board, h, current_player)
            with torch.no_grad():
                _, v = model(t.to(device))
            return float(v.item())
        return fn

    wins = 0
    losses = 0
    draws = 0

    for game_i in range(cfg.n_eval_games):
        # Alternate: half the games current plays black, half white
        if game_i < cfg.n_eval_games // 2:
            black_model, white_model = current_model, best_model
        else:
            black_model, white_model = best_model, current_model

        black_value_fn = make_value_fn(black_model)
        white_value_fn = make_value_fn(white_model)

        board = GoBoard(GameConfig(board_size=cfg.board_size))
        hist_black = encoder.init_history()
        hist_white = encoder.init_history()
        passes = 0

        for _ in range(cfg.max_moves):
            color = board.current_player
            model_to_use = black_model if color == BLACK else white_model
            value_fn = black_value_fn if color == BLACK else white_value_fn
            hist = hist_black if color == BLACK else hist_white

            # MCTS search (single tree — eval games are sequential)
            tree = MCTSTree(board, mcts_cfg)
            tensor = encoder.encode(board, hist, color)
            with torch.no_grad():
                logits, root_value = model_to_use(tensor.to(device))
            pi = tree.search(
                board, logits[0].cpu().numpy(), float(root_value.item()),
                add_dirichlet=False, value_fn=value_fn,
            )

            move = action_to_move(int(np.argmax(pi)), cfg.board_size)

            # Update history for the player who is about to move
            encoder.update_history(board, hist, color)

            if move is None:
                passes += 1
            else:
                passes = 0
            board.apply_move(move)

            if passes >= 2:
                break

        winner = board.winner()

        # Determine if "current" won
        if (game_i < cfg.n_eval_games // 2 and winner == BLACK) or \
           (game_i >= cfg.n_eval_games // 2 and winner == WHITE):
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    win_rate = wins / cfg.n_eval_games
    result = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
    }
    logger.log(f"Eval result: {wins}W {losses}L {draws}D | win_rate={win_rate:.3f}")
    return result


# --- checkpoint ---


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: AlphaGoConfig,
    iteration: int,
    metrics: dict,
    is_best: bool,
    logger: TrainLogger,
):
    """Save full checkpoint and weights-only to Drive."""
    drive = cfg.drive_ckpt_dir
    local = f"{cfg.local_output_dir}/checkpoints"
    _ensure_dirs(drive, local)

    # Full checkpoint (optimizer + model + iteration + metrics)
    full = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "metrics": metrics,
        "board_size": cfg.board_size,
        "n_residual": cfg.n_residual,
        "n_filters": cfg.n_filters,
        "n_history": cfg.n_history,
    }
    torch.save(full, f"{local}/latest.pt")
    shutil.copy2(f"{local}/latest.pt", f"{drive}/latest.pt")

    # Weights-only (for network changes or direct inference)
    torch.save(model.state_dict(), f"{local}/latest_weights.pt")
    shutil.copy2(f"{local}/latest_weights.pt", f"{drive}/latest_weights.pt")

    # Best model
    if is_best:
        shutil.copy2(f"{local}/latest_weights.pt", f"{drive}/best.pt")
        with open(f"{drive}/version.txt", "w") as f:
            f.write(f"{iteration}\n{metrics.get('win_rate', 0):.4f}\n{metrics.get('elo', 0):.1f}\n")

    logger.log(f"Checkpoint saved: iter={iteration} best={is_best} drive={drive}")


def load_checkpoint(cfg: AlphaGoConfig, device: str) -> tuple[AlphaGoNet, torch.optim.Optimizer, int, dict]:
    """Load model, optimizer, iteration, metrics from Drive. Returns fresh model if no checkpoint."""
    ckpt_path = f"{cfg.drive_ckpt_dir}/latest.pt"

    model = AlphaGoNet(
        board_size=cfg.board_size,
        n_residual=cfg.n_residual,
        n_filters=cfg.n_filters,
        n_history=cfg.n_history,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.l2_weight_decay,
    )

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        iteration = ckpt.get("iteration", 0)
        metrics = ckpt.get("metrics", {})
        return model, optimizer, iteration, metrics

    return model, optimizer, 0, {}


def load_best_model(cfg: AlphaGoConfig, device: str) -> AlphaGoNet:
    """Load best model weights from Drive."""
    model = AlphaGoNet(
        board_size=cfg.board_size,
        n_residual=cfg.n_residual,
        n_filters=cfg.n_filters,
        n_history=cfg.n_history,
    ).to(device)

    best_path = f"{cfg.drive_ckpt_dir}/best.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model


# --- time budget ---


class TimeBudget:
    def __init__(self, max_s: int, logger: TrainLogger):
        self.max_s = max_s
        self.start = time.time()
        self.logger = logger

    def elapsed(self) -> float:
        return time.time() - self.start

    def check(self, fraction: float, phase: str) -> bool:
        """True if we're within budget for this phase."""
        ok = self.elapsed() < self.max_s * fraction
        if not ok:
            self.logger.log(f"[BUDGET] {phase} over budget ({self.elapsed():.0f}s / {self.max_s * fraction:.0f}s)")
        return ok

    def hard_cutoff(self) -> bool:
        """True if we're at the absolute cutoff."""
        return self.elapsed() >= self.max_s * 0.95

    def status(self) -> str:
        return f"elapsed={self.elapsed():.0f}s/{self.max_s}s"


# --- main ---


def main(cfg: AlphaGoConfig | None = None):
    if cfg is None:
        cfg = CONFIG_9X9_FIRST

    _openmp_workaround()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_dirs(f"{cfg.local_output_dir}/logs", f"{cfg.local_output_dir}/pngs", cfg.drive_ckpt_dir)

    logger = TrainLogger(f"{cfg.local_output_dir}/logs/train.log")
    csv = MetricsCSV(f"{cfg.local_output_dir}/metrics.csv",
                     ["iteration", "policy_loss", "value_loss", "win_rate", "wins", "losses", "draws",
                      "n_positions", "elapsed_s"])

    logger.log(f"=== AlphaGo Zero | {cfg.board_size}×{cfg.board_size} | "
               f"res={cfg.n_residual} filters={cfg.n_filters} device={device} ===")
    logger.log(f"Self-play: {cfg.n_selfplay_games} games × {cfg.num_simulations} sims | "
               f"Train: {cfg.n_epochs} epochs × {cfg.batch_size} batch | "
               f"Eval: {cfg.n_eval_games} games")
    if cfg.first_session:
        logger.log("FIRST SESSION — reduced params for cold start")

    budget = TimeBudget(cfg.max_session_s, logger)
    mcts_cfg = MCTSConfig(
        num_simulations=cfg.num_simulations,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_epsilon=cfg.dirichlet_epsilon,
        temperature=cfg.temperature,
        temperature_threshold=cfg.temperature_threshold,
    )
    encoder = BatchEncoder(board_size=cfg.board_size, n_history=cfg.n_history)

    try:
        # --- load ---
        model, optimizer, iteration, prev_metrics = load_checkpoint(cfg, device)
        best_model = load_best_model(cfg, device)
        logger.log(f"Loaded: iter={iteration} prev_metrics={prev_metrics} "
                   f"params={model.num_params:,} device={device}")

        iteration += 1
        session_start = time.time()

        # --- self-play ---
        if budget.check(cfg.budget_selfplay, "self-play"):
            positions = selfplay_games(model, encoder, cfg, mcts_cfg, cfg.n_selfplay_games, device, logger)
        else:
            logger.log("[SKIP] Self-play over budget, skipping")
            positions = []

        if budget.hard_cutoff():
            logger.log("[HARD-CUTOFF] Saving and exiting before training")
            save_checkpoint(model, optimizer, cfg, iteration, prev_metrics, False, logger)
            return

        # --- train ---
        train_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        if positions and budget.check(cfg.budget_train, "train"):
            logger.log(f"Training: {len(positions)} positions, {cfg.n_epochs} epochs")
            for ep in range(cfg.n_epochs):
                ep_metrics = train_epoch(model, optimizer, positions, cfg, device)
                logger.log(f"  Epoch {ep+1}/{cfg.n_epochs} | "
                           f"policy_loss={ep_metrics['policy_loss']:.4f} | "
                           f"value_loss={ep_metrics['value_loss']:.4f}")
                train_metrics = ep_metrics

                if budget.hard_cutoff():
                    logger.log("[HARD-CUTOFF] During training — saving and exiting")
                    save_checkpoint(model, optimizer, cfg, iteration, train_metrics, False, logger)
                    return

        # --- eval ---
        eval_metrics = {"wins": 0, "losses": 0, "draws": 0, "win_rate": 0.0}
        if budget.check(cfg.budget_eval, "eval"):
            eval_metrics = evaluate(model, best_model, encoder, cfg, device, logger)
        else:
            logger.log("[SKIP] Eval over budget")

        # --- save ---
        is_best = eval_metrics["win_rate"] > 0.55
        all_metrics = {**train_metrics, **eval_metrics, "iteration": iteration}
        save_checkpoint(model, optimizer, cfg, iteration, all_metrics, is_best, logger)

        # --- write artifacts ---
        elapsed = time.time() - session_start
        csv.write_row(
            iteration=iteration,
            policy_loss=train_metrics["policy_loss"],
            value_loss=train_metrics["value_loss"],
            win_rate=eval_metrics["win_rate"],
            wins=eval_metrics["wins"],
            losses=eval_metrics["losses"],
            draws=eval_metrics["draws"],
            n_positions=len(positions),
            elapsed_s=elapsed,
        )

        # Summary
        summary = {
            "iteration": iteration,
            "board_size": cfg.board_size,
            "params": model.num_params,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "is_best": is_best,
            "elapsed_s": elapsed,
            "n_positions": len(positions),
        }
        with open(f"{cfg.local_output_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.log(f"=== Session complete | iter={iteration} | "
                   f"best={'YES' if is_best else 'no'} | {budget.status()} ===")

    except RuntimeError as e:
        msg = str(e)
        if "NaN" in msg or "Inf" in msg:
            logger.log(f"[FATAL] NaN/Inf detected — saving and exiting: {e}")
            save_checkpoint(model, optimizer, cfg, iteration, prev_metrics, False, logger)
        else:
            logger.log(f"[FATAL] Runtime error: {e}")
            raise
    except Exception as e:
        logger.log(f"[FATAL] Unexpected error: {e}")
        raise
    finally:
        logger.close()
        csv.close()


# --- smoke test (local, no Drive, no GPU required) ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--first", action="store_true", help="Use first-session config (reduced)")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test with minimal params")
    args = parser.parse_args()

    if args.dry_run:
        cfg = AlphaGoConfig(
            board_size=9,
            n_residual=3,
            n_filters=32,
            n_history=2,
            num_simulations=20,
            n_selfplay_games=3,
            n_eval_games=4,
            n_epochs=2,
            max_session_s=600,
            drive_ckpt_dir="/tmp/alphago-test-ckpt",
            local_output_dir="/tmp/alphago-test-out",
            first_session=False,
        )
        print(f"Dry run: {cfg.n_selfplay_games} games × {cfg.num_simulations} sims, "
              f"{cfg.n_epochs} epochs, {cfg.board_size}×{cfg.board_size}")
        main(cfg)
    elif args.first:
        main(CONFIG_9X9_FIRST)
    else:
        main(CONFIG_9X9_FAST)
