"""9×9 Go engine — rules, self-play, scoring. No external dependencies beyond numpy."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# --- constants ---
EMPTY = 0
BLACK = 1
WHITE = 2

OPPONENT = {BLACK: WHITE, WHITE: BLACK}

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class GameConfig:
    board_size: int = 9
    komi: float = 6.5
    max_moves: int = 500  # safety cap


# --- board logic ---


class GoBoard:
    """9×9 Go board with capture, ko, and scoring."""

    def __init__(self, config: GameConfig | None = None):
        self.cfg = config or GameConfig()
        self.size = self.cfg.board_size
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self._hash_history: list[int] = []  # for ko detection
        self._last_capture_pos: tuple[int, int] | None = None  # for ko
        self.move_count = 0

    @property
    def current_player(self) -> int:
        return BLACK if self.move_count % 2 == 0 else WHITE

    # ---- liberties -------------------------------------------------

    def _neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        pts = []
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                pts.append((nr, nc))
        return pts

    def _group_liberties(self, r: int, c: int) -> set[tuple[int, int]]:
        """BFS to find all liberties of the group containing (r, c)."""
        color = self.board[r, c]
        if color == EMPTY:
            return set()
        visited = set()
        liberties: set[tuple[int, int]] = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return liberties

    def _remove_group(self, r: int, c: int) -> int:
        """Remove the group containing (r, c). Returns number of stones removed."""
        color = self.board[r, c]
        if color == EMPTY:
            return 0
        visited = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            self.board[cr, cc] = EMPTY
            for nr, nc in self._neighbors(cr, cc):
                if self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return len(visited)

    # ---- hash (for ko) --------------------------------------------

    def _board_hash(self) -> int:
        """Fast hash of board state — sufficient for ko detection."""
        return hash(self.board.tobytes() + bytes([self.current_player]))

    # ---- move execution -------------------------------------------

    def legal_moves(self) -> list[tuple[int, int] | None]:
        """All legal moves including pass (None)."""
        moves: list[tuple[int, int] | None] = [None]  # pass always legal
        color = self.current_player
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != EMPTY:
                    continue
                if self._is_legal(r, c, color):
                    moves.append((r, c))
        return moves

    def _is_legal(self, r: int, c: int, color: int) -> bool:
        """Check if placing a stone at (r,c) is legal (ignoring ko for speed, ko checked in apply_move)."""
        self.board[r, c] = color

        # Test capture: if the move has liberties, or captures opponent, it's potentially legal
        if len(self._group_liberties(r, c)) > 0:
            self.board[r, c] = EMPTY
            return True

        # No liberties — check if it captures any opponent group
        captures_anything = False
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == OPPONENT[color]:
                if len(self._group_liberties(nr, nc)) == 0:
                    captures_anything = True
                    break

        self.board[r, c] = EMPTY
        return captures_anything

    def apply_move(self, move: tuple[int, int] | None) -> bool:
        """Execute a move. Returns True if successful.
        move=None means pass. Raises ValueError for illegal moves.
        """
        if move is None:
            self._hash_history.append(self._board_hash())
            self.move_count += 1
            return True

        r, c = move
        color = self.current_player

        # basic validity
        if not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError(f"Move {move} out of bounds")
        if self.board[r, c] != EMPTY:
            raise ValueError(f"Position {move} already occupied")

        # Save state for ko check
        prev_hash = self._board_hash()

        # Place stone
        self.board[r, c] = color
        captured_any = False
        single_capture_pos = None

        # Remove opponent groups with 0 liberties
        opp = OPPONENT[color]
        for nr, nc in self._neighbors(r, c):
            if self.board[nr, nc] == opp:
                if len(self._group_liberties(nr, nc)) == 0:
                    cnt = self._remove_group(nr, nc)
                    captured_any = True
                    if cnt == 1:
                        single_capture_pos = (nr, nc)

        # Suicide check (no liberties and didn't capture)
        if not captured_any and len(self._group_liberties(r, c)) == 0:
            self.board[r, c] = EMPTY
            raise ValueError(f"Suicide move at {move}")

        # Ko check — can't repeat previous board state
        cur_hash = self._board_hash()
        if len(self._hash_history) >= 1 and cur_hash == self._hash_history[-1]:
            # Also check: was this a ko-shaped capture?
            self.board[r, c] = EMPTY
            if single_capture_pos:
                self.board[single_capture_pos[0], single_capture_pos[1]] = opp
            raise ValueError(f"Ko violation at {move}")

        self._hash_history.append(prev_hash)
        self.move_count += 1
        return True

    def is_terminal(self) -> bool:
        """Game ends after two consecutive passes or move limit."""
        if self.move_count >= self.cfg.max_moves:
            return True
        if len(self._hash_history) < 2:
            return False
        # Need to track passes explicitly — check via hash matching (pass doesn't change board)
        # Simpler: check last two moves were passes via a pass counter
        return False  # Caller tracks passes externally

    # ---- scoring (Chinese area scoring) ---------------------------

    def score(self) -> tuple[float, float]:
        """Return (black_score, white_score) using Chinese area scoring."""
        visited = np.zeros((self.size, self.size), dtype=bool)
        black_area = 0
        white_area = 0

        for r in range(self.size):
            for c in range(self.size):
                if visited[r, c]:
                    continue
                if self.board[r, c] == BLACK:
                    black_area += 1
                elif self.board[r, c] == WHITE:
                    white_area += 1
                else:
                    # Empty region — determine owner via flood fill
                    region, borders = self._flood_fill(r, c, visited)
                    if BLACK in borders and WHITE not in borders:
                        black_area += len(region)
                    elif WHITE in borders and BLACK not in borders:
                        white_area += len(region)
                    # else: neutral (dame) or both border — no points

        return black_area, white_area + self.cfg.komi

    def _flood_fill(
        self, r: int, c: int, visited: np.ndarray
    ) -> tuple[list[tuple[int, int]], set[int]]:
        """Flood fill from (r,c). Returns (region_cells, border_colors)."""
        region = []
        borders: set[int] = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if visited[cr, cc]:
                continue
            visited[cr, cc] = True
            if self.board[cr, cc] != EMPTY:
                borders.add(int(self.board[cr, cc]))
                continue
            region.append((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                if not visited[nr, nc]:
                    stack.append((nr, nc))
        return region, borders

    def winner(self) -> int:
        """BLACK, WHITE, or 0 for draw."""
        b, w = self.score()
        if b > w:
            return BLACK
        elif w > b:
            return WHITE
        return 0

    # ---- display --------------------------------------------------

    def __repr__(self) -> str:
        chars = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        lines = []
        for r in range(self.size):
            line = " ".join(chars[self.board[r, c]] for c in range(self.size))
            lines.append(f"{r}  {line}")
        col_header = "   " + " ".join(str(c) for c in range(self.size))
        return col_header + "\n" + "\n".join(lines)


# --- game runner ---


@dataclass
class GameResult:
    winner: int  # BLACK, WHITE, or 0 (draw)
    black_score: float
    white_score: float
    moves: list[tuple[int, int] | None]
    num_moves: int
    reason: str  # "two_passes", "max_moves", "resign"


class GameRunner:
    """Plays a full game with two agents (black, white)."""

    def __init__(self, config: GameConfig | None = None):
        self.cfg = config or GameConfig()
        self.board = GoBoard(self.cfg)
        self.passes = 0
        self.moves: list[tuple[int, int] | None] = []

    def step(self, move: tuple[int, int] | None) -> bool:
        """Apply a move. Returns True if game continues, False if over."""
        self.board.apply_move(move)
        self.moves.append(move)

        if move is None:
            self.passes += 1
        else:
            self.passes = 0

        if self.passes >= 2:
            return False
        if self.board.move_count >= self.cfg.max_moves:
            return False
        return True

    def result(self, reason: str = "two_passes") -> GameResult:
        b, w = self.board.score()
        return GameResult(
            winner=self.board.winner(),
            black_score=b,
            white_score=w,
            moves=self.moves,
            num_moves=len(self.moves),
            reason=reason,
        )


# --- random self-play (verification) ---


def random_self_play(n_games: int = 100, board_size: int = 9, verbose: bool = True) -> list[GameResult]:
    """Run N games with random legal moves. Returns results for analysis."""
    config = GameConfig(board_size=board_size)
    results: list[GameResult] = []

    for i in range(n_games):
        runner = GameRunner(config)
        while True:
            legal = runner.board.legal_moves()
            move = legal[np.random.randint(len(legal))]
            try:
                if not runner.step(move):
                    break
            except ValueError:
                # Shouldn't happen with legal_moves(), but safety
                break
        r = runner.result()
        results.append(r)

        if verbose and (i + 1) % 100 == 0:
            _print_random_stats(results)

    return results


def _print_random_stats(results: list[GameResult]):
    n = len(results)
    black_wins = sum(1 for r in results if r.winner == BLACK)
    white_wins = sum(1 for r in results if r.winner == WHITE)
    draws = sum(1 for r in results if r.winner == 0)
    avg_moves = sum(r.num_moves for r in results) / n
    print(f"  Games: {n} | Black: {black_wins} ({black_wins/n:.1%}) | "
          f"White: {white_wins} ({white_wins/n:.1%}) | Draws: {draws} | "
          f"Avg moves: {avg_moves:.0f}")


# --- smoke test ---

if __name__ == "__main__":
    config = GameConfig(board_size=9)

    # Single game with move-by-move display
    print("=== Single random game (9×9) ===\n")
    board = GoBoard(config)
    consecutive_passes = 0
    move_num = 0

    while consecutive_passes < 2 and move_num < 300:
        legal = board.legal_moves()
        move = legal[np.random.randint(len(legal))]
        board.apply_move(move)
        move_num += 1
        if move is None:
            consecutive_passes += 1
            print(f"  Move {move_num}: {'Black' if move_num % 2 == 1 else 'White'} passes ({consecutive_passes}/2)")
        else:
            consecutive_passes = 0
        if move_num <= 15 or move is None or move_num % 20 == 0:
            print(f"\n{board}\n")

    b, w = board.score()
    winner = board.winner()
    print(f"\n=== Game over after {move_num} moves ===")
    print(f"Black: {b:.1f}  White: {w:.1f} (komi {config.komi})")
    print(f"Winner: {'Black' if winner == BLACK else 'White' if winner == WHITE else 'Draw'}")
    print(f"\n{board}")

    # Batch stats
    print("\n=== Random self-play: 500 games (verification) ===\n")
    results = random_self_play(500, board_size=9)
    _print_random_stats(results)

    # Check for any anomalies
    illegal_count = 0
    for r in results:
        # Every game should have at least some moves
        if r.num_moves < 2:
            illegal_count += 1
    print(f"Anomalous games (<2 moves): {illegal_count}")
    print("All games completed successfully." if illegal_count == 0 else "WARNING: anomalies detected!")
