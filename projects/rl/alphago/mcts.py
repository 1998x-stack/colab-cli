"""MCTS for AlphaGo Zero — PUCT selection, Dirichlet noise, temperature sampling.

Single-tree MCTS: use MCTSTree directly.
Batch multi-tree: use MCTSBatchRunner (multiple games, batched network forward).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from game import GoBoard, GameConfig, BLACK, WHITE, EMPTY, OPPONENT

# --- MCTS node ---


class MCTSNode:
    __slots__ = ("action", "parent", "children", "N", "W", "Q", "P", "is_expanded")

    def __init__(self, action: int | None = None, parent: "MCTSNode | None" = None):
        self.action = action  # action that led to this node
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}
        self.N: dict[int, int] = {}  # visit count per action
        self.W: dict[int, float] = {}  # total value per action
        self.Q: dict[int, float] = {}  # mean value per action
        self.P: dict[int, float] = {}  # prior probability per action
        self.is_expanded = False


# --- MCTS config ---


class MCTSConfig:
    def __init__(
        self,
        num_simulations: int = 200,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        temperature_threshold: int = 30,  # first N moves use τ=1
        temperature: float = 1.0,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_threshold = temperature_threshold
        self.temperature = temperature


# --- single-tree MCTS ---


class MCTSTree:
    """MCTS for one board position. Call search() → get policy π."""

    def __init__(self, board: GoBoard, config: MCTSConfig | None = None):
        self.cfg = config or MCTSConfig()
        self.root = MCTSNode()
        self.board_size = board.size

    def _action_to_move(self, action: int) -> tuple[int, int] | None:
        """Map action index (0..N²) to (r,c) or None (pass)."""
        if action == self.board_size * self.board_size:
            return None  # pass
        r = action // self.board_size
        c = action % self.board_size
        return (r, c)

    def _move_to_action(self, move: tuple[int, int] | None) -> int:
        if move is None:
            return self.board_size * self.board_size
        r, c = move
        return r * self.board_size + c

    def _legal_action_mask(self, board: GoBoard) -> tuple[list[int], np.ndarray]:
        """Return (legal_actions, mask_array) where mask_array[action] = 1 if legal."""
        legal_moves = board.legal_moves()
        legal_actions = [self._move_to_action(m) for m in legal_moves]
        mask = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
        mask[legal_actions] = 1.0
        return legal_actions, mask

    def search(
        self,
        board: GoBoard,
        policy_logits: np.ndarray,  # from network, raw logits over all actions
        value: float,  # from network, tanh value of root state
        add_dirichlet: bool = True,
        value_fn: "callable | None" = None,  # (GoBoard, int) → float — network value eval
    ) -> np.ndarray:
        """Run MCTS search from root. Returns visit-count policy π.

        Args:
            board: current board (will be modified and restored).
            policy_logits: shape (num_actions,) — raw network policy output.
            value: root state value from network.
            add_dirichlet: add Dirichlet noise at root (True for self-play, False for eval).
            value_fn: if provided, used for leaf evaluation instead of random rollout.
                      Called as value_fn(board, current_player) → float in [-1, 1].

        Returns:
            π: shape (num_actions,) — visit count distribution.
        """
        num_actions = self.board_size * self.board_size + 1
        legal_actions, mask = self._legal_action_mask(board)

        # Apply mask to policy logits, then softmax
        masked_logits = policy_logits.copy()
        masked_logits[mask == 0] = -1e10
        priors = self._softmax(masked_logits)

        # Expand root
        self.root.is_expanded = True
        for a in legal_actions:
            self.root.P[a] = priors[a]
            self.root.N[a] = 0
            self.root.W[a] = 0.0
            self.root.Q[a] = 0.0

        # Dirichlet noise at root
        if add_dirichlet:
            alpha_vec = np.full(len(legal_actions), self.cfg.dirichlet_alpha)
            noise = np.random.dirichlet(alpha_vec)
            eps = self.cfg.dirichlet_epsilon
            for i, a in enumerate(legal_actions):
                self.root.P[a] = (1 - eps) * self.root.P[a] + eps * noise[i]

        # --- simulation loop ---
        for _ in range(self.cfg.num_simulations):
            sim_board = self._copy_board(board)
            self._simulate(sim_board, self.root, value_fn=value_fn)

        # Build π from visit counts
        total_N = sum(self.root.N.values())
        pi = np.zeros(num_actions, dtype=np.float32)
        if total_N > 0:
            for a in self.root.N:
                pi[a] = self.root.N[a] / total_N

        return pi

    def _simulate(self, board: GoBoard, node: MCTSNode, value_fn: "callable | None" = None):
        """One MCTS simulation: select → expand → evaluate → backup.

        If value_fn is provided, leaf evaluation uses network value (GPU-fast).
        Otherwise falls back to random rollout (CPU-slow, for smoke tests only).
        """
        path: list[tuple[MCTSNode, int]] = []
        cur = node

        # Traverse tree to a leaf
        while True:
            # If node not yet expanded → evaluate it
            if not cur.is_expanded:
                break

            # Select best action via PUCT
            a = self._select(cur)
            if a < 0:
                # No legal actions available → terminal
                v = self._terminal_value(board)
                self._backup(path, v)
                return

            # Apply move
            board.apply_move(self._action_to_move(a))
            path.append((cur, a))

            # Move to existing child or create a new leaf node
            if a in cur.children:
                cur = cur.children[a]
            else:
                cur.children[a] = MCTSNode(action=a, parent=cur)
                cur = cur.children[a]
                break

        # cur is an unexpanded leaf → evaluate
        if value_fn is not None:
            v = value_fn(board, board.current_player)
        else:
            v = self._random_rollout(board)
        self._backup(path, v)

    def _select(self, node: MCTSNode) -> int:
        """PUCT selection: argmax Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))."""
        best_a = -1
        best_score = -float("inf")
        sqrt_N = math.sqrt(sum(node.N.values()) + 1)

        for a in node.P:
            q = node.Q.get(a, 0.0)
            u = self.cfg.c_puct * node.P[a] * sqrt_N / (1 + node.N.get(a, 0))
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a

        return best_a

    def _backup(self, path: list[tuple[MCTSNode, int]], leaf_value: float):
        """Backpropagate value through the path. Value is from current player's perspective."""
        for node, action in reversed(path):
            node.N[action] = node.N.get(action, 0) + 1
            node.W[action] = node.W.get(action, 0.0) + leaf_value
            node.Q[action] = node.W[action] / node.N[action]
            leaf_value = -leaf_value  # flip for opponent

    def _random_rollout(self, board: GoBoard, max_steps: int = 50) -> float:
        """Fast random rollout for leaf evaluation (fallback, replaced by network in batch mode)."""
        b = self._copy_board(board)
        passes = 0
        for _ in range(max_steps):
            legal = b.legal_moves()
            if len(legal) == 1 and legal[0] is None:
                break
            m = legal[np.random.randint(len(legal))]
            if m is None:
                passes += 1
                if passes >= 2:
                    break
            else:
                passes = 0
            b.apply_move(m)

        black_score, white_score = b.score()
        winner = b.winner()
        if winner == board.current_player:
            return 1.0
        elif winner == OPPONENT[board.current_player]:
            return -1.0
        return 0.0

    def _terminal_value(self, board: GoBoard) -> float:
        """Value at terminal position. Returns +1 if current player wins, -1 if loses."""
        winner = board.winner()
        if winner == 0:
            return 0.0  # draw
        return 1.0 if winner == board.current_player else -1.0

    # --- helpers ---

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def _copy_board(board: GoBoard) -> GoBoard:
        b = GoBoard(GameConfig(board_size=board.size))
        b.board = board.board.copy()
        b._hash_history = list(board._hash_history)
        b.move_count = board.move_count
        return b


# --- batched multi-tree MCTS ---


class MCTSBatchRunner:
    """Runs multiple MCTS trees in parallel with batched network evaluation.

    Usage:
        runner = MCTSBatchRunner(model, encoder, mcts_config, device)
        pis = runner.search_all(boards)  # list of π arrays
    """

    def __init__(
        self,
        model: "torch.nn.Module",  # noqa: F821
        encoder: "BatchEncoder",  # noqa: F821
        config: MCTSConfig | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.encoder = encoder
        self.cfg = config or MCTSConfig()
        self.device = device

    def search_all(
        self,
        boards: list[GoBoard],
        histories: list[list[torch.Tensor]],
        add_dirichlet: bool = True,
    ) -> list[np.ndarray]:
        """Run MCTS on all boards in parallel with batched network queries.

        Each board gets its own MCTS tree. Network calls are batched across trees.
        """
        n_boards = len(boards)
        trees = [MCTSTree(b, self.cfg) for b in boards]
        board_copies = [MCTSTree._copy_board(b) for b in boards]

        # Pre-compute root network values for all boards
        root_tensors = self._encode_batch(boards, histories)
        with torch.no_grad():
            logits_batch, values_batch = self.model(root_tensors.to(self.device))
        logits_batch = logits_batch.cpu().numpy()
        values_batch = values_batch.cpu().numpy()

        # Expand roots
        for i, tree in enumerate(trees):
            self._expand_root(
                tree, boards[i], logits_batch[i], values_batch[i], add_dirichlet
            )

        # Simulation loop — batch network calls at leaf nodes
        for sim in range(self.cfg.num_simulations):
            # Collect leaf states from all trees
            queries: list[tuple[int, MCTSNode, GoBoard]] = []  # (tree_idx, node, board copy)
            for i, tree in enumerate(trees):
                node, board = self._traverse_to_leaf(tree, board_copies[i])
                if board is not None:
                    queries.append((i, node, board))

            if not queries:
                break

            # Batch network evaluation for leaves
            leaf_boards = [q[2] for q in queries]
            leaf_histories = [histories[q[0]] for q in queries]
            leaf_tensors = self._encode_batch(leaf_boards, leaf_histories)
            with torch.no_grad():
                leaf_logits, leaf_values = self.model(leaf_tensors.to(self.device))
            leaf_logits = leaf_logits.cpu().numpy()
            leaf_values = leaf_values.squeeze(-1).cpu().numpy()

            # Expand leaf nodes and backup
            for qi, (tree_idx, node, board) in enumerate(queries):
                tree = trees[tree_idx]
                self._expand_and_backup(
                    tree, node, board, leaf_logits[qi], float(leaf_values[qi])
                )

        # Build π for all trees
        pis = []
        for tree in trees:
            total_N = sum(tree.root.N.values())
            pi = np.zeros(tree.board_size * tree.board_size + 1, dtype=np.float32)
            if total_N > 0:
                for a in tree.root.N:
                    pi[a] = tree.root.N[a] / total_N
            pis.append(pi)

        return pis

    def _expand_root(
        self,
        tree: MCTSTree,
        board: GoBoard,
        policy_logits: np.ndarray,
        value: float,
        add_dirichlet: bool,
    ):
        """Expand root with network priors."""
        num_actions = tree.board_size * tree.board_size + 1
        legal_moves = board.legal_moves()
        legal_actions = [tree._move_to_action(m) for m in legal_moves]

        # Masked softmax
        mask = np.full(num_actions, -1e10, dtype=np.float32)
        mask[legal_actions] = policy_logits[legal_actions]
        priors = MCTSTree._softmax(mask)

        tree.root.is_expanded = True
        for a in legal_actions:
            tree.root.P[a] = priors[a]
            tree.root.N[a] = 0
            tree.root.W[a] = 0.0
            tree.root.Q[a] = 0.0

        # Dirichlet noise
        if add_dirichlet:
            alpha_vec = np.full(len(legal_actions), tree.cfg.dirichlet_alpha)
            noise = np.random.dirichlet(alpha_vec)
            eps = tree.cfg.dirichlet_epsilon
            for i, a in enumerate(legal_actions):
                tree.root.P[a] = (1 - eps) * tree.root.P[a] + eps * noise[i]

    def _traverse_to_leaf(
        self, tree: MCTSTree, board: GoBoard
    ) -> tuple[MCTSNode, GoBoard | None]:
        """Traverse MCTS tree to find a leaf node. Returns (leaf_node, board_copy | None).

        If all paths terminate, returns (node, None).
        """
        node = tree.root
        path = []

        while node.is_expanded:
            # Check terminal
            if self._is_game_over(board):
                break

            # Select best action
            best_a = tree._select(node)
            if best_a < 0:
                break

            # Apply move
            try:
                board.apply_move(tree._action_to_move(best_a))
            except ValueError:
                break

            # Move to child or create one
            if best_a in node.children:
                node = node.children[best_a]
            else:
                # New leaf — need expansion
                child = MCTSNode(action=best_a, parent=node)
                node.children[best_a] = child
                return child, board

        # Reached a terminal or fully-explored node
        if not node.is_expanded:
            return node, board
        return node, None

    def _expand_and_backup(
        self,
        tree: MCTSTree,
        node: MCTSNode,
        board: GoBoard,
        policy_logits: np.ndarray,
        value: float,
    ):
        """Expand a leaf node with network output, then backup value."""
        if self._is_game_over(board):
            v = self._terminal_value(board, board.current_player)
            self._backup_from_node(node, v)
            return

        num_actions = tree.board_size * tree.board_size + 1
        legal_moves = board.legal_moves()
        legal_actions = [tree._move_to_action(m) for m in legal_moves]

        if not legal_actions:
            v = self._terminal_value(board, board.current_player)
            self._backup_from_node(node, v)
            return

        # Softmax over legal actions
        mask = np.full(num_actions, -1e10, dtype=np.float32)
        mask[legal_actions] = policy_logits[legal_actions]
        priors = MCTSTree._softmax(mask)

        # Expand
        node.is_expanded = True
        for a in legal_actions:
            node.P[a] = priors[a]
            node.N[a] = 0
            node.W[a] = 0.0
            node.Q[a] = 0.0

        # Backup from this node
        self._backup_from_node(node, value)

    def _backup_from_node(self, node: MCTSNode, value: float):
        """Backpropagate value from node up to root."""
        cur = node
        v = value
        while cur is not None and cur.action is not None:
            parent = cur.parent
            if parent is not None:
                a = cur.action
                parent.N[a] = parent.N.get(a, 0) + 1
                parent.W[a] = parent.W.get(a, 0.0) + v
                parent.Q[a] = parent.W[a] / parent.N[a]
            v = -v
            cur = parent

    def _encode_batch(
        self,
        boards: list[GoBoard],
        histories: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Encode a batch of boards into network input tensor."""
        tensors = []
        for b, h in zip(boards, histories):
            t = self.encoder.encode(b, h, b.current_player)
            tensors.append(t)
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _is_game_over(board: GoBoard) -> bool:
        return board.move_count >= board.cfg.max_moves

    @staticmethod
    def _terminal_value(board: GoBoard, current_player: int) -> float:
        winner = board.winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == current_player else -1.0


# --- smoke test ---

if __name__ == "__main__":
    import torch
    from model import AlphaGoNet, BatchEncoder

    print("=== MCTS smoke test (single tree) ===\n")

    board_size = 9
    board = GoBoard(GameConfig(board_size=board_size))
    model = AlphaGoNet(board_size=board_size, n_residual=5, n_filters=64, n_history=4)
    model.eval()
    encoder = BatchEncoder(board_size=board_size, n_history=4)

    cfg = MCTSConfig(num_simulations=50)

    # Play a few moves first
    for move in [(4, 4), (3, 3), (4, 3), (3, 4)]:
        board.apply_move(move)

    # Run MCTS
    tree = MCTSTree(board, cfg)
    hist = encoder.init_history()
    # Manually encode board for network
    tensor = encoder.encode(board, hist, board.current_player)
    with torch.no_grad():
        logits, value = model(tensor)

    pi = tree.search(board, logits[0].numpy(), value.item(), add_dirichlet=True)

    print(f"Board after {board.move_count} moves (Black to play):")
    print(board)
    print(f"\nMCTS π (top 10 actions):")
    top10 = np.argsort(-pi)
    for i, a in enumerate(top10[:10]):
        move = tree._action_to_move(int(a))
        if move is not None:
            print(f"  {i+1}. {move}: π={pi[a]:.4f}")
        else:
            print(f"  {i+1}. pass: π={pi[a]:.4f}")
    if board.move_count < tree.cfg.temperature_threshold:
        print("\n  (τ=1 sampling — diverse)")
    else:
        print("\n  (τ→0 — greedy)")

    print("\n=== Batch MCTS smoke test (3 parallel boards) ===\n")

    boards = [GoBoard(GameConfig(board_size=board_size)) for _ in range(3)]
    histories = [encoder.init_history() for _ in range(3)]
    # Play different openings
    openings = [[(4, 4)], [(0, 0)], [(4, 4), (3, 3), (4, 3)]]
    for b, ops in zip(boards, openings):
        for m in ops:
            b.apply_move(m)

    batch_runner = MCTSBatchRunner(model, encoder, cfg)
    pis = batch_runner.search_all(boards, histories, add_dirichlet=False)

    for i, (b, pi) in enumerate(zip(boards, pis)):
        top3 = np.argsort(-pi)[:3]
        moves = []
        for a in top3:
            move = tree._action_to_move(int(a))
            label = f"{move}" if move is not None else "pass"
            moves.append(f"{label} (π={pi[a]:.3f})")
        print(f"  Board {i} ({b.move_count} moves): {b.current_player} to play → top3: {moves}")

    print("\n  OK: MCTS works (single + batch modes).")
