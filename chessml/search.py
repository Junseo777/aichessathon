import threading
import time
from dataclasses import dataclass

import chess
import numpy as np
import numpy.typing as npt

from chessml.encoding import transposition_key
from chessml.net import PolicyValueNet


class Node:
    __slots__ = ("children", "moves", "n", "priors", "terminal", "total", "value", "w")

    def __init__(
        self,
        moves: list[chess.Move],
        priors: npt.NDArray[np.float32],
        value: float,
        terminal: float | None = None,
    ) -> None:
        self.moves = moves
        self.priors = priors
        self.value = value
        self.terminal = terminal
        self.n: npt.NDArray[np.float32] = np.zeros(len(moves), dtype=np.float32)
        self.w: npt.NDArray[np.float32] = np.zeros(len(moves), dtype=np.float32)
        self.children: list[Node | None] = [None] * len(moves)
        self.total = 0


_NO_MOVES: list[chess.Move] = []
_NO_PRIORS: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)


def _terminal(value: float) -> Node:
    return Node(_NO_MOVES, _NO_PRIORS, value, terminal=value)


@dataclass
class SearchResult:
    moves: list[chess.Move]
    visits: npt.NDArray[np.float32]
    q: npt.NDArray[np.float32]
    root_value: float
    simulations: int
    expanded: int
    root: Node


class MCTS:
    def __init__(
        self, net: PolicyValueNet, c_puct: float = 1.5, fpu_reduction: float = 0.25
    ) -> None:
        self.net = net
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction

    def _select(self, node: Node) -> int:
        # first-play urgency, docs/FINDING_fpu.md
        running = (node.value + float(node.w.sum())) / (1.0 + node.total)
        fpu = np.float32(running - self.fpu_reduction)
        q = np.divide(node.w, node.n, out=np.full_like(node.w, fpu), where=node.n > 0)
        u = (self.c_puct * np.sqrt(float(node.total) + 1.0)) * node.priors / (1.0 + node.n)
        return int(np.argmax(q + u))

    def _expand(self, board: chess.Board) -> Node:
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return _terminal(0.0)
        moves, priors, value = self.net.evaluate(board)
        if not moves:
            return _terminal(-1.0 if board.is_check() else 0.0)
        return Node(moves, priors, value)

    def run(
        self,
        board: chess.Board,
        key_counts: dict[object, int],
        deadline: float,
        max_sims: int = 100_000,
        root: Node | None = None,
        stop: threading.Event | None = None,
        node_budget: int | None = None,
    ) -> SearchResult:
        if root is None:
            root = self._expand(board)
        if root.terminal is not None:
            raise ValueError("no legal moves at search root")

        sims = 0
        expanded = 0
        while sims < max_sims and time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                break
            if node_budget is not None and expanded >= node_budget:
                break
            sim_board = board.copy(stack=False)
            path_keys: list[object] = []
            node = root
            path: list[tuple[Node, int]] = []

            while True:
                if node.terminal is not None:
                    leaf_value = node.terminal
                    break
                idx = self._select(node)
                sim_board.push(node.moves[idx])
                path.append((node, idx))
                key = transposition_key(sim_board)
                if key_counts.get(key, 0) + path_keys.count(key) >= 2:
                    leaf_value = 0.0
                    break
                child = node.children[idx]
                if child is None:
                    child = self._expand(sim_board)
                    node.children[idx] = child
                    expanded += 1
                    leaf_value = child.value
                    break
                path_keys.append(key)
                node = child

            value = leaf_value
            for parent, edge in reversed(path):
                value = -value
                parent.n[edge] += 1.0
                parent.w[edge] += value
                parent.total += 1
            sims += 1

        q = np.divide(root.w, root.n, out=np.zeros_like(root.w), where=root.n > 0)
        return SearchResult(root.moves, root.n.copy(), q, root.value, sims, expanded, root)
