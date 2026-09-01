import time
from dataclasses import dataclass

import chess
import numpy as np
import numpy.typing as npt

from chessml.net import PolicyValueNet


def transposition_key(board: chess.Board) -> object:
    return board._transposition_key()


class Node:
    __slots__ = ("children", "moves", "n", "priors", "terminal", "total", "w")

    def __init__(
        self,
        moves: list[chess.Move],
        priors: npt.NDArray[np.float32],
        terminal: float | None = None,
    ) -> None:
        self.moves = moves
        self.priors = priors
        self.terminal = terminal
        self.n: npt.NDArray[np.float32] = np.zeros(len(moves), dtype=np.float32)
        self.w: npt.NDArray[np.float32] = np.zeros(len(moves), dtype=np.float32)
        self.children: list[Node | None] = [None] * len(moves)
        self.total = 0


_NO_MOVES: list[chess.Move] = []
_NO_PRIORS: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)


@dataclass
class SearchResult:
    moves: list[chess.Move]
    visits: npt.NDArray[np.float32]
    q: npt.NDArray[np.float32]
    root_value: float
    simulations: int


class MCTS:
    def __init__(self, net: PolicyValueNet, c_puct: float = 1.5) -> None:
        self.net = net
        self.c_puct = c_puct

    def _select(self, node: Node) -> int:
        q = np.divide(node.w, node.n, out=np.zeros_like(node.w), where=node.n > 0)
        u = (self.c_puct * np.sqrt(float(node.total) + 1.0)) * node.priors / (1.0 + node.n)
        return int(np.argmax(q + u))

    def _expand(
        self,
        board: chess.Board,
        key_counts: dict[object, int],
        path_keys: list[object],
    ) -> tuple[Node, float]:
        if board.halfmove_clock >= 100:
            return Node(_NO_MOVES, _NO_PRIORS, terminal=0.0), 0.0
        key = transposition_key(board)
        if key_counts.get(key, 0) + path_keys.count(key) >= 2:
            return Node(_NO_MOVES, _NO_PRIORS, terminal=0.0), 0.0
        if board.is_insufficient_material():
            return Node(_NO_MOVES, _NO_PRIORS, terminal=0.0), 0.0
        moves = list(board.legal_moves)
        if not moves:
            value = -1.0 if board.is_check() else 0.0
            return Node(_NO_MOVES, _NO_PRIORS, terminal=value), value
        moves, priors, value = self.net.evaluate(board)
        return Node(moves, priors), value

    def run(
        self,
        board: chess.Board,
        key_counts: dict[object, int],
        deadline: float,
        max_sims: int = 100_000,
    ) -> SearchResult:
        moves, priors, root_value = self.net.evaluate(board)
        if not moves:
            raise ValueError("no legal moves at search root")
        root = Node(moves, priors)

        sims = 0
        while sims < max_sims and time.monotonic() < deadline:
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
                child = node.children[idx]
                if child is None:
                    child, leaf_value = self._expand(sim_board, key_counts, path_keys)
                    node.children[idx] = child
                    break
                path_keys.append(transposition_key(sim_board))
                node = child

            value = leaf_value
            for parent, edge in reversed(path):
                value = -value
                parent.n[edge] += 1.0
                parent.w[edge] += value
                parent.total += 1
            sims += 1

        q = np.divide(root.w, root.n, out=np.zeros_like(root.w), where=root.n > 0)
        return SearchResult(root.moves, root.n.copy(), q, root_value, sims)
