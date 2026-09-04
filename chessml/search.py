import math
import threading
import time
from dataclasses import dataclass

import chess
import numpy as np
import numpy.typing as npt

from chessml.encoding import transposition_key
from chessml.net import PolicyValueNet


class Node:
    __slots__ = (
        "children",
        "moves",
        "n",
        "priors",
        "proof",
        "proof_gen",
        "terminal",
        "total",
        "value",
        "w",
    )

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
        # exact value for the side to move, when the subtree has been solved; a
        # terminal's is permanent, a derived one is valid for one search only
        self.proof: float | None = terminal
        self.proof_gen = -1


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
    proofs: npt.NDArray[np.float32]
    pruned: bool = False


class MCTS:
    def __init__(
        self,
        net: PolicyValueNet,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.25,
        proofs: bool = True,
        pruning_factor: float | None = 1.33,
        fpu_scaled: bool = False,
        root_fpu: float | None = None,
    ) -> None:
        self.net = net
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.fpu_scaled = fpu_scaled
        # absolute first-play urgency at the root: with 1.0 every root move is tried
        # once before any is repeated, so no root move can go unvisited
        self.root_fpu = root_fpu
        self.proofs = proofs
        self.pruning_factor = pruning_factor
        self.generation = 0

    def _cannot_be_overtaken(self, root: Node, sims: int, started: float, deadline: float) -> bool:
        # smart pruning: stop once the visit lead exceeds what the remaining time can
        # produce, since the move is chosen by visits and nothing can change it
        if self.pruning_factor is None or deadline == math.inf or len(root.moves) < 2:
            return False
        now = time.monotonic()
        elapsed = now - started
        if elapsed <= 0.0:
            return False
        remaining = (deadline - now) * (sims / elapsed) / self.pruning_factor
        top = np.partition(root.n, -2)[-2:]
        return float(top[1] - top[0]) > remaining

    @staticmethod
    def _proof(node: Node, gen: int) -> float | None:
        if node.proof is None or (node.terminal is None and node.proof_gen != gen):
            return None
        return node.proof

    def _prove(self, path: list[tuple[Node, int]], gen: int) -> None:
        # a proven-lost child proves the parent won; all children proven proves the
        # parent the best of them. Derived proofs are stamped with the search
        # generation because a repetition claim after the game moves on can void them
        for parent, edge in reversed(path):
            child = parent.children[edge]
            if child is None or self._proof(child, gen) is None:
                return
            if self._proof(parent, gen) is not None:
                return
            if child.proof == -1.0:
                parent.proof, parent.proof_gen = 1.0, gen
                continue
            best = -1.0
            for other in parent.children:
                value = None if other is None else self._proof(other, gen)
                if value is None:
                    return
                best = max(best, -value)
            parent.proof, parent.proof_gen = best, gen

    def _fpu(self, node: Node) -> float:
        # first-play urgency, docs/FINDING_fpu.md; Lc0's form scales the reduction by
        # the square root of the policy mass already visited, so a fresh node has none
        running = (node.value + float(node.w.sum())) / (1.0 + node.total)
        reduction = self.fpu_reduction
        if self.fpu_scaled:
            reduction *= math.sqrt(float(node.priors[node.n > 0].sum()))
        return running - reduction

    def _select(self, node: Node, is_root: bool = False) -> int:
        fpu = np.float32(
            self.root_fpu if is_root and self.root_fpu is not None else self._fpu(node)
        )
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
        self.generation += 1
        gen = self.generation
        started = time.monotonic()

        sims = 0
        expanded = 0
        pruned = False
        while sims < max_sims and time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                break
            if node_budget is not None and expanded >= node_budget:
                break
            sim_board = board.copy(stack=False)
            path_keys: list[object] = []
            node = root
            path: list[tuple[Node, int]] = []

            solved = False
            while True:
                proof = self._proof(node, gen) if self.proofs else node.terminal
                if proof is not None:
                    leaf_value = proof
                    solved = True
                    break
                idx = self._select(node, node is root)
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
                    solved = child.terminal is not None
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
            if solved and self.proofs:
                self._prove(path, gen)
                if self._proof(root, gen) is not None:
                    break
            if sims % 32 == 0 and self._cannot_be_overtaken(root, sims, started, deadline):
                pruned = True
                break

        q = np.divide(root.w, root.n, out=np.zeros_like(root.w), where=root.n > 0)
        proofs = np.full(len(root.moves), np.nan, dtype=np.float32)
        for i, child in enumerate(root.children):
            if child is not None and self.proofs:
                proof = self._proof(child, gen)
                if proof is not None:
                    proofs[i] = -proof
        return SearchResult(
            root.moves, root.n.copy(), q, root.value, sims, expanded, root, proofs, pruned
        )
