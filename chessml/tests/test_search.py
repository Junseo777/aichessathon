import time

import chess
import numpy as np
import pytest

from chessml.encoding import transposition_key
from chessml.net import PolicyValueNet, load_fastest
from chessml.search import MCTS, Node
from chessml.tests.weights_fixture import require_weights

WEIGHTS = require_weights()


@pytest.fixture(scope="module")
def net() -> PolicyValueNet:
    return load_fastest(WEIGHTS)[0]


def fresh_counts(board: chess.Board) -> dict[object, int]:
    return {transposition_key(board): 1}


def test_finds_mate_in_one(net: PolicyValueNet) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    mcts = MCTS(net)
    result = mcts.run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=400)
    best = result.moves[int(np.argmax(result.visits))]
    assert best == chess.Move.from_uci("a1a8"), [
        (m.uci(), int(v)) for m, v in zip(result.moves, result.visits, strict=False) if v > 0
    ]


def test_finds_mate_in_one_as_black(net: PolicyValueNet) -> None:
    board = chess.Board("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    mcts = MCTS(net)
    result = mcts.run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=400)
    best = result.moves[int(np.argmax(result.visits))]
    assert best == chess.Move.from_uci("a8a1")


def test_deadline_is_respected(net: PolicyValueNet) -> None:
    board = chess.Board()
    mcts = MCTS(net)
    start = time.monotonic()
    result = mcts.run(board, fresh_counts(board), start + 0.15)
    elapsed = time.monotonic() - start
    assert result.simulations > 0
    assert elapsed < 0.5, f"deadline overrun: {elapsed:.3f}s"


def test_third_occurrence_is_a_terminal_draw(net: PolicyValueNet) -> None:
    board = chess.Board()
    counts = {transposition_key(board): 2}
    node, value = MCTS(net)._expand(board, counts, [])
    assert node.terminal == 0.0 and value == 0.0


def test_fifty_move_clock_is_a_terminal_draw(net: PolicyValueNet) -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 100 70")
    node, value = MCTS(net)._expand(board, {}, [])
    assert node.terminal == 0.0 and value == 0.0


def test_checkmate_and_stalemate_are_terminal(net: PolicyValueNet) -> None:
    mated = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    node, value = MCTS(net)._expand(mated, fresh_counts(mated), [])
    assert node.terminal == -1.0 and value == -1.0

    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    node, value = MCTS(net)._expand(stalemate, fresh_counts(stalemate), [])
    assert node.terminal == 0.0 and value == 0.0


def test_node_arrays_align() -> None:
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    node = Node(moves, np.array([0.7, 0.3], dtype=np.float32))
    assert len(node.children) == len(moves) == len(node.n) == len(node.w)
    assert node.terminal is None
