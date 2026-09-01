import time

import chess
import numpy as np
import pytest

from chessml.tests.weights_fixture import require_weights

require_weights()

import agent  # noqa: E402
from chessml.encoding import transposition_key  # noqa: E402
from chessml.search import Node, SearchResult  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_game() -> None:
    agent._GAME = None


def test_returns_legal_move_and_tracks_history() -> None:
    board = chess.Board()
    first = agent.get_move(board.fen(), 30_000)
    move = chess.Move.from_uci(first)
    assert move in board.legal_moves
    board.push(move)

    reply = next(iter(board.legal_moves))
    board.push(reply)
    second = agent.get_move(board.fen(), 29_000)
    assert chess.Move.from_uci(second) in board.legal_moves

    game = agent._GAME
    assert game is not None
    assert len(game.board.move_stack) == 3
    assert sum(game.key_counts.values()) == 4


def test_rebases_on_unrelated_position() -> None:
    agent.get_move(chess.Board().fen(), 30_000)
    other = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    move = agent.get_move(other.fen(), 30_000)
    assert chess.Move.from_uci(move) in other.legal_moves
    game = agent._GAME
    assert game is not None and len(game.board.move_stack) == 1


def test_crash_falls_back_to_legal_move(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> SearchResult:
        raise RuntimeError("injected")

    monkeypatch.setattr(agent._MCTS, "run", boom)
    board = chess.Board()
    move = agent.get_move(board.fen(), 30_000)
    assert chess.Move.from_uci(move) in board.legal_moves


def test_low_clock_modes_are_fast_and_legal() -> None:
    board = chess.Board()
    for clock_ms in (1_800, 200):
        agent._GAME = None
        start = time.monotonic()
        move = agent.get_move(board.fen(), clock_ms)
        assert chess.Move.from_uci(move) in board.legal_moves
        assert time.monotonic() - start < 0.5


def _result(board: chess.Board, visits: list[int], q: list[float]) -> SearchResult:
    moves = list(board.legal_moves)[: len(visits)]
    priors = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
    return SearchResult(
        moves=moves,
        visits=np.array(visits, dtype=np.float32),
        q=np.array(q, dtype=np.float32),
        root_value=q[0],
        simulations=int(sum(visits)),
        expanded=0,
        root=Node(moves, priors, q[0]),
    )


def test_pick_vetoes_draw_claim_when_winning() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[0.8, 0.7])
    favourite, second = result.moves[0], result.moves[1]

    after = board.copy(stack=False)
    after.push(favourite)
    counts = {transposition_key(after): 2}
    assert agent._pick(board, counts, result) == second
    assert agent._pick(board, {}, result) == favourite


def test_pick_seeks_draw_claim_when_losing() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[-0.8, -0.9])
    second = result.moves[1]

    after = board.copy(stack=False)
    after.push(second)
    counts = {transposition_key(after): 2}
    assert agent._pick(board, counts, result) == second
