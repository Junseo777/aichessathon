import os
import time
from collections.abc import Iterator

import chess
import numpy as np
import pytest

from chessml.tests.weights_fixture import require_weights

require_weights()
os.environ.setdefault("CHESS_PRESEARCH_S", "0.5")

import agent  # noqa: E402
from chessml.encoding import transposition_key  # noqa: E402
from chessml.search import Node, SearchResult  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_game() -> Iterator[None]:
    if agent._GAME is not None:
        agent._stop_ponder(agent._GAME)
    agent._GAME = None
    yield
    if agent._GAME is not None:
        agent._stop_ponder(agent._GAME)
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


def test_ponders_between_moves_and_reuses_the_subtree() -> None:
    board = chess.Board()
    first = agent.get_move(board.fen(), 30_000)
    game = agent._GAME
    assert game is not None and game.ponder is not None and game.ponder.is_alive()
    time.sleep(0.4)

    board.push_uci(first)
    reply = next(iter(board.legal_moves))
    board.push(reply)
    agent._stop_ponder(game)
    assert game.ponder is None and game.tree is not None
    assert game.tree.total > 0
    reused = agent._reusable_root(game, reply)
    assert reused is not None or game.tree.children[game.tree.moves.index(reply)] is None

    second = agent.get_move(board.fen(), 29_000)
    assert chess.Move.from_uci(second) in board.legal_moves
    assert agent._GAME is game and game.ponder is not None and game.ponder.is_alive()


def test_ponder_stops_within_a_bounded_time() -> None:
    agent.get_move(chess.Board().fen(), 30_000)
    game = agent._GAME
    assert game is not None and game.ponder is not None
    time.sleep(0.2)
    start = time.monotonic()
    agent._stop_ponder(game)
    assert time.monotonic() - start < 0.5
    assert game.ponder is None and game.ponder_ok


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
    assert agent._GAME is None


def test_low_clock_modes_are_fast_and_legal() -> None:
    board = chess.Board()
    for clock_ms in (1_800, 200):
        if agent._GAME is not None:
            agent._stop_ponder(agent._GAME)
        agent._GAME = None
        start = time.monotonic()
        move = agent.get_move(board.fen(), clock_ms)
        assert chess.Move.from_uci(move) in board.legal_moves
        assert time.monotonic() - start < 0.5


def _result(
    board: chess.Board,
    visits: list[int],
    q: list[float],
    proofs: list[float] | None = None,
    moves: list[chess.Move] | None = None,
    var: list[float] | None = None,
) -> SearchResult:
    if moves is None:
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
        proofs=np.array(proofs if proofs is not None else [np.nan] * len(moves), dtype=np.float32),
        var=None if var is None else np.array(var, dtype=np.float32),
    )


def _extension_setup(
    monkeypatch: pytest.MonkeyPatch, agree_after: int
) -> tuple[chess.Board, SearchResult, list[float]]:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    deadlines: list[float] = []

    def run(*args: object, **kwargs: object) -> SearchResult:
        deadline = float(args[2])  # type: ignore[arg-type]
        deadlines.append(deadline)
        time.sleep(max(0.0, deadline - time.monotonic()))
        if len(deadlines) >= agree_after:
            return _result(board, visits=[100, 50], q=[0.3, 0.1])
        return _result(board, visits=[100, 50], q=[0.1, 0.3])

    monkeypatch.setattr(agent._MCTS, "run", run)
    return board, _result(board, visits=[100, 50], q=[0.1, 0.3]), deadlines


def test_extension_runs_in_slices_until_the_leaders_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "_EXTEND_FACTOR", 2.0)
    board, first, deadlines = _extension_setup(monkeypatch, agree_after=2)
    started = time.monotonic()
    result = agent._extend(board, {}, first, started, 1.0, 60.0)
    assert len(deadlines) == 2 and not agent._leaders_disagree(result)
    assert result.simulations == 450
    assert all(started + 0.2 <= d <= started + 2.0 for d in deadlines)


def test_extension_stops_at_twice_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "_EXTEND_FACTOR", 2.0)
    board, first, deadlines = _extension_setup(monkeypatch, agree_after=99)
    started = time.monotonic() - 1.0
    result = agent._extend(board, {}, first, started, 1.0, 60.0)
    assert deadlines and max(deadlines) <= started + 2.0 + 1e-6
    assert time.monotonic() <= started + 2.1
    assert agent._leaders_disagree(result)


def test_extension_respects_the_clock_guard_and_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board, first, deadlines = _extension_setup(monkeypatch, agree_after=99)
    started = time.monotonic() - 1.0  # the budget of 1 s is spent
    assert agent._extend(board, {}, first, started, 1.0, 60.0) is first
    monkeypatch.setattr(agent, "_EXTEND_FACTOR", 2.0)
    assert agent._extend(board, {}, first, started, 1.0, 1.5) is first
    assert deadlines == []


def test_leaders_disagree_only_among_candidates() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    assert agent._leaders_disagree(_result(board, visits=[100, 20], q=[0.1, 0.3]))
    assert not agent._leaders_disagree(_result(board, visits=[100, 19], q=[0.1, 0.3]))
    assert not agent._leaders_disagree(_result(board, visits=[100, 50], q=[0.3, 0.1]))


def test_pick_by_lower_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    better = _result(board, visits=[100, 20], q=[0.1, 0.3], var=[0.01, 0.01])
    assert agent._pick(board, {}, better) == better.moves[0]
    monkeypatch.setattr(agent, "_LCB_Z", 2.0)
    assert agent._pick(board, {}, better) == better.moves[1]
    few_visits = _result(board, visits=[100, 19], q=[0.1, 0.3], var=[0.01, 0.01])
    assert agent._pick(board, {}, few_visits) == few_visits.moves[0]
    noisy = _result(board, visits=[100, 20], q=[0.1, 0.3], var=[0.01, 1.0])
    assert agent._pick(board, {}, noisy) == noisy.moves[0]


def test_pick_by_lower_bound_still_avoids_a_repetition_when_winning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "_LCB_Z", 2.0)
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 20, 10], q=[0.5, 0.8, 0.4], var=[0.01, 0.01, 0.01])
    after = board.copy(stack=False)
    after.push(result.moves[1])
    assert agent._pick(board, {transposition_key(after): 1}, result) == result.moves[0]


def test_pick_prefers_a_proven_win_over_visits() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[0.6, 0.4], proofs=[np.nan, 1.0])
    assert agent._pick(board, {}, result) == result.moves[1]


def test_pick_avoids_a_proven_loss() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50, 10], q=[0.1, 0.0, 0.0], proofs=[-1.0, np.nan, np.nan])
    assert agent._pick(board, {}, result) == result.moves[1]
    all_lost = _result(board, visits=[100, 50], q=[-0.9, -0.9], proofs=[-1.0, -1.0])
    assert agent._pick(board, {}, all_lost) == all_lost.moves[0]


def test_pick_vetoes_a_proven_win_that_hands_over_a_claim() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[0.6, 0.4], proofs=[np.nan, 1.0])
    after = board.copy(stack=False)
    after.push(result.moves[1])
    assert agent._pick(board, {transposition_key(after): 2}, result) == result.moves[0]


def test_pick_vetoes_draw_claim_when_winning() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[0.8, 0.7])
    favourite, second = result.moves[0], result.moves[1]

    after = board.copy(stack=False)
    after.push(favourite)
    counts = {transposition_key(after): 2}
    assert agent._pick(board, counts, result) == second
    assert agent._pick(board, {}, result) == favourite


def test_pick_avoids_a_second_occurrence_when_winning() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[0.8, 0.7])
    favourite, second = result.moves[0], result.moves[1]

    after = board.copy(stack=False)
    after.push(favourite)
    assert agent._pick(board, {transposition_key(after): 1}, result) == second


def test_pick_avoids_running_down_the_fifty_move_count_when_winning() -> None:
    quiet, pawn = chess.Move.from_uci("d4e4"), chess.Move.from_uci("f2f3")
    drifting = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 97 45")
    result = _result(drifting, visits=[100, 50], q=[0.8, 0.7], moves=[quiet, pawn])
    assert agent._pick(drifting, {}, result) == pawn
    fresh = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    assert agent._pick(fresh, {}, result) == quiet


def test_pick_seeks_a_draw_the_reply_completes_when_losing() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[-0.8, -0.9])
    second = result.moves[1]

    after = board.copy(stack=False)
    after.push(second)
    after.push(next(iter(after.legal_moves)))
    assert agent._pick(board, {transposition_key(after): 2}, result) == second


def test_referee_draws_matches_the_harness_referee() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 92 45")
    counts = {transposition_key(board): 1}
    for uci in ("d4e4", "g7h7", "e4d4", "h7g7", "d4e4", "g7h7", "e4d4", "h7g7"):
        for move in board.legal_moves:
            probe = board.copy()
            probe.push(move)
            referee = probe.outcome(claim_draw=True) is not None
            assert agent._referee_draws(board, counts, move) == referee, (board.fen(), move)
        board.push_uci(uci)
        counts[transposition_key(board)] = counts.get(transposition_key(board), 0) + 1


def test_pick_seeks_draw_claim_when_losing() -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    result = _result(board, visits=[100, 50], q=[-0.8, -0.9])
    second = result.moves[1]

    after = board.copy(stack=False)
    after.push(second)
    counts = {transposition_key(after): 2}
    assert agent._pick(board, counts, result) == second


def test_pick_with_no_visits_uses_priors() -> None:
    board = chess.Board()
    result = _result(board, visits=[0, 0, 0], q=[0.0, 0.0, 0.0])
    result.root.priors[:] = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    assert agent._pick(board, {}, result) == result.moves[1]


def test_opening_presearch_is_adopted_for_white() -> None:
    agent._OPENING_TREE = agent._presearch(0.5)
    tree = agent._OPENING_TREE
    assert tree is not None and tree.total > 0
    assert agent._adopt_opening(chess.Board()) is tree
    assert agent._OPENING_TREE is None


def test_opening_presearch_is_adopted_for_black() -> None:
    agent._OPENING_TREE = agent._presearch(0.5)
    board = chess.Board()
    board.push_uci("e2e4")
    child = agent._adopt_opening(board)
    assert child is not None and child.terminal is None
    assert agent._OPENING_TREE is None


def test_opening_presearch_is_adopted_several_plies_in() -> None:
    agent._OPENING_TREE = agent._presearch(1.0)
    tree = agent._OPENING_TREE
    assert tree is not None
    board = chess.Board()
    node = tree
    for _ in range(2):
        idx = int(np.argmax(node.n))
        child = node.children[idx]
        assert child is not None and child.terminal is None
        board.push(node.moves[idx])
        node = child
    assert agent._adopt_opening(board) is node
    assert agent._OPENING_TREE is None


def test_opening_presearch_ignores_unrelated_positions() -> None:
    agent._OPENING_TREE = agent._presearch(0.2)
    assert agent._adopt_opening(chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")) is None
    assert agent._OPENING_TREE is None
