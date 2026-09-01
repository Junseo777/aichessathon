import chess
import numpy as np

from chessml.net import PolicyValueNet, load_fastest
from chessml.tests.weights_fixture import require_weights

WEIGHTS = require_weights()


def fresh_net() -> PolicyValueNet:
    return load_fastest(WEIGHTS)[0]


def test_stackless_boards_are_cached_exactly() -> None:
    net = fresh_net()
    board = chess.Board("r1bq1rk1/2pp1ppp/p1n2n2/1pb1p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9")
    first = net.evaluate(board)
    second = net.evaluate(chess.Board(board.fen()))
    assert net.forwards == 1 and net.hits == 1
    assert second[0] == first[0]
    np.testing.assert_array_equal(second[1], first[1])
    assert second[2] == first[2]


def test_boards_with_history_bypass_the_cache() -> None:
    net = fresh_net()
    board = chess.Board()
    board.push_uci("e2e4")
    net.evaluate(board)
    net.evaluate(board)
    assert net.forwards == 2 and net.hits == 0 and not net.cache


def test_clock_planes_are_part_of_the_key() -> None:
    net = fresh_net()
    a = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 4 45")
    b = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 60 45")
    net.evaluate(a)
    net.evaluate(b)
    assert net.forwards == 2 and net.hits == 0


def test_cache_result_matches_uncached_result() -> None:
    net = fresh_net()
    board = chess.Board("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3")
    cached_moves, cached_priors, cached_value = net.evaluate(board)
    net.cache.clear()
    moves, priors, value = net.evaluate(board)
    assert moves == cached_moves and value == cached_value
    np.testing.assert_array_equal(priors, cached_priors)


def test_no_legal_moves_returns_empty_without_a_forward() -> None:
    net = fresh_net()
    mated = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    moves, priors, value = net.evaluate(mated)
    assert moves == [] and priors.shape == (0,) and value == 0.0
    assert net.forwards == 0


def test_cache_clears_when_full() -> None:
    net = fresh_net()
    net.cache_size = 2
    for fen in (
        "8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45",
        "8/5pk1/6p1/8/4K3/8/5PP1/8 w - - 0 45",
        "8/5pk1/6p1/8/2K5/8/5PP1/8 w - - 0 45",
    ):
        net.evaluate(chess.Board(fen))
    assert len(net.cache) == 1
