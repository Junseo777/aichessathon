import chess
import numpy as np

from chessml.encoding import transposition_key
from chessml.net import PolicyValueNet, load_fastest
from chessml.tests.weights_fixture import require_weights

WEIGHTS = require_weights()


def fresh_net() -> PolicyValueNet:
    return load_fastest(WEIGHTS)[0]


def same_evaluation(a: tuple[object, ...], b: tuple[object, ...]) -> None:
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_stackless_boards_are_cached_exactly() -> None:
    net = fresh_net()
    board = chess.Board("r1bq1rk1/2pp1ppp/p1n2n2/1pb1p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9")
    first = net.evaluate(board)
    second = net.evaluate(chess.Board(board.fen()))
    assert net.forwards == 1 and net.hits == 1
    same_evaluation(second, first)


def test_boards_with_history_are_cached() -> None:
    net = fresh_net()
    board = chess.Board()
    board.push_uci("e2e4")
    first = net.evaluate(board)
    same_evaluation(net.evaluate(board), first)
    same_evaluation(net.evaluate(chess.Board(board.fen())), first)
    assert net.forwards == 1 and net.hits == 2


def test_repetition_state_is_part_of_the_key() -> None:
    net = fresh_net()
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert transposition_key(board) == transposition_key(chess.Board())
    fresh = net.evaluate(chess.Board())
    repeated = net.evaluate(board)
    assert net.forwards == 2 and net.hits == 0
    assert repeated[2] != fresh[2]
    same_evaluation(net.evaluate(board), repeated)
    assert net.hits == 1


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


def test_policy_temperature_flattens_the_priors_without_reordering_them() -> None:
    sharp = fresh_net()
    flat = PolicyValueNet(sharp.session, sharp.name, policy_temperature=1.359)
    board = chess.Board("r1bq1rk1/2pp1ppp/p1n2n2/1pb1p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9")
    moves, p1, v1 = sharp.evaluate(board)
    same_moves, p2, v2 = flat.evaluate(board)
    assert same_moves == moves and v2 == v1
    assert p2.max() < p1.max() and p2.min() > p1.min()
    assert int(np.argmax(p2)) == int(np.argmax(p1))
    np.testing.assert_allclose(p2.sum(), 1.0, atol=1e-5)
    unit = PolicyValueNet(sharp.session, sharp.name, policy_temperature=1.0)
    np.testing.assert_array_equal(unit.evaluate(board)[1], p1)
