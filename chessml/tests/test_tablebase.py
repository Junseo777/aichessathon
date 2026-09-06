import math
from pathlib import Path

import chess
import pytest

from chessml.encoding import transposition_key
from chessml.net import PolicyValueNet, load_fastest
from chessml.search import MCTS
from chessml.tablebase import Tablebase, load_tablebase
from chessml.tests.weights_fixture import require_weights
from harness.package import members

ROOT = Path(__file__).resolve().parents[2]
SYZYGY = ROOT / "syzygy"
WEIGHTS = require_weights()


@pytest.fixture(scope="module")
def tables() -> Tablebase:
    loaded = load_tablebase(SYZYGY)
    if loaded is None:
        pytest.skip(f"no Syzygy tables in {SYZYGY}; see syzygy/SOURCE.md")
    return loaded


@pytest.fixture(scope="module")
def net() -> PolicyValueNet:
    return load_fastest(WEIGHTS)[0]


def entries(tables: Tablebase, fen: str) -> dict[str, tuple[int, int]]:
    classified = tables.root_moves(chess.Board(fen))
    assert classified is not None
    return {entry.move.uci(): (entry.outcome, entry.dtz) for entry in classified}


def test_the_three_and_four_piece_set_is_complete(tables: Tablebase) -> None:
    # 35 endings, each a WDL and a DTZ file
    assert tables.count == 70
    assert tables.max_pieces == 4


def test_value_is_exact_for_the_side_to_move(tables: Tablebase) -> None:
    assert tables.value(chess.Board("8/8/8/4k3/8/8/8/4K2Q w - - 0 1")) == 1.0
    assert tables.value(chess.Board("8/8/8/4k3/8/8/8/4K2Q b - - 0 1")) == -1.0
    assert tables.value(chess.Board("8/8/8/4k3/8/8/8/2B1K1N1 w - - 0 1")) == 1.0
    assert tables.value(chess.Board("8/8/8/4k3/8/8/8/2N1K1N1 w - - 0 1")) == 0.0
    # a cursed win is a draw under the rules the referee applies
    assert tables.value(chess.Board("8/8/8/8/8/2k5/8/K1N1N3 w - - 0 1")) == 0.0


def test_value_is_none_outside_the_tables_or_without_a_move(tables: Tablebase) -> None:
    assert tables.value(chess.Board()) is None
    assert tables.value(chess.Board("8/8/8/3rk3/8/8/8/3QK2R w - - 0 1")) is None
    assert tables.value(chess.Board("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")) is None
    assert tables.value(chess.Board("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1")) is None


def test_root_moves_put_mate_first_and_a_stalemate_at_zero(tables: Tablebase) -> None:
    # queen d1, king f7 against the king on h8: Qh5 and Qh1 mate; Qd3, Qc2 and Qb1
    # stalemate; everything else still wins
    ranked = entries(tables, "7k/5K2/8/8/8/8/8/3Q4 w - - 0 1")
    assert ranked["d1h5"] == (2, 0)
    assert ranked["d1h1"] == (2, 0)
    assert ranked["d1d3"] == (0, 0)
    assert ranked["d1c2"] == (0, 0)
    assert ranked["d1b1"] == (0, 0)
    assert {outcome for outcome, _ in ranked.values()} == {0, 2}
    mates = {"d1h5", "d1h1"}
    assert all(
        dtz >= 2 for uci, (outcome, dtz) in ranked.items() if outcome == 2 and uci not in mates
    )


def test_root_moves_see_the_fifty_move_counter(tables: Tablebase) -> None:
    # KBNvK from the wrong corner takes more than the seven plies left on the counter:
    # every move is a win the rule takes away, except Bh7, which hangs the bishop
    ranked = entries(tables, "7k/8/8/8/8/8/8/KB1N4 w - - 92 1")
    assert max(outcome for outcome, _ in ranked.values()) == 1
    assert ranked["b1h7"] == (0, 0)
    assert all(outcome == 1 for uci, (outcome, _) in ranked.items() if uci != "b1h7")
    fresh = entries(tables, "7k/8/8/8/8/8/8/KB1N4 w - - 0 1")
    assert any(outcome == 2 for outcome, _ in fresh.values())


def test_root_moves_are_none_outside_the_tables(tables: Tablebase) -> None:
    assert tables.root_moves(chess.Board()) is None
    assert tables.root_moves(chess.Board("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")) is None


def test_a_losing_side_ranks_the_longest_resistance_last(tables: Tablebase) -> None:
    # king e1 against king e3 and queen a2: Kd1 and Kf1 are the moves, both lost
    ranked = entries(tables, "8/8/8/8/8/4k3/q7/4K3 w - - 0 1")
    assert set(ranked) == {"e1d1", "e1f1"}
    assert all(outcome == -2 and dtz < 0 for outcome, dtz in ranked.values())


def test_search_backs_up_exact_values_from_the_tables(
    net: PolicyValueNet, tables: Tablebase
) -> None:
    # KQvKR with the queen attacked: taking the rook enters KQvK, a table win; a king
    # move loses the queen into KRvK, a table loss. Every child is inside the tables.
    board = chess.Board("k7/8/8/8/8/8/8/K2Q3r w - - 0 1")
    search = MCTS(net, tablebase=tables, proofs=True, root_fpu=1.0)
    # The losing child is exact whether or not the search reaches it: with proofs on, the
    # root is proven the moment the capture is visited and the search stops there, and
    # which child comes first depends on the net's priors (CI's random-init net ordered
    # them differently from a trained one). So probe that child directly: Black to move
    # wins the queen, a table win for the mover.
    hung = board.copy()
    hung.push(chess.Move.from_uci("a1a2"))
    assert search._expand(hung).terminal == 1.0
    result = search.run(board, {transposition_key(board): 1}, math.inf, max_sims=200)
    take = result.moves.index(chess.Move.from_uci("d1h1"))
    child = result.root.children[take]
    assert child is not None and child.terminal == -1.0
    assert result.q[take] == 1.0
    assert result.proofs[take] == 1.0
    assert result.root.terminal is None
    hang = result.moves.index(chess.Move.from_uci("a1a2"))
    if result.root.children[hang] is not None:
        assert result.q[hang] == -1.0
        assert result.proofs[hang] == -1.0


def test_the_root_is_expanded_without_the_tables(net: PolicyValueNet, tables: Tablebase) -> None:
    board = chess.Board("8/8/8/4k3/8/8/8/4K2Q b - - 0 1")
    result = MCTS(net, tablebase=tables, proofs=False).run(
        board, {transposition_key(board): 1}, math.inf, max_sims=50
    )
    assert result.root.terminal is None
    assert len(result.moves) == len(list(board.legal_moves))
    assert result.simulations == 50


def test_the_tables_ship_in_the_zip() -> None:
    names = {name for _, name in members(ROOT, ("weights", "syzygy"))}
    assert "syzygy/KBNvK.rtbw" in names and "syzygy/KBNvK.rtbz" in names
    assert "syzygy/CHECKSUMS.txt" in names
