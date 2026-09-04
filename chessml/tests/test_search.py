import math
import threading
import time

import chess
import numpy as np
import pytest

from chessml.encoding import transposition_key
from chessml.net import PolicyValueNet, load_fastest
from chessml.search import MCTS, Node
from chessml.tests.weights_fixture import needs_trained_net, require_weights

WEIGHTS = require_weights()


@pytest.fixture(scope="module")
def net() -> PolicyValueNet:
    return load_fastest(WEIGHTS)[0]


def fresh_counts(board: chess.Board) -> dict[object, int]:
    return {transposition_key(board): 1}


@needs_trained_net()
def test_finds_mate_in_one(net: PolicyValueNet) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    result = MCTS(net).run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=400)
    best = result.moves[int(np.argmax(result.visits))]
    assert best == chess.Move.from_uci("a1a8")


@needs_trained_net()
def test_finds_mate_in_one_as_black(net: PolicyValueNet) -> None:
    board = chess.Board("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    result = MCTS(net).run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=400)
    best = result.moves[int(np.argmax(result.visits))]
    assert best == chess.Move.from_uci("a8a1")


def test_deadline_is_respected(net: PolicyValueNet) -> None:
    board = chess.Board()
    start = time.monotonic()
    result = MCTS(net).run(board, fresh_counts(board), start + 0.15)
    elapsed = time.monotonic() - start
    assert result.simulations > 0
    assert elapsed < 0.5, f"deadline overrun: {elapsed:.3f}s"


def test_second_occurrence_is_a_draw_in_the_tree(net: PolicyValueNet) -> None:
    board = chess.Board()
    board.push_uci("g1f3")
    repeating = chess.Move.from_uci("g8f6")
    after = board.copy(stack=False)
    after.push(repeating)
    counts = {transposition_key(board): 1, transposition_key(after): 1}
    result = MCTS(net).run(board, counts, time.monotonic() + 30.0, max_sims=300)
    idx = result.moves.index(repeating)
    assert result.visits[idx] > 0
    assert result.q[idx] == 0.0
    assert result.root.children[idx] is None


def test_third_occurrence_is_a_draw_on_the_path(net: PolicyValueNet) -> None:
    board = chess.Board()
    board.push_uci("g1f3")
    repeating = chess.Move.from_uci("g8f6")
    after = board.copy(stack=False)
    after.push(repeating)
    counts = {transposition_key(board): 1, transposition_key(after): 2}
    result = MCTS(net).run(board, counts, time.monotonic() + 30.0, max_sims=300)
    idx = result.moves.index(repeating)
    assert result.visits[idx] > 0
    assert abs(float(result.q[idx])) < 1e-6
    assert result.root.children[idx] is None


def test_fifty_move_clock_is_a_terminal_draw(net: PolicyValueNet) -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 100 70")
    node = MCTS(net)._expand(board)
    assert node.terminal == 0.0 and node.value == 0.0


def test_checkmate_and_stalemate_are_terminal(net: PolicyValueNet) -> None:
    mated = chess.Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")
    node = MCTS(net)._expand(mated)
    assert node.terminal == -1.0

    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    node = MCTS(net)._expand(stalemate)
    assert node.terminal == 0.0


def test_root_reuse_continues_the_subtree(net: PolicyValueNet) -> None:
    board = chess.Board()
    counts = fresh_counts(board)
    mcts = MCTS(net)
    first = mcts.run(board, counts, time.monotonic() + 30.0, max_sims=200)
    idx = int(np.argmax(first.visits))
    child = first.root.children[idx]
    assert child is not None
    visits_before = child.total

    board.push(first.moves[idx])
    counts[transposition_key(board)] = 1
    second = mcts.run(board, counts, time.monotonic() + 30.0, max_sims=100, root=child)
    assert second.root is child
    assert child.total == visits_before + 100
    assert second.simulations == 100


def test_stop_event_interrupts_an_unbounded_search(net: PolicyValueNet) -> None:
    board = chess.Board()
    stop = threading.Event()
    box: dict[str, int] = {}

    def search() -> None:
        result = MCTS(net).run(board, fresh_counts(board), math.inf, stop=stop)
        box["sims"] = result.simulations

    thread = threading.Thread(target=search)
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert box["sims"] > 0


def test_node_budget_bounds_expansion(net: PolicyValueNet) -> None:
    board = chess.Board()
    result = MCTS(net).run(
        board, fresh_counts(board), time.monotonic() + 30.0, max_sims=200, node_budget=5
    )
    assert result.expanded <= 5
    assert result.simulations <= 5


def test_node_arrays_align() -> None:
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    node = Node(moves, np.array([0.7, 0.3], dtype=np.float32), 0.1)
    assert len(node.children) == len(moves) == len(node.n) == len(node.w)
    assert node.terminal is None and node.value == 0.1


def _leaf(value: float, terminal: bool = False) -> Node:
    return Node(
        _two_moves(),
        np.array([0.5, 0.5], dtype=np.float32),
        value,
        terminal=value if terminal else None,
    )


def _two_moves() -> list[chess.Move]:
    return [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]


def test_a_proven_lost_child_proves_the_parent_won() -> None:
    mcts = MCTS.__new__(MCTS)
    root = _leaf(0.1)
    root.children[0] = _leaf(-1.0, terminal=True)
    mcts._prove([(root, 0)], gen=3)
    assert root.proof == 1.0 and root.proof_gen == 3
    assert MCTS._proof(root, 3) == 1.0 and MCTS._proof(root, 4) is None


def test_all_children_proven_takes_the_best_of_them() -> None:
    mcts = MCTS.__new__(MCTS)
    root = _leaf(0.1)
    root.children[0] = _leaf(1.0, terminal=True)
    mcts._prove([(root, 0)], gen=1)
    assert root.proof is None
    root.children[1] = _leaf(0.0, terminal=True)
    mcts._prove([(root, 1)], gen=1)
    assert root.proof == 0.0


def test_derived_proofs_cascade_and_terminals_persist() -> None:
    mcts = MCTS.__new__(MCTS)
    root, mid = _leaf(0.0), _leaf(0.0)
    root.children[1] = mid
    mid.children[0] = _leaf(-1.0, terminal=True)
    mcts._prove([(root, 1), (mid, 0)], gen=7)
    assert mid.proof == 1.0 and root.proof is None
    mid.children[1] = _leaf(1.0, terminal=True)
    mcts._prove([(root, 1), (mid, 1)], gen=7)
    assert root.proof is None
    mated = mid.children[0]
    assert mated is not None and MCTS._proof(mated, 99) == -1.0


@needs_trained_net()
def test_mate_in_two_is_proven_and_stops_the_search(net: PolicyValueNet) -> None:
    board = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
    mcts = MCTS(net)
    result = mcts.run(board, fresh_counts(board), time.monotonic() + 60.0, max_sims=2000)
    assert mcts._proof(result.root, mcts.generation) == 1.0
    assert result.simulations < 2000
    winners = {m.uci() for m, p in zip(result.moves, result.proofs, strict=True) if p == 1.0}
    assert winners & {"a2a7", "b1b7"}


@needs_trained_net()
def test_mate_in_one_is_proven_on_the_first_visit(net: PolicyValueNet) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    result = MCTS(net).run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=400)
    idx = result.moves.index(chess.Move.from_uci("a1a8"))
    assert result.proofs[idx] == 1.0 and result.root.proof == 1.0
    assert result.simulations < 400


@needs_trained_net()
def test_proofs_can_be_switched_off(net: PolicyValueNet) -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    result = MCTS(net, proofs=False).run(
        board, fresh_counts(board), time.monotonic() + 30.0, max_sims=200
    )
    assert result.simulations == 200 and result.root.proof is None
    assert np.isnan(result.proofs).all()


def test_smart_pruning_stops_when_the_lead_cannot_be_overtaken(net: PolicyValueNet) -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    mcts = MCTS(net, proofs=False)
    first = mcts.run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=64)
    assert not first.pruned
    lead = int(np.argmax(first.root.n))
    first.root.n[lead] += 5000.0
    first.root.total += 5000
    second = mcts.run(board, fresh_counts(board), time.monotonic() + 1.0, root=first.root)
    assert second.pruned and second.simulations == 32
    assert int(np.argmax(second.visits)) == lead


def test_smart_pruning_needs_a_deadline_and_can_be_disabled(net: PolicyValueNet) -> None:
    board = chess.Board("8/5pk1/6p1/8/3K4/8/5PP1/8 w - - 0 45")
    mcts = MCTS(net, proofs=False)
    seed = mcts.run(board, fresh_counts(board), time.monotonic() + 30.0, max_sims=64)
    seed.root.n[int(np.argmax(seed.root.n))] += 5000.0
    seed.root.total += 5000
    unbounded = mcts.run(board, fresh_counts(board), math.inf, max_sims=64, root=seed.root)
    assert not unbounded.pruned and unbounded.simulations == 64
    off = MCTS(net, proofs=False, pruning_factor=None)
    plain = off.run(board, fresh_counts(board), time.monotonic() + 1.0, max_sims=64, root=seed.root)
    assert not plain.pruned and plain.simulations == 64


def test_scaled_fpu_reduction_grows_with_visited_policy() -> None:
    node = Node(_two_moves(), np.array([0.64, 0.36], dtype=np.float32), 0.2)
    constant = MCTS.__new__(MCTS)
    constant.fpu_reduction, constant.fpu_scaled = 0.25, False
    scaled = MCTS.__new__(MCTS)
    scaled.fpu_reduction, scaled.fpu_scaled = 0.25, True
    assert constant._fpu(node) == pytest.approx(0.2 - 0.25)
    assert scaled._fpu(node) == pytest.approx(0.2)
    node.n[0], node.w[0], node.total = 1.0, 0.1, 1
    running = (0.2 + 0.1) / 2
    assert constant._fpu(node) == pytest.approx(running - 0.25)
    assert scaled._fpu(node) == pytest.approx(running - 0.25 * 0.8)


def test_absolute_root_fpu_visits_every_root_move(net: PolicyValueNet) -> None:
    board = chess.Board()
    sims = len(list(board.legal_moves)) * 6
    every = MCTS(net, proofs=False, root_fpu=1.0).run(
        board, fresh_counts(board), time.monotonic() + 30.0, max_sims=sims
    )
    assert (every.visits > 0).all()
    plain = MCTS(net, proofs=False).run(
        board, fresh_counts(board), time.monotonic() + 30.0, max_sims=sims
    )
    assert (plain.visits == 0).any()


def test_draw_score_makes_a_repetition_worth_less_when_ahead(net: PolicyValueNet) -> None:
    board = chess.Board()
    board.push_uci("g1f3")
    repeating = chess.Move.from_uci("g8f6")
    after = board.copy(stack=False)
    after.push(repeating)
    counts = {transposition_key(board): 1, transposition_key(after): 2}
    mcts = MCTS(net, proofs=False, draw_score=0.1)
    ahead = mcts._expand(board)
    ahead.value = 0.9
    result = mcts.run(board, counts, time.monotonic() + 30.0, max_sims=200, root=ahead)
    idx = result.moves.index(repeating)
    assert result.visits[idx] > 0 and result.q[idx] == pytest.approx(-0.1, abs=1e-5)
    behind = mcts._expand(board)
    behind.value = -0.9
    result = mcts.run(board, counts, time.monotonic() + 30.0, max_sims=200, root=behind)
    idx = result.moves.index(repeating)
    assert result.visits[idx] > 0 and result.q[idx] == pytest.approx(0.1, abs=1e-5)
    level = mcts._expand(board)
    level.value = 0.0
    result = mcts.run(board, counts, time.monotonic() + 30.0, max_sims=200, root=level)
    assert abs(float(result.q[result.moves.index(repeating)])) < 1e-6
