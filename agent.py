import math
import os
import threading
import time
import traceback
from pathlib import Path

import chess
import numpy as np

from chessml.encoding import transposition_key
from chessml.net import load_fastest
from chessml.search import MCTS, Node, SearchResult

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_PLY_CAP = 300
_PONDER_NODE_BUDGET = 100_000
_PONDER_JOIN_S = 2.0
_PRESEARCH_S = float(os.environ.get("CHESS_PRESEARCH_S", "5"))
# clock: left / (horizon - move) with the divisor floored, and at least the floor while
# the clock is above 15 s; rated rounds 1 to 14 ran 46 / 14 / no floor and hit 16 s by move 45
_BUDGET_HORIZON = 60
_BUDGET_DIVISOR_FLOOR = 20
_BUDGET_FLOOR_S = 1.0
_BUDGET_FLOOR_ABOVE_S = 15.0
# pick: a root move with this share of the top visits is a candidate; the search runs on
# to _EXTEND_FACTOR x budget while the visit leader is not the best-q candidate, and
# _LCB_Z picks by q minus that many standard errors instead of by visits
_CANDIDATE_SHARE = 0.2
_EXTEND_FACTOR: float | None = None
_LCB_Z: float | None = None
_START_KEY = transposition_key(chess.Board())
_NET, _MANIFEST = load_fastest(Path(__file__).resolve().parent / "weights")
_MCTS = MCTS(_NET, fpu_reduction=0.25, proofs=False, pruning_factor=1.33)
print(f"init: {_MANIFEST}")


def _presearch(seconds: float) -> Node | None:
    if seconds <= 0:
        return None
    result = _MCTS.run(chess.Board(), {_START_KEY: 1}, time.monotonic() + seconds)
    print(f"init: opening pre-search sims={result.simulations}")
    return result.root


_OPENING_TREE: Node | None = _presearch(_PRESEARCH_S)


_ADOPT_NODE_LIMIT = 4096


def _find_in_tree(tree: Node, key: object, limit: int = _ADOPT_NODE_LIMIT) -> Node | None:
    # depth-first walk from the standard start for the position received; rated games
    # begin from curated positions several plies in, so one ply is never enough
    board = chess.Board()
    budget = limit

    def walk(node: Node) -> Node | None:
        nonlocal budget
        for idx, move in enumerate(node.moves):
            child = node.children[idx]
            if child is None or child.terminal is not None or budget <= 0:
                continue
            budget -= 1
            board.push(move)
            found = child if transposition_key(board) == key else walk(child)
            board.pop()
            if found is not None:
                return found
        return None

    return walk(tree)


def _adopt_opening(board: chess.Board) -> Node | None:
    global _OPENING_TREE
    tree = _OPENING_TREE
    if tree is None:
        return None
    _OPENING_TREE = None
    key = transposition_key(board)
    if key == _START_KEY:
        return tree
    return _find_in_tree(tree, key)


class _PonderResult:
    __slots__ = ("root", "sims")

    def __init__(self) -> None:
        self.root: Node | None = None
        self.sims = 0


class _Game:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.key_counts: dict[object, int] = {transposition_key(board): 1}
        self.tree: Node | None = None
        self.ponder: threading.Thread | None = None
        self.stop = threading.Event()
        self.box = _PonderResult()
        self.ponder_ok = True

    def record(self) -> None:
        key = transposition_key(self.board)
        self.key_counts[key] = self.key_counts.get(key, 0) + 1


_GAME: _Game | None = None


def _ponder(
    board: chess.Board,
    key_counts: dict[object, int],
    root: Node | None,
    stop: threading.Event,
    box: _PonderResult,
) -> None:
    try:
        result = _MCTS.run(
            board,
            key_counts,
            math.inf,
            max_sims=1_000_000_000,
            root=root,
            stop=stop,
            node_budget=_PONDER_NODE_BUDGET,
        )
        box.root = result.root
        box.sims = result.simulations
    except Exception:
        traceback.print_exc()
        box.root = None


def _start_ponder(game: _Game, root: Node | None) -> None:
    if not game.ponder_ok:
        return
    game.stop = threading.Event()
    game.box = _PonderResult()
    thread = threading.Thread(
        target=_ponder,
        args=(game.board.copy(), game.key_counts, root, game.stop, game.box),
        daemon=True,
    )
    game.ponder = thread
    thread.start()


def _stop_ponder(game: _Game) -> None:
    thread = game.ponder
    if thread is None:
        return
    game.stop.set()
    thread.join(timeout=_PONDER_JOIN_S)
    game.ponder = None
    if thread.is_alive():
        game.ponder_ok = False
        game.tree = None
        print("ponder: thread did not stop; pondering disabled for this game")
        return
    game.tree = game.box.root


def _sync(fen: str) -> tuple[_Game, chess.Move | None]:
    global _GAME
    received_key = transposition_key(chess.Board(fen))
    game = _GAME
    if game is not None:
        board = game.board
        for move in board.legal_moves:
            board.push(move)
            if transposition_key(board) == received_key:
                game.record()
                return game, move
            board.pop()
    game = _Game(chess.Board(fen))
    _GAME = game
    return game, None


def _reusable_root(game: _Game, opponent_move: chess.Move | None) -> Node | None:
    tree = game.tree
    if opponent_move is None or tree is None or tree.terminal is not None:
        return None
    try:
        idx = tree.moves.index(opponent_move)
    except ValueError:
        return None
    child = tree.children[idx]
    if child is None or child.terminal is not None:
        return None
    return child


def _budget_s(time_left_ms: int, move_number: int) -> float:
    left = time_left_ms / 1000.0
    budget = left / max(_BUDGET_DIVISOR_FLOOR, _BUDGET_HORIZON - move_number) + 0.4
    if left > _BUDGET_FLOOR_ABOVE_S:
        budget = max(budget, _BUDGET_FLOOR_S)
    return max(0.05, min(4.0, budget, left - 1.0))


def _candidates(result: SearchResult, order: list[int]) -> list[int]:
    top = float(result.visits[order[0]])
    return [idx for idx in order if float(result.visits[idx]) >= _CANDIDATE_SHARE * top]


def _leaders_disagree(result: SearchResult) -> bool:
    order = [int(i) for i in np.argsort(-result.visits)]
    best_q = max(_candidates(result, order), key=lambda idx: float(result.q[idx]))
    return best_q != order[0]


def _extend(
    board: chess.Board,
    key_counts: dict[object, int],
    result: SearchResult,
    started: float,
    budget: float,
    left: float,
) -> SearchResult:
    # in slices of a quarter budget, so the search stops as soon as the leaders agree;
    # never past the clock guard the budget itself respects
    if _EXTEND_FACTOR is None:
        return result
    limit = started + min(_EXTEND_FACTOR * budget, left - 1.0)
    step = max(0.1, budget / 4.0)
    while _leaders_disagree(result):
        now = time.monotonic()
        if now + 0.05 >= limit:
            break
        more = _MCTS.run(board, key_counts, min(now + step, limit), root=result.root)
        more.simulations += result.simulations
        result = more
    return result


def _lower_bound_best(result: SearchResult, order: list[int]) -> int:
    # KataGo's selection: the best lower confidence bound on q among the candidates
    if _LCB_Z is None:
        return order[0]
    var = result.var if result.var is not None else np.zeros_like(result.q)
    bound = result.q - _LCB_Z * np.sqrt(var / np.maximum(result.visits, 1.0))
    return max(_candidates(result, order), key=lambda idx: float(bound[idx]))


def _material_for_mover(board: chess.Board) -> int:
    mover = board.turn
    return sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in _PIECE_VALUES.items()
    )


def _referee_draws(board: chess.Board, key_counts: dict[object, int], move: chess.Move) -> bool:
    # the referee ends the game once the side to move could claim: a third occurrence
    # or the fifty-move count reached by this move, or reachable by any reply to it
    after = board.copy(stack=False)
    after.push(move)
    clock = after.halfmove_clock
    if clock >= 100 or key_counts.get(transposition_key(after), 0) >= 2:
        return True
    for reply in after.legal_moves:
        if clock >= 99 and not after.is_zeroing(reply):
            return True
        after.push(reply)
        repeated = key_counts.get(transposition_key(after), 0) >= 2
        after.pop()
        if repeated:
            return True
    return False


def _repeats(board: chess.Board, key_counts: dict[object, int], move: chess.Move) -> bool:
    # a second occurrence, or a fifty-move count the opponent can run down, lets a
    # shuffling opponent force the referee's claim two plies later
    after = board.copy(stack=False)
    after.push(move)
    return after.halfmove_clock >= 98 or key_counts.get(transposition_key(after), 0) >= 1


def _pick(board: chess.Board, key_counts: dict[object, int], result: SearchResult) -> chess.Move:
    order = [int(i) for i in np.argsort(-result.visits)]
    for idx in order:
        if result.proofs[idx] == 1.0 and not _referee_draws(board, key_counts, result.moves[idx]):
            return result.moves[idx]
    safe = [idx for idx in order if result.proofs[idx] != -1.0]
    if safe:
        order = safe
    if float(result.visits[order].max()) <= 0.0:
        return result.moves[order[int(np.argmax(result.root.priors[order]))]]
    best = _lower_bound_best(result, order)
    order.remove(best)
    order.insert(0, best)
    q_best = float(result.q[best])

    near_adjudication = board.ply() >= _PLY_CAP - 60
    material = _material_for_mover(board) if near_adjudication else 0
    winning = q_best > 0.3 or (near_adjudication and material > 0)
    losing = q_best < -0.3 or (near_adjudication and material < 0)

    if losing:
        for idx in order:
            if _referee_draws(board, key_counts, result.moves[idx]):
                return result.moves[idx]
    if winning:
        for idx in order:
            if not _repeats(board, key_counts, result.moves[idx]):
                return result.moves[idx]
    return result.moves[best]


def _play(fen: str, time_left_ms: int) -> str:
    started = time.monotonic()
    if _GAME is not None:
        _stop_ponder(_GAME)
    game, opponent_move = _sync(fen)
    board = game.board
    if opponent_move is not None:
        root = _reusable_root(game, opponent_move)
    else:
        root = _adopt_opening(board)
    child: Node | None = None

    if time_left_ms < 250:
        move = next(iter(board.legal_moves))
    elif time_left_ms < 2000:
        moves, priors, _ = _NET.evaluate(board)
        move = moves[int(np.argmax(priors))]
    else:
        budget = _budget_s(time_left_ms, board.fullmove_number)
        inherited = root.total if root is not None else 0
        result = _MCTS.run(board, game.key_counts, started + budget, root=root)
        searched = result.simulations
        result = _extend(board, game.key_counts, result, started, budget, time_left_ms / 1000.0)
        move = _pick(board, game.key_counts, result)
        child = result.root.children[result.moves.index(move)]
        elapsed = time.monotonic() - started
        pondered = game.box.sims
        print(
            f"move {board.fullmove_number}: {move.uci()} sims={result.simulations} "
            f"reused={inherited} pondered={pondered} "
            f"q={result.q[result.moves.index(move)]:+.2f} t={elapsed:.2f}s"
            f"{' pruned' if result.pruned else ''}"
            f"{' extended' if result.simulations > searched else ''}"
        )

    game.board.push(move)
    game.record()
    _start_ponder(game, child)
    return move.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    global _GAME
    try:
        return _play(fen, time_left_ms)
    except Exception:
        traceback.print_exc()
        try:
            if _GAME is not None:
                _stop_ponder(_GAME)
        except Exception:
            traceback.print_exc()
        _GAME = None
        return next(iter(chess.Board(fen).legal_moves)).uci()
