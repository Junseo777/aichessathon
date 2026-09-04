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
_START_KEY = transposition_key(chess.Board())
_NET, _MANIFEST = load_fastest(
    Path(__file__).resolve().parent / "weights", policy_temperature=1.359
)
_MCTS = MCTS(
    _NET,
    fpu_reduction=0.33,
    fpu_scaled=True,
    root_fpu=1.0,
    proofs=True,
    pruning_factor=1.33,
    draw_score=0.1,
)
print(f"init: {_MANIFEST}")


def _presearch(seconds: float) -> Node | None:
    if seconds <= 0:
        return None
    result = _MCTS.run(chess.Board(), {_START_KEY: 1}, time.monotonic() + seconds)
    print(f"init: opening pre-search sims={result.simulations}")
    return result.root


_OPENING_TREE: Node | None = _presearch(_PRESEARCH_S)


def _adopt_opening(board: chess.Board) -> Node | None:
    global _OPENING_TREE
    tree = _OPENING_TREE
    if tree is None:
        return None
    _OPENING_TREE = None
    key = transposition_key(board)
    if key == _START_KEY:
        return tree
    start = chess.Board()
    for idx, move in enumerate(tree.moves):
        start.push(move)
        if transposition_key(start) == key:
            child = tree.children[idx]
            return child if child is not None and child.terminal is None else None
        start.pop()
    return None


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
    budget = left / max(14, 46 - move_number) + 0.4
    return max(0.05, min(4.0, budget, left - 1.0))


def _material_for_mover(board: chess.Board) -> int:
    mover = board.turn
    return sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in _PIECE_VALUES.items()
    )


def _hands_over_draw_claim(
    board: chess.Board, key_counts: dict[object, int], move: chess.Move
) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    if after.halfmove_clock >= 100:
        return True
    return key_counts.get(transposition_key(after), 0) >= 2


def _pick(board: chess.Board, key_counts: dict[object, int], result: SearchResult) -> chess.Move:
    order = [int(i) for i in np.argsort(-result.visits)]
    for idx in order:
        if result.proofs[idx] == 1.0 and not _hands_over_draw_claim(
            board, key_counts, result.moves[idx]
        ):
            return result.moves[idx]
    safe = [idx for idx in order if result.proofs[idx] != -1.0]
    if safe:
        order = safe
    if float(result.visits[order].max()) <= 0.0:
        return result.moves[order[int(np.argmax(result.root.priors[order]))]]
    best = order[0]
    q_best = float(result.q[best])

    near_adjudication = board.ply() >= _PLY_CAP - 60
    material = _material_for_mover(board) if near_adjudication else 0
    winning = q_best > 0.3 or (near_adjudication and material > 0)
    losing = q_best < -0.3 or (near_adjudication and material < 0)

    if losing:
        for idx in order:
            if _hands_over_draw_claim(board, key_counts, result.moves[idx]):
                return result.moves[idx]
    if winning:
        for idx in order:
            if not _hands_over_draw_claim(board, key_counts, result.moves[idx]):
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
        deadline = started + _budget_s(time_left_ms, board.fullmove_number)
        inherited = root.total if root is not None else 0
        result = _MCTS.run(board, game.key_counts, deadline, root=root)
        move = _pick(board, game.key_counts, result)
        child = result.root.children[result.moves.index(move)]
        elapsed = time.monotonic() - started
        pondered = game.box.sims
        print(
            f"move {board.fullmove_number}: {move.uci()} sims={result.simulations} "
            f"reused={inherited} pondered={pondered} "
            f"q={result.q[result.moves.index(move)]:+.2f} t={elapsed:.2f}s"
            f"{' pruned' if result.pruned else ''}"
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
