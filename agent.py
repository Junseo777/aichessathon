import time
import traceback
from pathlib import Path

import chess
import numpy as np

from chessml.net import load_fastest
from chessml.search import MCTS, SearchResult, transposition_key

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_PLY_CAP = 300
_NET, _MANIFEST = load_fastest(Path(__file__).resolve().parent / "weights")
_MCTS = MCTS(_NET)
print(f"init: {_MANIFEST}")


class _Game:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.key_counts: dict[object, int] = {transposition_key(board): 1}

    def record(self) -> None:
        key = transposition_key(self.board)
        self.key_counts[key] = self.key_counts.get(key, 0) + 1


_GAME: _Game | None = None


def _sync(fen: str) -> _Game:
    global _GAME
    received_key = transposition_key(chess.Board(fen))
    game = _GAME
    if game is not None:
        board = game.board
        for move in board.legal_moves:
            board.push(move)
            if transposition_key(board) == received_key:
                game.record()
                return game
            board.pop()
    game = _Game(chess.Board(fen))
    _GAME = game
    return game


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
    best = order[0]
    q_best = float(result.q[best]) if result.visits[best] > 0 else result.root_value

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
    game = _sync(fen)
    board = game.board

    if time_left_ms < 250:
        move = next(iter(board.legal_moves))
    elif time_left_ms < 2000:
        moves, priors, _ = _NET.evaluate(board)
        move = moves[int(np.argmax(priors))]
    else:
        deadline = started + _budget_s(time_left_ms, board.fullmove_number)
        result = _MCTS.run(board, game.key_counts, deadline)
        move = _pick(board, game.key_counts, result)
        elapsed = time.monotonic() - started
        print(
            f"move {board.fullmove_number}: {move.uci()} sims={result.simulations} "
            f"q={result.q[result.moves.index(move)]:+.2f} t={elapsed:.2f}s"
        )

    game.board.push(move)
    game.record()
    return move.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    try:
        return _play(fen, time_left_ms)
    except Exception:
        traceback.print_exc()
        global _GAME
        _GAME = None
        return next(iter(chess.Board(fen).legal_moves)).uci()
