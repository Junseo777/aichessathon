from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import chess
import chess.pgn


@dataclass(frozen=True)
class Row:
    board: chess.Board
    """Position *before* ``move`` was played."""
    move: chess.Move
    """The move played from ``board``."""
    eval: chess.engine.PovScore | None
    """Engine score of ``board`` itself, White-relative, or None."""
    ply: int


def iter_game_rows(game: chess.pgn.Game) -> Iterator[Row]:
    node: chess.pgn.GameNode = game
    board = game.board()
    ply = 0
    while node.variations:
        child = node.variations[0]
        move = child.move
        if move is None:
            break
        if not board.is_legal(move):
            return
        yield Row(board=board.copy(stack=False), move=move, eval=node.eval(), ply=ply)
        board.push(move)
        node = child
        ply += 1
