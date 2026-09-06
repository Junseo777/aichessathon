"""Syzygy endgame tables: the exact result of every position with few pieces.

Chess is solved for positions with up to seven pieces, kings included. The three-
and four-piece tables are 70 files of about 6 MB (the five-piece set is a gigabyte
and stays out), ship in the zip under syzygy/, and are read with python-chess,
which the platform preinstalls. Two uses:

- in the search, a position inside the tables is a terminal node holding its exact
  value, so the net is never asked about it and a trade into a won or drawn ending
  is backed up exactly (WDL: win, draw, loss);
- at the root, when the position itself is inside the tables, the move is chosen
  by distance to zeroing (DTZ): the winning move that forces a capture, pawn move
  or mate soonest, within what the fifty-move counter still allows. Minimaxing
  DTZ makes progress every move, so a table win is always converted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.syzygy

# python-chess WDL, for the side to move: 2 win, 1 cursed win (a win the fifty-move
# rule takes away), 0 draw, -1 blessed loss, -2 loss
_WIN = 2
_LOSS = -2


@dataclass(frozen=True)
class RootMove:
    move: chess.Move
    # for the side to move at the root: 2 a win the fifty-move rule cannot take away,
    # 1 a win it can (cursed, or the counter is too far gone), 0 a draw, -1 a loss the
    # fifty-move rule reaches first, -2 a loss
    outcome: int
    # plies to a capture, pawn move or mate on best play, counted after the move:
    # 0 for a mate in one, 1 for a zeroing move that keeps the win; negative when losing
    dtz: int


class Tablebase:
    def __init__(self, directory: Path, max_pieces: int = 4) -> None:
        self.directory = directory
        self.max_pieces = max_pieces
        self._tables = chess.syzygy.Tablebase()
        self.count = self._tables.add_directory(str(directory))

    def covers(self, board: chess.Board) -> bool:
        return chess.popcount(board.occupied) <= self.max_pieces and not board.castling_rights

    def value(self, board: chess.Board) -> float | None:
        """Exact value for the side to move, or None outside the tables or with no
        legal move, which the search scores itself. The fifty-move counter is ignored:
        a cursed win counts as a draw, a win with the counter far gone as a win."""
        if not self.covers(board) or not any(board.generate_legal_moves()):
            return None
        try:
            wdl = self._tables.probe_wdl(board)
        except KeyError:
            return None
        return 1.0 if wdl == _WIN else -1.0 if wdl == _LOSS else 0.0

    def root_moves(self, board: chess.Board) -> list[RootMove] | None:
        """Every legal move classified by the tables, or None outside them."""
        if not self.covers(board):
            return None
        entries: list[RootMove] = []
        try:
            for move in board.legal_moves:
                after = board.copy(stack=False)
                after.push(move)
                entries.append(self._classify(move, after))
        except KeyError:
            return None
        return entries or None

    def _classify(self, move: chess.Move, after: chess.Board) -> RootMove:
        if after.is_checkmate():
            return RootMove(move, 2, 0)
        if after.is_insufficient_material() or not any(after.generate_legal_moves()):
            return RootMove(move, 0, 0)
        wdl = -self._tables.probe_wdl(after)
        if wdl == 0:
            return RootMove(move, 0, 0)
        clock = after.halfmove_clock
        if clock == 0:
            # a capture or pawn move restarts the count, so the table's verdict is final
            if wdl > 0:
                return RootMove(move, 2 if wdl == _WIN else 1, 1)
            return RootMove(move, -2 if wdl == _LOSS else -1, -1)
        dtz = -self._tables.probe_dtz(after)
        if wdl > 0:
            # the zeroing move must come before the count reaches a hundred; the table's
            # count can be one short, so a ply is kept in hand
            return RootMove(move, 2 if wdl == _WIN and dtz + clock <= 99 else 1, dtz)
        return RootMove(move, -2 if wdl == _LOSS and clock - dtz <= 99 else -1, dtz)


def load_tablebase(directory: Path, max_pieces: int = 4) -> Tablebase | None:
    """The tables under a directory, or None when there are none: the agent then plays
    without them, as the sparring copies without a syzygy/ link do."""
    if not directory.is_dir():
        return None
    tables = Tablebase(directory, max_pieces)
    return tables if tables.count > 0 else None
