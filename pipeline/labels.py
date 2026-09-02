from __future__ import annotations

import math

import chess
import chess.engine

_CP_SCALE = 0.00368208

MATE_VALUE = 0.995


def score_to_value(score: chess.engine.Score) -> float:
    mate = score.mate()
    if mate is not None:
        return MATE_VALUE if mate > 0 else -MATE_VALUE
    cp = score.score()
    if cp is None:
        raise ValueError("score is neither mate nor centipawns")
    return 2.0 / (1.0 + math.exp(-_CP_SCALE * cp)) - 1.0


def pov_value(score: chess.engine.PovScore, turn: chess.Color) -> float:
    return score_to_value(score.pov(turn))


def result_value(result: str, turn: chess.Color) -> int:
    if result == "1/2-1/2":
        return 0
    if result == "1-0":
        return 1 if turn == chess.WHITE else -1
    if result == "0-1":
        return 1 if turn == chess.BLACK else -1
    raise ValueError(f"unexpected result {result!r}")
