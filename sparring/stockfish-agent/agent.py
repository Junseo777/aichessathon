"""Harness wrapper around Stockfish at a capped UCI_Elo.

Local sparring only. Stockfish is GPL and is not a legal submission dependency;
this lives outside the aichessathon repo on purpose, exactly as numbfish-agent does.

Strength is capped with UCI_LimitStrength/UCI_Elo (default 2400, override with
SF_ELO). Threads=1 so it gets the same one core the platform gives each agent.

The harness reports only our own remaining clock, so the opponent's is assumed
symmetric -- Stockfish's manager is driven mainly by its own side, and every
budget is additionally capped well inside the clock so it cannot flag.
"""

import os
import sys
import time

import chess
import chess.engine

ENGINE = os.environ.get("SF_BIN", "/opt/homebrew/bin/stockfish")
ELO = int(os.environ.get("SF_ELO", "2400"))
INCREMENT_S = float(os.environ.get("SF_INC_S", "0.5"))

# Never hand the engine the whole clock: the harness allows WATCHDOG_GRACE_MS=500
# beyond the flag, and a lost game on time would be measurement noise, not strength.
SAFETY_MS = 700.0

# One constant token for the life of this process = one game.
_GAME = "harness-game"

_engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
_engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": ELO})
# No atexit hook: the harness SIGKILLs its runners, so it would never fire, and on a
# normal exit quit() races the engine's event-loop thread and stalls. Stockfish exits
# by itself when its stdin pipe closes, which is what happens when the runner dies.


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    usable_s = max(0.05, (time_left_ms - SAFETY_MS) / 1000.0)

    if board.turn == chess.WHITE:
        clocks = {"white_clock": usable_s, "black_clock": usable_s}
    else:
        clocks = {"black_clock": usable_s, "white_clock": usable_s}

    limit = chess.engine.Limit(
        white_inc=INCREMENT_S, black_inc=INCREMENT_S, **clocks
    )
    # A STABLE game token matters: a changing one makes python-chess send
    # `ucinewgame` before every move, which resets Stockfish's time manager so it
    # re-spends its opening-move allocation each time -- roughly double the correct
    # budget per move, which hands the opponent the game on the clock alone.
    t0 = time.time()
    result = _engine.play(board, limit, game=_GAME)
    print(
        "sf: left=%.1fs spent=%.2fs" % (time_left_ms / 1000.0, time.time() - t0),
        file=sys.stderr, flush=True,
    )
    move = result.move
    if move is None:
        move = next(iter(board.legal_moves))
    return move.uci()
