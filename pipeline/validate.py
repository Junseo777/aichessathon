from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot
import numpy as np

from chessml.encoding import NUM_MOVES, SCALE, decode_move, featurize_int8
from pipeline.labels import pov_value, result_value
from pipeline.positions import iter_game_rows
from pipeline.shard import (
    MULTIPV,
    SOURCE_LICHESS,
    SOURCE_NONE,
    SOURCE_STOCKFISH,
    TIER_MID,
    TIER_TOP,
    Shard,
)

ONE = SCALE


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(name)
        print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""), flush=True)

    def bad(self, name: str, detail: str) -> None:
        self.failed.append(f"{name}: {detail}")
        print(f"  FAIL  {name} -- {detail}", flush=True)

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  note  {text}", flush=True)


def _rng_rows(n: int, sample: int, seed: int = 0) -> np.ndarray:
    if sample >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=sample, replace=False))


def a1_position_before_move(sh: Shard, rows: np.ndarray, r: Results) -> None:
    name = "1 position-before-move (policy decodes to a legal move)"
    bad = 0
    examples: list[str] = []
    for i in rows:
        fen = sh.fen(int(i))
        board = chess.Board(fen)
        flat = int(sh.Y_policy[i])
        probe = board if board.turn == chess.WHITE else board.mirror()
        move = decode_move(flat, probe)
        if move == chess.Move.null() or move not in probe.legal_moves:
            bad += 1
            if len(examples) < 3:
                examples.append(f"row {i} fen={fen} idx={flat} move={move}")
    if bad:
        r.bad(name, f"{bad}/{len(rows)} illegal; e.g. {examples}")
    else:
        r.ok(name, f"{len(rows):,} rows all legal")


def a2_mirror(sh: Shard, rows: np.ndarray, r: Results) -> None:
    name = "2 mirror (own pieces in planes 0-5, no back-rank pawns)"
    bad_pawn = 0
    bad_own = 0
    for i in rows:
        x = sh.X[i]
        if x[0, 0, :].any() or x[0, 7, :].any() or x[6, 0, :].any() or x[6, 7, :].any():
            bad_pawn += 1
        board = chess.Board(sh.fen(int(i)))
        mover = board.turn
        n_own_pawns = int(board.pieces(chess.PAWN, mover).__len__())
        if int(x[0].sum() // ONE) != n_own_pawns:
            bad_own += 1
    if bad_pawn or bad_own:
        r.bad(name, f"{bad_pawn} back-rank-pawn rows, {bad_own} own-pawn-count mismatches")
    else:
        r.ok(name, f"{len(rows):,} rows")


def a3_no_truncation(sh: Shard, rows: np.ndarray, r: Results) -> None:
    name = "3 no truncation (X == featurize_int8 recomputed)"
    bad = 0
    example = ""
    for i in rows:
        board = chess.Board(sh.fen(int(i)))
        want = featurize_int8(board)
        got = np.asarray(sh.X[i])
        if not np.array_equal(want, got):
            bad += 1
            if not example:
                d = np.argwhere(want != got)[:3]
                example = f"row {i} first diffs {d.tolist()}"
    if bad:
        r.bad(name, f"{bad}/{len(rows)} differ; {example}")
    else:
        r.ok(name, f"{len(rows):,} rows bit-identical")


MINIATURE = """[Event "Scholar's mate"]
[White "W"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def a4_value_sign(r: Results) -> None:
    name = "4 value sign (White mates -> +1 at final white-to-move row)"
    game = chess.pgn.read_game(io.StringIO(MINIATURE))
    rows = list(iter_game_rows(game))
    last = rows[-1]
    if last.board.turn != chess.WHITE:
        r.bad(name, "final row is not white-to-move")
        return
    v = result_value("1-0", last.board.turn)
    san = last.board.san(last.move)
    if v != 1 or san != "Qxf7#":
        r.bad(name, f"value={v} move={san}")
        return
    whites = [x for x in rows if x.board.turn == chess.WHITE]
    blacks = [x for x in rows if x.board.turn == chess.BLACK]
    if all(result_value("1-0", x.board.turn) == 1 for x in whites) and all(
        result_value("1-0", x.board.turn) == -1 for x in blacks
    ):
        r.ok(name, f"final move {san}, +1 white rows / -1 black rows")
    else:
        r.bad(name, "value sign inconsistent across rows")


EVAL_PGN = """[Event "eval attachment"]
[White "W"]
[Black "B"]
[Result "1-0"]

1. e4 { [%eval 0.17] } e5 { [%eval 0.24] } 2. Nf3 { [%eval 0.15] } Nc6 { [%eval 0.28] }
3. Bb5 { [%eval 0.22] } Nd4 { [%eval 1.85] } 4. Nxe5 { [%eval 1.79] } 1-0
"""


def a6b_eval_attachment(r: Results) -> None:
    name = "6b engine-value attachment (hand-checked plies, incl. post-swing)"
    game = chess.pgn.read_game(io.StringIO(EVAL_PGN))
    rows = list(iter_game_rows(game))
    checks = [
        (0, None, "startpos carries no eval"),
        (1, 17, "after e4"),
        (5, 22, "before the blunder"),
        (6, 185, "immediately after the 0.22 -> 1.85 swing"),
    ]
    for ply, want_cp, why in checks:
        row = rows[ply]
        got = None if row.eval is None else row.eval.white().score()
        if got != want_cp:
            r.bad(name, f"ply {ply} ({why}): expected {want_cp}, got {got}")
            return
    if pov_value(rows[1].eval, chess.BLACK) >= 0:
        r.bad(name, "black's pov of a white-relative advantage is not negative")
        return
    r.ok(name, "4 hand-checked plies incl. post-swing row")


def a6_engine_value_sign(sh: Shard, r: Results) -> None:
    name = "6 engine-value sign and correlation with outcome"
    src = np.asarray(sh.value_source)
    have = np.flatnonzero(src != SOURCE_NONE)
    if have.size < 1000:
        r.note(f"6 skipped: only {have.size} engine-labelled rows (needs Part C)")
        return
    ve = np.asarray(sh.Y_value_engine[have], dtype=np.float64)
    vo = np.asarray(sh.Y_value[have], dtype=np.float64)
    good = ~np.isnan(ve)
    ve, vo = ve[good], vo[good]
    won = vo > 0
    mean_when_won = float(ve[won].mean()) if won.any() else float("nan")
    corr = float(np.corrcoef(ve, vo)[0, 1])
    if not (mean_when_won > 0.05):
        r.bad(name, f"mean engine value when side-to-move won = {mean_when_won:+.4f}, not positive")
    elif not (corr > 0):
        r.bad(name, f"corr(engine, outcome) = {corr:+.4f}, not positive")
    else:
        r.ok(name, f"mean|won={mean_when_won:+.4f}  corr={corr:+.4f}  n={ve.size:,}")


def a6c_cross_check(sh: Shard, r: Results) -> None:
    name = "6c stockfish-vs-lichess eval agreement (corr > 0.9)"
    src = np.asarray(sh.value_source)
    n_s = int((src == SOURCE_STOCKFISH).sum())
    if n_s == 0:
        n_l = int((src == SOURCE_LICHESS).sum())
        r.note(f"6c not applicable yet: no Stockfish rows (lichess={n_l:,})")
        return
    ours = np.asarray(sh.Y_value_engine, dtype=np.float64)
    lich = np.asarray(sh.Y_value_lichess, dtype=np.float64)
    both = ~np.isnan(ours) & ~np.isnan(lich)
    n = int(both.sum())
    if n < 1000:
        r.note(f"6c only {n} rows labelled by both")
        return
    corr = float(np.corrcoef(ours[both], lich[both])[0, 1])
    if corr > 0.9:
        r.ok(name, f"corr={corr:.4f} over {n:,} rows labelled by both")
    else:
        r.bad(name, f"corr={corr:.4f} over {n:,} rows (needs > 0.9)")


def a6d_column_zero(sh: Shard, r: Results) -> None:
    name = "6d Y_*_engine == Y_*_engine4[:, 0] byte-exact"
    v4 = np.asarray(sh.Y_value_engine4)[:, 0]
    v1 = np.asarray(sh.Y_value_engine)
    p4 = np.asarray(sh.Y_policy_engine4)[:, 0]
    p1 = np.asarray(sh.Y_policy_engine)
    labelled = np.asarray(sh.value_source) == SOURCE_STOCKFISH
    if not labelled.any():
        r.note("6d not applicable yet: no Stockfish rows")
        return
    v_ok = np.array_equal(v4[labelled].view(np.uint16), v1[labelled].view(np.uint16))
    p_ok = np.array_equal(p4[labelled], p1[labelled])
    if v_ok and p_ok:
        r.ok(name, f"{int(labelled.sum()):,} labelled rows")
    else:
        r.bad(name, f"value_match={v_ok} policy_match={p_ok}")


def a6e_multipv_shape(sh: Shard, rows: np.ndarray, r: Results) -> None:
    name = "6e MultiPV rows well-formed (monotone, legal, distinct)"
    src = np.asarray(sh.value_source)
    sel = [int(i) for i in rows if src[i] == SOURCE_STOCKFISH]
    if len(sel) < 100:
        r.note(f"6e not applicable yet: {len(sel)} Stockfish rows in sample")
        return
    bad_mono = bad_legal = bad_dupe = 0
    example = ""
    for i in sel:
        mv = np.asarray(sh.Y_policy_engine4[i])
        vv = np.asarray(sh.Y_value_engine4[i], dtype=np.float64)
        ks = [k for k in range(MULTIPV) if mv[k] >= 0 and not np.isnan(vv[k])]
        if len(ks) > 1 and any(vv[ks[j + 1]] > vv[ks[j]] + 1e-3 for j in range(len(ks) - 1)):
            bad_mono += 1
            if not example:
                example = f"row {i} values {vv.tolist()}"
        present = [int(mv[k]) for k in ks]
        if len(set(present)) != len(present):
            bad_dupe += 1
        board = chess.Board(sh.fen(i))
        probe = board if board.turn == chess.WHITE else board.mirror()
        for idx in present:
            m = decode_move(idx, probe)
            if m == chess.Move.null() or m not in probe.legal_moves:
                bad_legal += 1
                break
    if bad_mono or bad_legal or bad_dupe:
        r.bad(
            name,
            f"{bad_mono} non-monotone, {bad_legal} illegal, {bad_dupe} duplicate "
            f"of {len(sel):,}; {example}",
        )
    else:
        r.ok(name, f"{len(sel):,} labelled rows")


def a6f_human_engine_agreement(sh: Shard, rows: np.ndarray, r: Results) -> None:
    src = np.asarray(sh.value_source)
    sel = np.array([i for i in rows if src[i] == SOURCE_STOCKFISH], dtype=np.int64)
    if sel.size < 100:
        r.note("6f not applicable yet: no Stockfish rows in sample")
        return
    human = np.asarray(sh.Y_policy)[sel]
    lines = np.asarray(sh.Y_policy_engine4)[sel]
    match_any = (lines == human[:, None]).any(axis=1)
    match_top = lines[:, 0] == human
    try:
        tier = np.asarray(sh.tier)[sel]
    except (AttributeError, FileNotFoundError, ValueError):
        tier = None
    parts = [f"all={100 * match_any.mean():.1f}% (top-1 {100 * match_top.mean():.1f}%)"]
    if tier is not None and (tier >= 0).any():
        for t, label in ((TIER_TOP, "top"), (TIER_MID, "mid")):
            m = tier == t
            if m.any():
                parts.append(
                    f"{label}={100 * match_any[m].mean():.1f}% "
                    f"(top-1 {100 * match_top[m].mean():.1f}%, n={int(m.sum()):,})"
                )
    else:
        parts.append("per-tier unavailable (shard predates the tier column)")
    r.note("6f human move within engine top-4: " + "  ".join(parts))


def a7_invariants(sh: Shard, rows: np.ndarray, r: Results) -> None:
    name = "7 plane and label invariants"
    problems: list[str] = []
    planes = np.asarray(sh.X[rows])
    binary = planes[:, 0:17]
    if not np.isin(binary, (0, ONE)).all():
        problems.append("planes 0-16 not all in {0,50}")
    k_own = planes[:, 5].reshape(len(rows), -1).sum(axis=1)
    k_opp = planes[:, 11].reshape(len(rows), -1).sum(axis=1)
    if not (k_own == ONE).all():
        problems.append(f"plane 5 (own king) sums != 50 on {(k_own != ONE).sum()} rows")
    if not (k_opp == ONE).all():
        problems.append(f"plane 11 (opp king) sums != 50 on {(k_opp != ONE).sum()} rows")
    if not (planes[:, 19] == ONE).all():
        problems.append("plane 19 != 50")
    ep = planes[:, 16].reshape(len(rows), -1).sum(axis=1)
    if not np.isin(ep, (0, ONE)).all():
        problems.append("plane 16 (ep) sums not in {0,50}")

    pol = np.asarray(sh.Y_policy[rows])
    if not ((pol >= 0) & (pol < NUM_MOVES)).all():
        problems.append("Y_policy out of [0,4672)")
    val = np.asarray(sh.Y_value[rows])
    if not np.isin(val, (-1, 0, 1)).all():
        problems.append("Y_value not in {-1,0,1}")
    if (np.asarray(sh.Z[rows]) == 0).any():
        problems.append("Z contains zeros")
    pe = np.asarray(sh.Y_policy_engine[rows])
    if not (((pe == -1) | ((pe >= 0) & (pe < NUM_MOVES))).all()):
        problems.append("Y_policy_engine not -1 or a valid index")

    if problems:
        r.bad(name, "; ".join(problems))
    else:
        r.ok(name, f"{len(rows):,} rows")


def a_zobrist(sh: Shard, rows: np.ndarray, r: Results) -> None:
    name = "Z matches the stored FEN"
    bad = 0
    for i in rows[: min(len(rows), 20000)]:
        if int(sh.Z[i]) != chess.polyglot.zobrist_hash(chess.Board(sh.fen(int(i)))):
            bad += 1
    if bad:
        r.bad(name, f"{bad} mismatches")
    else:
        r.ok(name)


def a_split(sh: Shard, r: Results) -> None:
    name = "split is by game and near 2%"
    sp = np.asarray(sh.split)
    frac = float((sp == 1).mean())
    if not (0.005 < frac < 0.06):
        r.bad(name, f"val fraction {frac:.4%} outside sane range")
        return
    r.ok(name, f"val = {frac:.3%} of rows")


def a8_subset(small: Shard, big: Shard, r: Results, sample: int = 200_000) -> None:
    name = "8 subset (small shard is a prefix of the big one)"
    n = min(small.n, sample)
    idx = _rng_rows(small.n, n, seed=7)
    if not np.array_equal(np.asarray(small.Z[idx]), np.asarray(big.Z[idx])):
        r.bad(name, "Z differs at matching row indices")
        return
    if not np.array_equal(np.asarray(small.X[idx]), np.asarray(big.X[idx])):
        r.bad(name, "X differs at matching row indices")
        return
    if not np.array_equal(np.asarray(small.Y_policy[idx]), np.asarray(big.Y_policy[idx])):
        r.bad(name, "Y_policy differs")
        return
    r.ok(name, f"{n:,} rows identical in both")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=Path)
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--heavy-sample", type=int, default=20_000, help="rows for per-row FEN checks")
    ap.add_argument("--superset", type=Path, default=None, help="bigger shard, for assertion 8")
    args = ap.parse_args()

    sh = Shard(args.shard)
    print(f"shard {args.shard}  n={sh.n:,}")
    rows = _rng_rows(sh.n, args.sample, seed=1)
    heavy = _rng_rows(sh.n, args.heavy_sample, seed=2)
    r = Results()

    a1_position_before_move(sh, heavy, r)
    a2_mirror(sh, heavy, r)
    a3_no_truncation(sh, heavy, r)
    a4_value_sign(r)
    a6b_eval_attachment(r)
    a6_engine_value_sign(sh, r)
    a6c_cross_check(sh, r)
    a6d_column_zero(sh, r)
    a6e_multipv_shape(sh, heavy, r)
    a6f_human_engine_agreement(sh, rows, r)
    a7_invariants(sh, rows, r)
    a_zobrist(sh, heavy, r)
    a_split(sh, r)
    if args.superset is not None:
        a8_subset(sh, Shard(args.superset), r)

    print()
    print(f"{len(r.passed)} passed, {len(r.failed)} failed, {len(r.notes)} notes")
    if r.failed:
        print("FAILURES:")
        for f in r.failed:
            print(f"  - {f}")
        return 1
    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
