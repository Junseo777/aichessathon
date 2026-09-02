from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import chess.polyglot

from chessml.encoding import encode_move, featurize_int8, mirror_move
from pipeline.labels import pov_value, result_value
from pipeline.positions import iter_game_rows
from pipeline.shard import (
    SOURCE_LICHESS,
    SOURCE_NONE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TIER_MID,
    TIER_TOP,
    ShardWriter,
)

VAL_FRACTION_DENOM = 50
TIER_SHARE = {"top": 0.80, "mid": 0.20}


@dataclass
class Stream:
    label: str
    tier: str
    path: Path
    available: int
    weight: float = 0.0
    emitted: int = 0
    games: int = 0
    with_eval: int = 0
    fh: TextIO | None = None
    done: bool = False

    def open(self) -> TextIO:
        if self.fh is None:
            self.fh = open(self.path, encoding="utf-8", errors="replace")  # noqa: SIM115
        return self.fh

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def game_key(game: chess.pgn.Game) -> bytes:
    h = game.headers
    ident = "|".join(
        [
            h.get("White", ""),
            h.get("Black", ""),
            h.get("UTCDate", h.get("Date", "")),
            h.get("Result", ""),
            h.get("TimeControl", ""),
        ]
    )
    return hashlib.sha1(ident.encode("utf-8", "replace")).digest()


def split_of(key: bytes) -> int:
    return SPLIT_VAL if (int.from_bytes(key[:8], "big") % VAL_FRACTION_DENOM) == 0 else SPLIT_TRAIN


def available_positions(path: Path, tier: str) -> int:
    report = path.parent / "filter_report.json"
    if report.exists():
        data = json.loads(report.read_text())
        n = data.get("tier_positions", {}).get(tier)
        if n:
            return int(n)
    return max(path.stat().st_size, 1)


def assign_weights(streams: list[Stream]) -> None:
    for tier, share in TIER_SHARE.items():
        in_tier = [s for s in streams if s.tier == tier and not s.done]
        total = sum(s.available for s in in_tier)
        for s in in_tier:
            s.weight = share * (s.available / total) if total else 0.0
    live = sum(s.weight for s in streams if not s.done)
    if live > 0:
        for s in streams:
            s.weight = s.weight / live if not s.done else 0.0


def pick(streams: list[Stream], total: int) -> Stream | None:
    best: Stream | None = None
    best_deficit = float("-inf")
    for s in streams:
        if s.done or s.weight <= 0:
            continue
        deficit = s.weight * (total + 1) - s.emitted
        if deficit > best_deficit:
            best, best_deficit = s, deficit
    return best


def build(streams: list[Stream], out: Path, target: int, report_every: int = 1_000_000) -> dict:
    writer = ShardWriter(out)
    assign_weights(streams)
    eval_rows = 0
    games_skipped = 0
    started = time.time()
    next_report = report_every

    try:
        while writer.n < target:
            stream = pick(streams, writer.n)
            if stream is None:
                break
            game = chess.pgn.read_game(stream.open())
            if game is None:
                stream.done = True
                stream.close()
                assign_weights(streams)
                continue
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                games_skipped += 1
                continue

            split = split_of(game_key(game))
            before = writer.n
            for row in iter_game_rows(game):
                if writer.n >= target:
                    break
                board = row.board
                label = mirror_move(row.move) if board.turn == chess.BLACK else row.move
                policy = encode_move(label)
                if policy < 0:
                    continue
                if row.eval is not None:
                    v_lichess = pov_value(row.eval, board.turn)
                    source = SOURCE_LICHESS
                    eval_rows += 1
                    stream.with_eval += 1
                else:
                    v_lichess = float("nan")
                    source = SOURCE_NONE
                writer.append(
                    x=featurize_int8(board),
                    y_policy=policy,
                    y_value=result_value(result, board.turn),
                    z=chess.polyglot.zobrist_hash(board),
                    fen=board.fen(en_passant="fen"),
                    split=split,
                    y_value_engine=v_lichess,
                    y_policy_engine=-1,
                    y_value_lichess=v_lichess,
                    tier=TIER_TOP if stream.tier == "top" else TIER_MID,
                    value_source=source,
                )
            grown = writer.n - before
            if grown:
                stream.emitted += grown
                stream.games += 1

            if writer.n >= next_report:
                next_report += report_every
                rate = writer.n / max(time.time() - started, 1e-9)
                mix = " ".join(
                    f"{s.label}={100 * s.emitted / max(writer.n, 1):.1f}%" for s in streams
                )
                print(f"  {writer.n:,}/{target:,}  {rate:,.0f}/s  {mix}", flush=True)

        tier_positions: dict[str, int] = {}
        for s in streams:
            tier_positions[s.tier] = tier_positions.get(s.tier, 0) + s.emitted
        meta = {
            "target": target,
            "n_samples": writer.n,
            "games_used": sum(s.games for s in streams),
            "games_skipped": games_skipped,
            "tier_positions": tier_positions,
            "tier_pct": {
                t: round(100 * n / max(writer.n, 1), 2) for t, n in tier_positions.items()
            },
            "streams": [
                {
                    "label": s.label,
                    "tier": s.tier,
                    "weight": round(s.weight, 4),
                    "positions": s.emitted,
                    "pct": round(100 * s.emitted / max(writer.n, 1), 2),
                    "games": s.games,
                    "rows_with_eval": s.with_eval,
                    "exhausted": s.done,
                }
                for s in streams
            ],
            "eval_rows": eval_rows,
            "eval_coverage_pct": round(100 * eval_rows / max(writer.n, 1), 3),
            "val_fraction_denom": VAL_FRACTION_DENOM,
            "seconds": round(time.time() - started, 1),
        }
        writer.write_meta(meta)
        return meta
    finally:
        for s in streams:
            s.close()
        writer.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument(
        "--stream",
        action="append",
        required=True,
        metavar="LABEL:TIER:PATH",
        help="e.g. elite:top:/workspace/data/filtered_elite/tier_top.pgn",
    )
    args = ap.parse_args()

    streams: list[Stream] = []
    for spec in args.stream:
        label, tier, path = spec.split(":", 2)
        p = Path(path)
        streams.append(
            Stream(label=label, tier=tier, path=p, available=available_positions(p, tier))
        )

    print("streams:")
    assign_weights(streams)
    for s in streams:
        print(f"  {s.label:22s} tier={s.tier:3s} available={s.available:>12,}  w={s.weight:.4f}")
    print()

    meta = build(streams, args.out, args.target)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
