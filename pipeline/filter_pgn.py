from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

MIN_INITIAL_SECONDS = 180
ELO_DIFF_MAX = 400
MIN_PLIES = 10
ALLOWED_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})

RE_TAG = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$')
RE_TIME = re.compile(r"(?:\d+/)?(\d+)(?:\+\d+)?")
RE_CLK = re.compile(r"\s*\[%clk\s+[^\]]*\]")
RE_EMPTY_COMMENT = re.compile(r"\{\s*\}")
RE_RESULT_TOKEN = re.compile(r"(?:1-0|0-1|1/2-1/2|\*)")
RE_MOVE_NUM = re.compile(r"\d+\.+")
RE_COMMENT = re.compile(r"\{[^}]*\}")
RE_MOVE_TOKEN = re.compile(
    r"(?<!\w)(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBNqrbn])?[+#]?)"
)

KEEP_TAGS = ("Event", "White", "Black", "Result", "WhiteElo", "BlackElo", "TimeControl", "UTCDate")


@dataclass
class Stats:
    seen: int = 0
    kept: int = 0
    dup: int = 0
    reject: dict[str, int] = field(default_factory=dict)
    tier_games: dict[str, int] = field(default_factory=dict)
    tier_positions: dict[str, int] = field(default_factory=dict)
    tier_with_eval: dict[str, int] = field(default_factory=dict)

    def bump(self, why: str) -> None:
        self.reject[why] = self.reject.get(why, 0) + 1


@dataclass
class Game:
    tier: str
    movetext: str
    plies: int
    has_eval: bool
    key: bytes
    tags: dict[str, str]


def tier_of(white: int, black: int) -> str | None:
    weaker = min(white, black)
    if weaker >= 2400:
        return "top"
    if weaker >= 1900:
        return "mid"
    return None


def initial_seconds(tc: str) -> int | None:
    if not tc or tc == "-":
        return None
    m = RE_TIME.match(tc.strip())
    return int(m.group(1)) if m else None


def clean_movetext(movetext: str) -> str:
    s = RE_CLK.sub("", movetext)
    s = RE_EMPTY_COMMENT.sub("", s)
    return " ".join(s.split())


def _move_tokens(movetext: str) -> list[str]:
    s = RE_COMMENT.sub(" ", movetext)
    s = RE_RESULT_TOKEN.sub(" ", s)
    s = RE_MOVE_NUM.sub(" ", s)
    return [t for t in s.split() if RE_MOVE_TOKEN.fullmatch(t)]


def parse_tags(header: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line in header.split("\n"):
        m = RE_TAG.match(line.strip())
        if m:
            tags[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
    return tags


def iter_games(stream: TextIO) -> Iterator[tuple[str, str]]:
    header: list[str] = []
    body: list[str] = []
    in_body = False
    for line in stream:
        if line.startswith("["):
            if in_body:
                yield "".join(header), "".join(body)
                header, body, in_body = [], [], False
            header.append(line)
        elif line.strip():
            in_body = True
            body.append(line)
    if header and body:
        yield "".join(header), "".join(body)


def evaluate(tags: dict[str, str], movetext: str, stats: Stats) -> Game | None:
    try:
        white = int(tags["WhiteElo"])
        black = int(tags["BlackElo"])
    except (KeyError, ValueError):
        stats.bump("elo_missing")
        return None
    if white <= 0 or black <= 0:
        stats.bump("elo_zero")
        return None
    if abs(white - black) > ELO_DIFF_MAX:
        stats.bump("elo_gap")
        return None
    tier = tier_of(white, black)
    if tier is None:
        stats.bump("tier_below_1900")
        return None
    if tags.get("Result", "*") not in ALLOWED_RESULTS:
        stats.bump("result")
        return None
    if tags.get("Variant", "").strip().lower() not in ("", "standard"):
        stats.bump("variant")
        return None
    if tags.get("Termination", "").strip().lower() != "normal":
        stats.bump("termination")
        return None
    tc = tags.get("TimeControl", "").strip()
    if tc:
        secs = initial_seconds(tc)
        if secs is not None and secs < MIN_INITIAL_SECONDS:
            stats.bump("time_control")
            return None

    cleaned = clean_movetext(movetext)
    moves = _move_tokens(cleaned)
    if len(moves) < MIN_PLIES:
        stats.bump("too_short")
        return None

    ident = "|".join(
        [
            tags.get("White", ""),
            tags.get("Black", ""),
            tags.get("UTCDate", tags.get("Date", "")),
            tags.get("Result", ""),
            " ".join(moves[:20]),
        ]
    )
    return Game(
        tier=tier,
        movetext=cleaned,
        plies=len(moves),
        has_eval="%eval" in cleaned,
        key=hashlib.sha1(ident.encode("utf-8", "replace")).digest()[:12],
        tags=tags,
    )


def render(game: Game, source: str) -> str:
    lines = [f'[{k} "{game.tags.get(k, "")}"]' for k in KEEP_TAGS]
    lines.append(f'[Source "{source}"]')
    return "\n".join(lines) + "\n\n" + game.movetext + "\n\n"


def open_source(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.filename.endswith(".pgn"):
                    yield io.TextIOWrapper(z.open(info), encoding="utf-8", errors="replace")
    elif path.suffix == ".zst":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with open(path, "rb") as fh:
            yield io.TextIOWrapper(
                dctx.stream_reader(fh, read_size=1 << 22), encoding="utf-8", errors="replace"
            )
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            yield fh


def build_report(stats: Stats, sources: list[str]) -> dict[str, object]:
    return {
        "seen": stats.seen,
        "kept": stats.kept,
        "duplicates": stats.dup,
        "rejects": dict(sorted(stats.reject.items(), key=lambda kv: -kv[1])),
        "tier_games": stats.tier_games,
        "tier_positions": stats.tier_positions,
        "tier_with_eval": stats.tier_with_eval,
        "eval_yield_pct": {
            t: round(100 * stats.tier_with_eval.get(t, 0) / n, 2)
            for t, n in stats.tier_games.items()
            if n
        },
        "sources": sources,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("/workspace/data/filtered"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N games seen (debug)")
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    writers = {t: open(args.out / f"tier_{t}.pgn", "w", encoding="utf-8") for t in ("top", "mid")}  # noqa: SIM115
    seen: set[bytes] = set()
    stats = Stats()

    try:
        for src in args.sources:
            print(f"--- {src.name} ---", flush=True)
            for stream in open_source(src):
                for header, movetext in iter_games(stream):
                    stats.seen += 1
                    game = evaluate(parse_tags(header), movetext, stats)
                    if game is not None:
                        if game.key in seen:
                            stats.dup += 1
                        else:
                            seen.add(game.key)
                            writers[game.tier].write(render(game, src.name))
                            stats.kept += 1
                            t = game.tier
                            stats.tier_games[t] = stats.tier_games.get(t, 0) + 1
                            stats.tier_positions[t] = stats.tier_positions.get(t, 0) + game.plies
                            if game.has_eval:
                                stats.tier_with_eval[t] = stats.tier_with_eval.get(t, 0) + 1
                    if stats.seen % args.progress_every == 0:
                        print(
                            f"  seen {stats.seen:,} kept {stats.kept:,} "
                            f"top {stats.tier_games.get('top', 0):,} "
                            f"mid {stats.tier_games.get('mid', 0):,} dup {stats.dup:,}",
                            flush=True,
                        )
                    if args.limit and stats.seen >= args.limit:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        for w in writers.values():
            w.close()

    report = build_report(stats, [s.name for s in args.sources])
    (args.out / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
