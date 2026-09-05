"""One lane of the Stockfish ladder: bracket.py with a net-less (Stockfish wrapper) opponent allowed.

A candidate net plays R2 at the competition clock through the harness, unmodified.
Each agent process is pinned to one core with taskset; its peak resident memory is
read from /proc before the harness kills it. Games cycle through the book openings,
each played with both colour assignments. Runs on the box under the repo's venv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import signal
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

HERE = Path(__file__).resolve().parent
FIELDS = [
    "game", "opening", "cand_colour", "white", "black", "result", "termination",
    "score_cand", "seconds", "plies", "rss_white_kb", "rss_black_kb", "started_utc",
]


def _terminate(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def load_openings(path: Path) -> list[tuple[str, str]]:
    import chess

    rows: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, fen = line.split("\t", 1)
        chess.Board(fen.strip())
        rows.append((name.strip(), fen.strip()))
    if not rows:
        raise SystemExit(f"no openings in {path}")
    return rows


def net_sha256(agent_dir: Path) -> tuple[str, str]:
    model = agent_dir / "weights" / "model.onnx"
    if not model.exists():  # an int8-only agent (ARENA11) ships model.int8.onnx alone
        model = agent_dir / "weights" / "model.int8.onnx"
    if not model.exists():  # a Stockfish wrapper has no net: record its agent.py (rung config) instead
        model = agent_dir / "agent.py"
    return hashlib.sha256(model.read_bytes()).hexdigest(), str(model.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--opponent", type=Path, required=True)
    ap.add_argument("--cand-core", type=int, required=True)
    ap.add_argument("--opp-core", type=int, required=True)
    ap.add_argument("--openings", type=Path, default=HERE / "openings.tsv")
    ap.add_argument("--offset", type=int, default=0, help="index of the first opening")
    ap.add_argument("--first-black", action="store_true", help="candidate is Black in game 1")
    ap.add_argument("--hours", type=float, default=10.0, help="no game starts after this")
    ap.add_argument("--max-games", type=int, default=10**9)
    ap.add_argument("--base-ms", type=int, default=None)
    ap.add_argument("--increment-ms", type=int, default=None)
    ap.add_argument("--repo", type=Path, default=HERE / "repo")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fail-on-failure", action="store_true", help="exit 1 if any game fails")
    args = ap.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    import chess
    import chess.pgn
    import onnxruntime as ort
    from harness.referee import FAILED_TERMINATIONS, play_match
    from harness.rules import BASE_MS, INCREMENT_MS, PLY_CAP
    from harness.sandbox import RUNNER, Agent

    class PinnedAgent(Agent):
        def __init__(self, directory: Path, core: int) -> None:
            super().__init__(
                ["taskset", "-c", str(core), sys.executable, str(RUNNER), str(directory.resolve())]
            )
            self.peak_rss_kb: int | None = None

        def stop(self) -> None:
            process = self._process
            if process is not None:
                try:
                    with open(f"/proc/{process.pid}/status") as status:
                        for line in status:
                            if line.startswith("VmHWM:"):
                                self.peak_rss_kb = int(line.split()[1])
                                break
                except OSError:
                    pass
            super().stop()

    base_ms = BASE_MS if args.base_ms is None else args.base_ms
    inc_ms = INCREMENT_MS if args.increment_ms is None else args.increment_ms
    cand = args.candidate.resolve()
    opp = args.opponent.resolve()
    openings = load_openings(args.openings)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    lock = out / ".lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print(f"{args.lane}: already running (lock held at {lock})", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGHUP, _terminate)

    results = out / "results.csv"
    existing = 0
    if results.exists():
        with results.open() as handle:
            existing = sum(1 for _ in csv.DictReader(handle))

    cand_sha, cand_path = net_sha256(cand)
    opp_sha, opp_path = net_sha256(opp)
    commit_file = repo / "COMMIT"
    meta = {
        "lane": args.lane,
        "candidate": str(cand),
        "candidate_name": cand.name,
        "opponent": str(opp),
        "opponent_name": opp.name,
        "cand_core": args.cand_core,
        "opp_core": args.opp_core,
        "base_ms": base_ms,
        "increment_ms": inc_ms,
        "ply_cap": PLY_CAP,
        "hours": args.hours,
        "openings": str(args.openings.resolve()),
        "n_openings": len(openings),
        "offset": args.offset,
        "first_black": args.first_black,
        "commit": commit_file.read_text().strip() if commit_file.exists() else None,
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "python_chess": chess.__version__,
        "host": platform.node(),
        "candidate_net": {"sha256": cand_sha, "path": cand_path},
        "opponent_net": {"sha256": opp_sha, "path": opp_path},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resumed_from_game": existing,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"{args.lane}: {cand.name} (core {args.cand_core}, {cand_sha[:12]}) vs "
        f"{opp.name} (core {args.opp_core}, {opp_sha[:12]}), {base_ms}+{inc_ms} ms, "
        f"{len(openings)} openings, {args.hours} h, resume at game {existing + 1}",
        flush=True,
    )

    started = time.monotonic()
    deadline = started + args.hours * 3600.0
    failures = 0
    wins = draws = losses = 0
    g = existing
    try:
        with results.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if existing == 0:
                writer.writeheader()
                handle.flush()
            while g < args.max_games and time.monotonic() < deadline:
                pair, colour = divmod(g, 2)
                name, fen = openings[(pair + args.offset) % len(openings)]
                cand_white = (colour == 0) != args.first_black
                white_dir, black_dir = (cand, opp) if cand_white else (opp, cand)
                white_core, black_core = (
                    (args.cand_core, args.opp_core) if cand_white else (args.opp_core, args.cand_core)
                )
                g += 1
                tag = f"g{g:03d}_{name}_{'cw' if cand_white else 'cb'}"
                started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                elapsed = int(time.monotonic() - started)
                print(
                    f"[{time.strftime('%H:%M:%S')} t+{elapsed}s] game {g} {name} "
                    f"white={white_dir.name} black={black_dir.name}",
                    flush=True,
                )
                white = PinnedAgent(white_dir, white_core)
                black = PinnedAgent(black_dir, black_core)
                t0 = time.monotonic()
                error = ""
                try:
                    outcome = play_match(
                        white, black, base_ms, inc_ms, ply_cap=PLY_CAP, start_fen=fen
                    )
                    result, termination, pgn = outcome.result, outcome.termination, outcome.pgn
                except SystemExit:
                    raise
                except Exception:
                    error = traceback.format_exc()
                    result, termination, pgn = "void", "driver_error", ""
                seconds = round(time.monotonic() - t0, 1)

                plies = ""
                if pgn:
                    game = chess.pgn.read_game(io.StringIO(pgn))
                    if game is not None:
                        game.headers["Event"] = f"stockfish ladder {args.lane}"
                        game.headers["Round"] = str(g)
                        game.headers["White"] = white_dir.name
                        game.headers["Black"] = black_dir.name
                        game.headers["Opening"] = name
                        plies = str(len(list(game.mainline_moves())))
                        pgn = str(game)
                    (out / f"{tag}.pgn").write_text(pgn + "\n")

                if result == "draw":
                    score: float | str = 0.5
                elif result == "void":
                    score = "NA"
                else:
                    score = 1.0 if (result == "white") == cand_white else 0.0
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                elif score == 0.0:
                    losses += 1
                failed = termination in FAILED_TERMINATIONS or result == "void"
                failures += int(failed)

                with (out / f"{tag}.log").open("w") as log:
                    log.write(
                        f"lane {args.lane} game {g} opening {name}\nfen {fen}\n"
                        f"white {white_dir} core {white_core}\nblack {black_dir} core {black_core}\n"
                        f"clock {base_ms}+{inc_ms} ms  ply cap {PLY_CAP}\n"
                        f"outcome {result} by {termination} in {seconds}s, plies {plies}\n"
                        f"peak rss kB white {white.peak_rss_kb} black {black.peak_rss_kb}\n"
                    )
                    if error:
                        log.write(f"\ndriver error:\n{error}")
                    log.write(f"\n--- white stderr ---\n{white.stderr_tail}")
                    log.write(f"\n--- black stderr ---\n{black.stderr_tail}\n")

                writer.writerow(
                    {
                        "game": g,
                        "opening": name,
                        "cand_colour": "white" if cand_white else "black",
                        "white": white_dir.name,
                        "black": black_dir.name,
                        "result": result,
                        "termination": termination,
                        "score_cand": score,
                        "seconds": seconds,
                        "plies": plies,
                        "rss_white_kb": white.peak_rss_kb,
                        "rss_black_kb": black.peak_rss_kb,
                        "started_utc": started_utc,
                    }
                )
                handle.flush()
                flag = "  FAILED" if failed else ""
                print(
                    f"    -> {result} by {termination} ({cand.name} scores {score}) in {seconds}s, "
                    f"rss W {(white.peak_rss_kb or 0) // 1024} MB / B {(black.peak_rss_kb or 0) // 1024} MB"
                    f"{flag}",
                    flush=True,
                )
    finally:
        lock.rmdir()

    played = g - existing
    total = wins + draws + losses
    pct = (wins + draws / 2) / total if total else 0.0
    print(
        f"{args.lane} DONE: {played} games in {int(time.monotonic() - started)}s; "
        f"{cand.name} +{wins} ={draws} -{losses} ({pct:.1%}), failures {failures}",
        flush=True,
    )
    return 1 if args.fail_on_failure and failures else 0


if __name__ == "__main__":
    sys.exit(main())
