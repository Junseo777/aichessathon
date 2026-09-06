from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TELEMETRY = Path(__file__).resolve().parent / "telemetry"
PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

RESULT_RE = re.compile(r".* vs .*: (\w+) by (\w+)")

# a 120 s + 0.5 s game cannot legitimately run this long; a hang must not cost hours
GAME_TIMEOUT_S = 1200


def load_openings(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, fen = line.split("\t", 1)
        out.append((name.strip(), fen.strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--openings", type=Path, required=True)
    ap.add_argument("--base-ms", type=int, default=120_000)
    ap.add_argument("--increment-ms", type=int, default=500)
    args = ap.parse_args()

    openings = load_openings(args.openings)
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.csv"
    with open(results, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            ["game", "opening", "white", "black", "result", "termination", "score_with", "seconds"]
        )

    started = time.time()
    for n in range(1, args.games + 1):
        g = n + args.offset
        oname, fen = openings[((g - 1) // 2) % len(openings)]
        if g % 2 == 1:
            white, black, wtag, btag = args.candidate, args.reference, "with", "without"
        else:
            white, black, wtag, btag = args.reference, args.candidate, "without", "with"
        tag = f"g{g:03d}_{oname}_{wtag}w"
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        print(f"[{stamp}] game {n}/{args.games} (g{g}) {oname} white={wtag}", flush=True)
        t0 = time.time()
        tel = args.out / "tel" / tag
        tel.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(TELEMETRY) + os.pathsep + env.get("PYTHONPATH", "")
        env["CHESS_TELEMETRY_DIR"] = str(tel)
        timed_out = False
        with open(args.out / f"{tag}.log", "w", encoding="utf-8") as log:
            cmd = [
                str(PY),
                "-m",
                "harness.play",
                "--white",
                str(white),
                "--black",
                str(black),
                "--base-ms",
                str(args.base_ms),
                "--increment-ms",
                str(args.increment_ms),
                "--fen",
                fen,
                "--pgn",
                str(args.out / f"{tag}.pgn"),
            ]
            try:
                subprocess.run(
                    cmd,
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    timeout=GAME_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
        secs = int(time.time() - t0)
        text = (args.out / f"{tag}.log").read_text(encoding="utf-8", errors="replace")
        m = RESULT_RE.search(text)
        if timed_out:
            result, term = "void", "timeout"
        else:
            result, term = (m.group(1), m.group(2)) if m else ("void", "none")
        if (result == "white" and wtag == "with") or (result == "black" and btag == "with"):
            score = "1"
        elif result == "draw":
            score = "0.5"
        elif result == "void":
            score = "NA"
        else:
            score = "0"
        with open(results, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([g, oname, wtag, btag, result, term, score, secs])
        print(f"    -> {result} by {term} (with scores {score}) in {secs}s", flush=True)

    print(f"DONE after {int(time.time() - started)}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
