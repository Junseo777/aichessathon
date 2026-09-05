from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

SIMS_RE = re.compile(r"move \d+: \S+ sims=(\d+) reused=(\d+) pondered=(\d+) q=\S+ t=([\d.]+)s")
FWD_RE = re.compile(r"'forward_ms': ([\d.]+)")
PRE_RE = re.compile(r"opening pre-search sims=(\d+)")

FEN = "r1bq1rk1/1p2ppbp/p1np1np1/2p5/2P4P/2NP1NP1/PP2PPB1/1RBQK2R w K - 0 9"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure this machine's search rate by running one agent directly."
    )
    ap.add_argument("--agent", type=Path, required=True)
    ap.add_argument("--time-left-ms", type=int, default=120_000)
    ap.add_argument("--fen", default=FEN)
    ap.add_argument("--presearch-s", type=float, default=None)
    args = ap.parse_args()

    env = None
    if args.presearch_s is not None:
        import os

        env = dict(os.environ, CHESS_PRESEARCH_S=str(args.presearch_s))

    request = json.dumps({"fen": args.fen, "time_left_ms": args.time_left_ms}) + "\n"
    t0 = time.time()
    proc = subprocess.run(
        [str(PY), str(REPO / "harness" / "runner.py"), str(args.agent.resolve())],
        input=request,
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=env,
    )
    wall = time.time() - t0
    err = proc.stderr

    fwd = FWD_RE.search(err)
    pre = PRE_RE.search(err)
    moves = SIMS_RE.findall(err)

    out = {
        "forward_ms": float(fwd.group(1)) if fwd else None,
        "presearch_sims": int(pre.group(1)) if pre else None,
        "move_sims": int(moves[0][0]) if moves else None,
        "move_reused": int(moves[0][1]) if moves else None,
        "move_pondered": int(moves[0][2]) if moves else None,
        "move_seconds": float(moves[0][3]) if moves else None,
        "wall_s": round(wall, 2),
    }
    print(json.dumps(out))
    if out["move_sims"] is None:
        print("no telemetry parsed; stderr tail:", err[-400:], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
