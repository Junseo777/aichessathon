"""Summarise the Stockfish limit-strength ladder (the ARENA1 method).

usage: ladder_fit.py <lanes dir>      lane dirs from ladder_lane.py, opponent dirs named sf<rung>

Per rung: W-D-L, score, Wilson 95% interval, the rung's own implied rating, colour split,
terminations, mean seconds and plies. For the candidate: mean sims per move and total pondered
sims from its per-move stderr lines (the tails the lane driver kept). Joint rating: maximum
likelihood over every game against the standard logistic curve, draws scored 0.5, with the 95%
profile-likelihood interval (log-likelihood within 1.92 of the maximum).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

FAILED = {"crash", "illegal", "flag", "init", "both_failed", "driver_error"}


def wilson(score: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    d = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / d
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def implied(rung: int, score: float) -> str:
    if score <= 0.0 or score >= 1.0:
        return "unbounded"
    return f"{rung + 400.0 * math.log10(score / (1.0 - score)):.0f}"


def expected(r: float, rung: int) -> float:
    return 1.0 / (1.0 + 10 ** ((rung - r) / 400.0))


def loglik(r: float, games: list[tuple[int, float]]) -> float:
    ll = 0.0
    for rung, s in games:
        e = min(max(expected(r, rung), 1e-12), 1 - 1e-12)
        ll += s * math.log(e) + (1 - s) * math.log(1 - e)
    return ll


def fit(games: list[tuple[int, float]]) -> tuple[float, float, float]:
    lo, hi = 1000.0, 5000.0
    best_r, best_ll = lo, -1e18
    r = lo
    while r <= hi:
        v = loglik(r, games)
        if v > best_ll:
            best_r, best_ll = r, v
        r += 1.0
    thr = best_ll - 1.92
    r_lo = best_r
    while r_lo > lo and loglik(r_lo - 1.0, games) >= thr:
        r_lo -= 1.0
    r_hi = best_r
    while r_hi < hi and loglik(r_hi + 1.0, games) >= thr:
        r_hi += 1.0
    return best_r, r_lo, r_hi


def cand_telemetry(log: Path, colour: str) -> tuple[list[int], int]:
    text = log.read_text(errors="replace")
    parts = re.split(r"^--- (white|black) stderr ---$", text, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        if parts[i] == colour:
            body = parts[i + 1]
            sims = [int(m) for m in re.findall(r"^move \d+: \S+ sims=(\d+)", body, flags=re.M)]
            pondered = sum(int(m) for m in re.findall(r"pondered=(\d+)", body))
            return sims, pondered
    return [], 0


def main(root: Path) -> None:
    lanes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "results.csv").exists())
    if not lanes:
        print(f"no lanes under {root}")
        return
    by_rung: dict[int, list[dict[str, str]]] = defaultdict(list)
    metas: dict[str, dict[str, object]] = {}
    for lane in lanes:
        meta = json.loads((lane / "meta.json").read_text())
        metas[lane.name] = meta
        rung = int(re.sub(r"\D", "", str(meta["opponent_name"])))
        with (lane / "results.csv").open() as handle:
            for row in csv.DictReader(handle):
                row["lane"] = lane.name
                row["dir"] = str(lane)
                by_rung[rung].append(row)

    m0 = next(iter(metas.values()))
    print(f"Stockfish ladder under {root}")
    print(f"candidate {m0['candidate_name']} net {str(m0['candidate_net']['sha256'])[:12]}  "  # type: ignore[index]
          f"clock {m0['base_ms']}+{m0['increment_ms']} ms  host {m0['host']}  harness commit {m0.get('commit')}")
    for name, m in metas.items():
        print(f"  lane {name}: vs {m['opponent_name']} (wrapper {str(m['opponent_net']['sha256'])[:12]}), "  # type: ignore[index]
              f"cores {m['cand_core']}/{m['opp_core']}, offset {m['offset']}, started {m['started_utc']}")
    print()

    all_games: list[tuple[int, float]] = []
    all_sims: list[int] = []
    pondered_total = 0
    failures = 0
    print(f"{'rung':>5} {'games':>5} {'W-D-L':>10} {'score':>7} {'95% Wilson':>13} {'implied':>8}  white / black")
    for rung in sorted(by_rung):
        rows = by_rung[rung]
        valid = [r for r in rows if r["score_cand"] not in ("", "NA")]
        scores = [float(r["score_cand"]) for r in valid]
        n = len(scores)
        failures += sum(1 for r in rows if r["termination"] in FAILED or r["result"] == "void")
        if n == 0:
            print(f"{rung:>5} {0:>5}  no scored games")
            continue
        w = sum(1 for s in scores if s == 1.0)
        d = sum(1 for s in scores if s == 0.5)
        l = n - w - d
        score = sum(scores) / n
        lo, hi = wilson(score, n)
        colour = []
        for c in ("white", "black"):
            sub = [float(r["score_cand"]) for r in valid if r["cand_colour"] == c]
            colour.append(f"{sum(sub)/len(sub):.1%} ({len(sub)})" if sub else "-")
        print(f"{rung:>5} {n:>5} {f'+{w} ={d} -{l}':>10} {score:>7.1%} {f'{lo:.1%}-{hi:.1%}':>13} "
              f"{implied(rung, score):>8}  {colour[0]} / {colour[1]}")
        all_games += [(rung, s) for s in scores]
        for r in valid:
            log = Path(r["dir"]) / f"g{int(r['game']):03d}_{r['opening']}_{'cw' if r['cand_colour']=='white' else 'cb'}.log"
            if log.exists():
                sims, pondered = cand_telemetry(log, r["cand_colour"])
                all_sims += sims
                pondered_total += pondered
    print()
    for rung in sorted(by_rung):
        rows = by_rung[rung]
        terms: dict[str, int] = defaultdict(int)
        for r in rows:
            terms[r["termination"]] += 1
        secs = [float(r["seconds"]) for r in rows if r["seconds"]]
        plies = [int(r["plies"]) for r in rows if r["plies"]]
        print(f"rung {rung}: " + ", ".join(f"{k} {v}" for k, v in sorted(terms.items()))
              + (f"; mean {sum(secs)/len(secs):.0f} s, {sum(plies)/len(plies):.0f} plies" if secs and plies else ""))
    print()
    print(f"failed games: {failures}")
    if all_sims:
        print(f"candidate mean sims/move {sum(all_sims)/len(all_sims):.0f} over {len(all_sims)} logged moves; "
              f"pondered sims total {pondered_total} (0 = ponder off, as on the platform)")
    if all_games:
        r, lo, hi = fit(all_games)
        print(f"\njoint maximum-likelihood rating over {len(all_games)} games: {r:.0f}  "
              f"(95% profile-likelihood {lo:.0f}-{hi:.0f})")
        top = max(by_rung)
        top_scores = [float(x["score_cand"]) for x in by_rung[top] if x["score_cand"] not in ("", "NA")]
        if top_scores and sum(top_scores) / len(top_scores) >= 0.65:
            print(f"note: {sum(top_scores)/len(top_scores):.0%} at the top rung {top} (skill level 18.4 of 20, the depth-19 pick, not "
                  f"full strength): the ladder cannot place the candidate; extend with fixed-node rungs")
        print("scale: Stockfish 18 UCI_LimitStrength rungs on this machine at this clock, the ARENA1 method; "
              "not a FIDE/Lichess rating")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
