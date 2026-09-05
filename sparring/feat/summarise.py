"""Score a results.csv (or several): W-D-L, score, 95% interval, implied Elo, colour split, failures."""
import csv, math, sys
rows = []
for path in sys.argv[1:]:
    with open(path) as fh:
        rows += [r for r in csv.DictReader(fh) if r.get("score_with", r.get(list(r)[-2])) not in ("NA", "")]
key = "score_with" if "score_with" in rows[0] else [k for k in rows[0] if k.startswith("score_")][0]
s = [float(r[key]) for r in rows]; n = len(s)
w, d, l = s.count(1.0), s.count(0.5), s.count(0.0)
score = sum(s) / n; var = sum(x * x for x in s) / n - score * score; se = math.sqrt(max(var, 0)) / math.sqrt(n)
elo = lambda p: 400 * math.log10(p / (1 - p)) if 0 < p < 1 else float("inf") * (1 if p >= 1 else -1)
lo, hi = max(score - 1.96 * se, 1e-6), min(score + 1.96 * se, 1 - 1e-6)
fails = sum(1 for r in rows if r["termination"] in ("crash", "flag", "illegal", "init", "both_failed"))
tag = key.replace("score_", "")
print(f"{tag}: {n} games  +{w} ={d} -{l}  score {100*score:.1f}%  (95% {100*lo:.1f}-{100*hi:.1f}%)  elo {elo(score):+.0f} ({elo(lo):+.0f}..{elo(hi):+.0f})  failures {fails}")
for colour in ("white", "black"):
    sub = [float(r[key]) for r in rows if r["white"] == tag] if colour == "white" else [float(r[key]) for r in rows if r["black"] == tag]
    if sub: print(f"  as {colour}: {len(sub)} games, {100*sum(sub)/len(sub):.1f}%  (+{sub.count(1.0)} ={sub.count(0.5)} -{sub.count(0.0)})")
terms = {}
for r in rows: terms[r["termination"]] = terms.get(r["termination"], 0) + 1
print("  terminations:", ", ".join(f"{k} {v}" for k, v in sorted(terms.items(), key=lambda kv: -kv[1])))
