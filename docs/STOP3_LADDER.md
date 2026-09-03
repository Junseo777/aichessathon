# STOP AND REPORT #3 — Stockfish-limit-strength ladder, R0

Status report on where R0 (the 8M-shard checkpoint currently in `weights/`)
sits on the same ladder the reference project used for its 2,474 figure.
Written to be read cold: everything needed to act on it is below, with no
reference to prior chat.

Date: 2026-09-03. Box: local, Apple M1 (8 physical / 8 logical cores), macOS
26.5.2. Opponent: Stockfish 18 (Homebrew), `Threads=1`, `UCI_LimitStrength`.
Harness: `harness.play` at the competition clock, 120 s + 0.5 s, 300-ply cap
— `harness/rules.py` defaults, unmodified. Agent under test: `agent.py` at
HEAD (`698eed8`), i.e. R0 (`d96 x b12`, 1,381,050 params, EMA weights,
fp32 ONNX) exactly as it ships in `weights/model.onnx`.

Stockfish is driven through `sparring/stockfish-agent/agent.py`, outside this
repo on purpose — GPL, not a legal submission dependency, same quarantine as
`sparring/numbfish-agent`. `Hash=64`, single stable `game` token per process
(see §2 for why that matters), `Threads=1` so it gets the one core the
platform gives each side.

---

## 1. The number

Five rungs, 112 games total, round-robin so an early stop leaves every rung
equally sampled (it didn't need to: the bracket ran to completion inside the
scheduled window).

| rung | games | score | W–D–L | checkmate | 3-fold | stalemate |
|------|-------|-------|-------|-----------|--------|-----------|
| 2400 | 24 | 89.6% | +20 =3 −1 | 21 | 3 | 0 |
| 2500 | 24 | 85.4% | +19 =3 −2 | 21 | 2 | 1 |
| 2600 | 24 | 89.6% | +19 =5 −0 | 19 | 5 | 0 |
| 2800 | 20 | 52.5% | +4 =13 −3 | 7 | 12 | 1 |
| 3000 | 20 | 40.0% | +3 =10 −7 | 10 | 9 | 1 |

Maximum-likelihood logistic fit over all five rungs jointly:

**R0 = 2868, 95% likelihood interval 2786–2954.**

Zero games failed — no crash, flag, illegal move, or init timeout on either
side across 112 games. 8 PGNs sampled at random and replayed through
`python-chess`: all legal, move-for-move, from file.

The reference's own figure on this ladder, from `docs/DECISIONS.md`:
**116k hero, 2,474.** R0 is +394 by the same method, same rung set, same
harness. This is the answer to `docs/DECISIONS.md`'s open item — *"whether we
are actually at least as good as the reference"* — and it agrees with the
independent measurement: R0 scored 93.8% (+7 =1 −0, 8 games) head-to-head
against `baselines/reference-hero` (byte-identical `agent.py`, only the net
differs), which is nominally ~+470 on its own. Two unrelated measurements
landing within 80 points of each other is the strongest thing in this report.

## 2. A bug that cost the first run, and why the number above isn't it

The first 2400-only pass (8 games, discarded) went 8–0 and was not trusted on
sight — an 8-point spread against a labelled-2400 opponent, before this
ladder existed to calibrate what that should look like, was reason enough to
stop and check rather than report it. Two independent problems were found and
both are fixed in the artifacts this report is built from:

- **`ucinewgame` every move.** The wrapper's first version passed
  `game=object()` to `engine.play`, a fresh token each call. `python-chess`
  sends `ucinewgame` whenever the token changes, which resets Stockfish's time
  manager — it re-spent its opening-move allocation on every move instead of
  budgeting across the game. Traced directly: first two moves cost 17.7 s and
  18.6 s of a 120 s clock; by move 18 it had 5.2 s left and was moving in
  under a second for the rest of the game. Its own eval shows the effect —
  roughly level (−29 cp) after two moves, sliding to −429 as the clock ran
  out, not from being outplayed. Fixed with one stable `game` token held for
  the process lifetime (§ agent.py, `_GAME`), confirmed by direct A/B: 6.5 s
  vs 3.2 s median think time on move 2 of an otherwise identical call.
- **CPU contention.** The corrected rerun was launched, and — separately —
  the bracket script below was launched twice by mistake, so for roughly
  three minutes three `harness.play` processes and three Stockfish processes
  were competing for 8 cores at once. All three were killed and every game
  produced during that window was discarded before the run in §1 started.

The bracket in §1 ran as a single locked process (`mkdir`-based lock,
refuses a second instance) with nothing else competing for CPU, and its 2400
rung — re-measured clean at 24 games instead of the discarded 8 — landed at
89.6%, not 100%. That is the number to trust.

## 3. What this scale is and isn't

**Is:** the same maximum-likelihood-against-calibrated-rungs method the
reference used for 2,474, so R0's 2868 is directly comparable to it, and nice
convergence with the independent reference-hero result above says the method
is measuring something real.

**Isn't:** a FIDE or Lichess rating. `docs/DECISIONS.md` already flags that
`UCI_LimitStrength` rungs are "known to be poorly calibrated against
human/online Elo," and this run shows the compression directly — **2400 and
2600 both scored 89.6%**, i.e. adjacent-by-200 rungs were statistically
indistinguishable here, while the real signal sits between 2600 (89.6%) and
2800 (52.5%). Per-rung implied ratings (each rung's own score read against
the standard logistic curve, ignoring the other four) scatter from 2774 to
2974 — that spread, not the point estimate, is the honest error bar. Treat
2868 as "roughly where R0 sits on this one ladder," not a chess rating, and
not something to chain onto Numbfish's Lichess-bot 2,300 or the competition
ladder — `docs/DECISIONS.md` §1 already rules that chain out and this data
doesn't change that.

## 4. Two things the data flags, unresolved

**Colour asymmetry.** Bot as White: 80.4% pooled across all five rungs
(56 games). Bot as Black: 66.1% (56 games) — a 14-point gap where normal
first-move advantage is 3–4 points. Per rung: White led Black at every rung
except 2600 (91.7/87.5, 95.8/75.0, 91.7/87.5, 55.0/50.0, 55.0/25.0). The 3000
rung is the sharpest (30 points) but also the smallest sample. R0's *network*
is symmetric to three decimals in the STOP2 report (val_acc 0.5212 White /
0.5209 Black at epoch 8), so if this holds up under more games it lives in
`agent.py` — opening pre-search, time management, or the ponder thread — not
the net. Worth another ~60 games a side before treating it as real; plausibly
free Elo if it is.

**Draw wall at 2800.** 13 draws in 20 games (12 by threefold), only 3 losses,
against 2800's 4 wins. R0 holds that rung rather than losing to it, but
converts almost nothing — 4 wins where a similar score-shape at 2600 produced
19. That reads as a value-head / endgame-technique gap (can't finish a
won-ish position) rather than a search or opening problem, and is the kind of
gap more training data (R1 at 40M, still pending) would plausibly close.

## 5. Caveats that don't wash out

- **Pondering is asymmetric.** R0 thinks on the opponent's clock via a second
  thread; Stockfish runs `Threads=1`. Legal on the platform, where each side
  gets its own core — on this 8-core Mac with cores to spare it's real,
  unmeasured extra compute R0 gets that Stockfish doesn't. 2868 is inflated
  by some amount this report can't isolate.
- **This is still R0 on the 8M shard.** Everything above measures the
  checkpoint currently in `weights/`. R1 (40M shard, same architecture) is a
  separate, so-far-unmeasured checkpoint; STOP2's own R0 arena report already
  showed the train/val gap saturated at −0.0860 by epoch 8 on 8M, so the
  headroom this ladder can't see is in R1, not in training R0 longer.
- **Single machine, single architecture generation.** All 112 games plus the
  8-game reference-hero match ran on this Mac. Nothing here has been
  cross-checked on the RunPod box or any other hardware.

## 6. Artifacts

`scratchpad/bracket/` (session-local, not in this repo): 112 PGNs
(`e{elo}_r{round}_g{game}.pgn`), matching per-game logs, `results.csv` (rung,
round, game, bot colour, result, termination). `sparring/stockfish-agent/`
holds the wrapper, fixed version, outside the repo per the GPL/quarantine
rule stated above.

## 7. Open items

1. **R1 is the actual headroom question and is untouched by this report.**
   40M shard, same `d96 x b12`. STOP2 §8 has it running on the RunPod box;
   status there is unknown as of this writing.
2. **Colour asymmetry (§4) unresolved** — needs a dedicated same-conditions
   run, White-only vs Black-only, before treating it as a real effect worth
   fixing in `agent.py`.
3. **Pondering-asymmetry confound (§5) not quantified.** A same-machine run
   with R0's ponder thread disabled would isolate how much of 2868 is that.
4. **This ladder has not been run against `baselines/reference-hero`
   directly** (i.e. reference-hero vs the same five Stockfish rungs, on this
   machine, this harness). That would be the cleanest possible replication of
   the reference's own 2,474 and the strongest sanity check available for the
   method itself — nothing here currently proves this harness reproduces that
   number for a checkpoint whose answer is already known.
