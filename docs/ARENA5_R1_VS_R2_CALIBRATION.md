# ARENA #5 — R1 vs R2 with the draw rule controlled

Answers one question: is R1's win over R2 (the Mac's ARENA4 block, 67.5% at 40 games
with both sides at the default thresholds) a better net, or the draw rule's fixed
thresholds suiting R1's hotter value head? Written to be read cold.

Date: 2026-09-04, 06:10–08:57 UTC. Box: RunPod, Ryzen 9 7950X, one physical core per
agent via `taskset`, hyperthread siblings idle, cgroup CPU quota 13.6 cores. Harness:
`harness.referee.play_match` at the competition clock, 120 s + 0.5 s, 300-ply cap,
60 s init, `harness/rules.py` defaults, unmodified. Code: `aichessathon` at
**f2a78b8** on every side (first-play urgency 0.25, ungated evaluation cache).
Nets, EMA epoch 8, read back from each agent's `init:` line in every game log:
R1 `ea52cc6a…`, R2 `5c433379…`. Zero mismatches against the lane metadata.

---

## 1. Headline

**R1 beats R2 whether or not the draw rule is in play.** With the rule switched off on
both sides R1 scores 65.5%; with the thresholds set so the rule fires as often for R2
as for R1, 61.5%. Both are significant on 100 games each; both intervals contain the
Mac's default-rule 67.5%.

| arm | thresholds R1 / R2 | games | R1 W–D–L | score | 95% CI | implied Elo |
|---|---|---|---|---|---|---|
| A1 rule off | ±1.00 / ±1.00 | 100 | +40 =51 −9 | **65.5%** | 55.8–74.1% | **+111** (+40 to +182) |
| A2 per-net | ±0.30 / ±0.24 | 100 | +34 =55 −11 | **61.5%** | 51.7–70.4% | **+81** (+12 to +151) |

So the edge is in the net. Part C's depth-8 engine labels, the only difference between
R1 and R2, did not produce a stronger player than the outcome-labelled R1: they cost
roughly 80–110 Elo at this clock. Section 4 says what that does and does not imply.

## 2. What was controlled, and how

`agent.py`'s `_pick` vetoes moves that hand the referee a draw claim when
`q_best > 0.3` and seeks one when `q_best < -0.3`. The Mac's probe found R1's value head
about 25% hotter than R2's (slope 1.25), so at the same thresholds R1 crosses them in
more positions. Two arms, run in parallel, three lanes each:

- **A1, rule off.** Both thresholds set to ±1.00 on both sides, which no search value
  reaches. The near-adjudication material logic is untouched. Whatever remains is
  net plus search.
- **A2, per-net thresholds.** R1 at ±0.30, R2 at ±0.24 (0.30 / 1.25), so the rule
  fires at the same underlying evaluation on both sides.

Each side is a frozen real copy of `agent.py` at f2a78b8 with only those two literals
changed (`sparring/bracket_size/queue.sh`, `mkagent`), a `chessml` symlink into the
archived commit, and a symlink to the net in the run store. The eight book positions
in `sparring/openings.tsv` were cycled, each played with both colour assignments; the
three lanes of an arm start at different openings and opposite first colours, so
every arm has exactly 50 games per colour. Candidate and opponent each hold one
physical core for the whole lane; the six lanes occupied cores 0–11. Per-move
telemetry (`sims`, `reused`, `pondered`, `q`) was kept unbuffered and every PGN saved.

## 3. Results in detail

### A1 — rule off both sides

```
games 100   R1 +40 =51 -9   score 65.5%   (95% Wilson 55.8%-74.1%)
implied R1 - R2: +111 Elo (95% +40 to +182)
R1 as White: +26 =21 -3, 73.0%      R1 as Black: +14 =30 -6, 58.0%
White scored 57.5% over all games
lanes: 66.2% / 69.1% / 60.9%
terminations: checkmate 49, threefold 49, insufficient material 2   (threefold 49%)
mean game 291 s, mean 115 plies, longest 362 s
sims/move: R1 715 (pondered 603), R2 693 (pondered 540); peak RSS 155 / 137 MB
distinct games 94 of 100 (six replays across lanes from the same book position)
failed games: none
```

### A2 — R1 ±0.30, R2 ±0.24

```
games 100   R1 +34 =55 -11   score 61.5%   (95% Wilson 51.7%-70.4%)
implied R1 - R2: +81 Elo (95% +12 to +151)
R1 as White: +22 =26 -2, 70.0%      R1 as Black: +12 =29 -9, 53.0%
White scored 58.5% over all games
lanes: 69.1% / 63.2% / 51.6%
terminations: checkmate 45, threefold 52, insufficient material 3   (threefold 52%)
mean game 285 s, mean 109 plies, longest 363 s
sims/move: R1 667 (pondered 523), R2 607 (pondered 479); peak RSS 151 / 138 MB
distinct games 98 of 100
failed games: none
```

By opening (R1's score, n = 12–14 each): A1 ranges from 50% (Caro-Kann, Najdorf) to
88% (English); A2 from 50% (Winawer, King's Indian) to 68% (Slav). No opening
reverses the sign.

### Calibration probe

Run after the arms finished, on the arms' own games: 1,160 distinct positions
(ply ≥ 10, up to eight per game, from all 200 PGNs), every net's raw value head on
the same positions, side-to-move perspective (`sparring/bracket_size/probe.py`,
`sparring/bracket_box/probe.json`). Slope is OLS against R2's value; slope₀ is the
through-origin slope, which is the number a threshold should scale by.

| net | mean \|v\| | \|v\| > 0.30 | \|v\| > 0.24 | slope vs R2 | slope₀ | corr |
|---|---|---|---|---|---|---|
| R2 | 0.257 | 31.7% | 37.0% | 1.000 | 1.000 | 1.000 |
| **R1** | **0.387** | **47.1%** | **52.7%** | **1.305** | **1.292** | 0.930 |
| R3 | 0.261 | 32.7% | 38.3% | 0.985 | 0.985 | 0.979 |
| R4 | 0.255 | 32.2% | 36.5% | 0.968 | 0.968 | 0.975 |
| R5 | 0.262 | 32.9% | 38.9% | 0.985 | 0.986 | 0.978 |
| RA | 0.256 | 32.6% | 38.2% | 0.904 | 0.909 | 0.928 |

This reproduces the Mac's probe (slope 1.25, correlation 0.93 on 530 positions) on
different positions: R1 is about 30% hotter than R2 and crosses ±0.30 in 47% of
positions against R2's 32%. Every engine-labelled net, whatever its size or value
weight, is calibrated within 3% of R2; RA is 9% cooler. So the calibration split is
between the two label regimes, not between architectures, and the thresholds that
would equalise the rule are R1 ±0.39 with R2 at ±0.30, or R2 ±0.23 with R1 at ±0.30
(arm A2 used 0.24).

## 4. Reading it

1. **The rule is not the explanation.** Removing it entirely leaves R1 at 65.5%.
   Equalising its firing rate moves R1 to 61.5%, four points lower, well inside the
   noise of two 100-game samples (each interval is ±9 points). If the thresholds were
   what R1's win was made of, A1 would have collapsed toward 50%. It did not.
2. **Part C did not buy strength at d96 on 40M.** R1 and R2 share shard, seed,
   schedule, architecture and policy target; only the value target differs. STOP #3
   already showed that difference invisible in validation accuracy (+0.0012). Under
   search at the real clock it is visible, and it has the wrong sign: the net trained
   on game outcomes (with Lichess evals on 8.8% of rows) plays 80–110 Elo stronger
   than the one trained on 100% depth-8 Stockfish values. The likely mechanism is that
   depth-8 values are a smoother, lower-variance target that the value head fits
   tightly (R2's val MSE 0.015) but that carries less of what search needs at 600–700
   simulations, where the value head is mostly asked to tell won from drawn. This
   experiment does not identify the mechanism; it identifies the sign.
3. **The threefold rate is not the rule's doing.** With the rule off, 49% of games
   still ended by repetition; with the rule on, 52%. Both nets shuffle into
   repetitions on their own, mostly from positions the winner rated well above zero
   (game a1_z/g1 ended at +0.54 for R1). The conversion problem named in ARENA2 §7 is
   a search-and-value problem, not a `_pick` problem.
4. **White's edge here is 57–59%,** not the 66% seen in the Mac block. Same openings,
   same code; different hardware and a different pairing. Both arms had exactly 50
   games per colour, so the headline is not colour-biased.
5. **R2 searched slightly less than R1** (693 vs 715 sims/move with the rule off, 607
   vs 667 with it on). Same architecture, so this is cache hit rate and position
   character, not model cost; it is too small to explain 100 Elo.

## 5. Decision this supports

**Part C's engine labels hurt rather than helped, and R1 is the stronger d96 net.**
Sample: 200 games at the competition clock, 100 per arm, no failed games, two
independent treatments of the one confound that could have explained the earlier
result. Downstream: the size bracket (ARENA6) runs with the draw rule off on both
sides, since the rule does not decide outcomes and switching it off removes a
calibration-dependent term from a comparison between nets of different sizes; the
ship candidate is tested against R1, not R2 (ARENA7); and the next training run
should not assume depth-8 value labels are an upgrade over outcomes.

What this does not settle: whether deeper or MultiPV-derived engine values would
behave differently (the soft policy target, R8, is a separate question), and whether
a blend (R7's 0.5/0.5) recovers the loss. Both remain open.

## 6. Caveats

- Two arms, one box, one night. Both arms ran concurrently on separate pinned cores.
- Forward times measured on this box vary by up to 60% between runs on the same
  pinned core (frequency and cache pressure on a shared host), so absolute
  simulation counts are not transferable to the platform; the two sides of every
  game shared the same conditions at the same time, so the comparison is fair.
- Six of A1's 100 games and two of A2's are move-for-move replays of another game
  from the same book position; the effective sample is nearer 94 and 98.
- The box runs Python 3.14 against the platform's 3.12; onnxruntime 1.29, numpy 2.5
  and python-chess 1.11 match.

## 7. Artifacts

`sparring/bracket_box/lanes/A/` (pulled from the box's `/workspace/bracket/lanes/A/`):
six lanes, each with `results.csv`, `meta.json` (commit, cores, net sha256s), one PGN
and one log per game holding both agents' full stderr with per-move telemetry;
`summary.txt` from `sparring/bracket_size/analyse.py`. Probe: `probe.json`. Driver
and queue: `sparring/bracket_size/`.
