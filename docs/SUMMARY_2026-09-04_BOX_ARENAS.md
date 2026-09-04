# Summary — the night of 2026-09-04 on the box

One page, written to be read cold, answering the four questions in order. Evidence is
in ARENA5 (R1 vs R2 with the draw rule controlled), ARENA6 (the size bracket) and
ARENA7 (R1 vs R3), all played on the RunPod box at the competition clock, 120 s +
0.5 s, one pinned physical core per agent, both sides at `f2a78b8`, eight book
openings with both colours, no failed games in 584. The box's simulation rate drifted
upward through the day (about 700 per move in ARENA5, 880 in ARENA6, 1,500 in
ARENA7; no cgroup throttling at any point), so phases were played at different
budgets; within every game both sides shared the same conditions. The Mac's ARENA3 (R1 vs R2 and
R2 vs R3 at default thresholds) is the independent replication.

## 1. Which net should ship for the ladder: **R1**

R1 (d96, 40M shard, value target = Lichess evals where present, else game outcome)
is the strongest net measured, and it is not close:

| pairing | treatment | games | R1 score | Elo | source |
|---|---|---|---|---|---|
| R1 vs R2 | default ±0.30 both | 101 | 64.9% | +106 (+60 to +157) | ARENA3, Mac |
| R1 vs R2 | rule off both | 100 | 65.5% | +111 (+40 to +182) | ARENA5 A1 |
| R1 vs R2 | R1 ±0.30, R2 ±0.24 | 100 | 61.5% | +81 (+12 to +151) | ARENA5 A2 |
| **R1 vs R3** | rule off both | 84 (64 distinct) | **50.6%** | **+4 (−70 to +78)** | ARENA7 |

R2 is level with R3 (ARENA3: 49.0% over 98 games) and beats or ties every size
variant (ARENA6). The one loose end is R3: it tied R1 head-to-head over 84 games in
the last hour (ARENA7), where R2 could not, and White won 64% of those games from a
book the two nets replay (64 distinct games of 84). So R1 is the pick on three
independent 100-game wins over R2, and R3 is a live alternative that the evidence
cannot separate from it. Switch `weights/` to `R1_e8_ema` (`./use-weights.sh
R1_e8_ema`, then `make zip`); the export is in the run store with its checksum
(`ea52cc6a…`), same code and draw rule as today's R2 upload. Before the lock, 200+
R1 vs R3 games from a wider opening set would settle which of the two it is; if the
Lc0 chain (ARENA4, built on R3) ships, that decides it the other way for free.

## 2. Did Part C's engine labels help or hurt: **hurt**

R2 is R1 with 100% depth-8 Stockfish value labels instead of outcomes; everything
else is identical. At the real clock that costs 80–110 Elo, measured three times with
three different draw-rule treatments and on two machines. Validation accuracy could
not see it (STOP3: +0.0012), because Part C only changes the value target and the
value head is what search amplifies. R3 (value_weight 2.5 on the same labels) does
not recover it. The next full-track run should not assume depth-8 values are an
upgrade; R7's blend and a deeper or MultiPV-derived value are the open questions, and
R6 at 100M with outcome labels is now the safer primary.

## 3. Should the draw thresholds follow value calibration: **not needed; harmless if done**

R1's value head is ~30% hotter than R2's (slope 1.29, corr 0.93 on 1,160 positions;
R3, R4, R5 within 3% of R2, RA 9% cooler), so at ±0.30 it crosses the thresholds in
47% of positions against R2's 32%. But switching the rule off or equalising its
firing rate changes R1 vs R2 by four points, within noise, and the threefold rate is
~50% with the rule on or off. The rule is not what decides games. If it stays, scaling
each net's threshold by its slope against R2 (R1 ±0.39) costs nothing and makes the
rule mean the same thing for every net; it is not a strength lever. The draw problem
is conversion in search, not `_pick`.

## 4. Does d128 or d32 beat d96 at the real clock: **neither; d128 ties, d32 collapses**

| vs R2 (d96) | games | score | Elo | sims/move vs R2's |
|---|---|---|---|---|
| R5 d128 | 100 | 48.5% | −10 (−78 to +57) | 0.86x |
| R4 d64 | 100 | 42.0% | −56 (−125 to +12) | 1.18x |
| RA d32 | 100 | 10.5% | −372 (−482 to −263) | 1.26x |

The reference's size is out by 370 Elo, and the reason is measured: 3–4 ms of Python
per expansion outside the forward pass caps every net at a few hundred simulations
per second, so d32 gets 1.26x d96's simulations, not 5x, and d128 keeps 86%. Capacity
is nearly free up to d128 at this clock; the lever that would actually buy
simulations is the cost per expansion. For the full track, d128 is a defensible
primary and d64 or smaller is ruled out.

## Deliverables and where they are

- Exports R4, R5, RA (e8 and e8_ema) made on the box by the pinned exporter, in the
  run store with checksums in `weights/CHECKSUMS.txt`; checkpoints and histories in
  `checkpoints/`; logs, histories and export records committed under `provenance/`.
  All EMA checkpoints pass `train.verify_provenance`. Every export fails the int8 gate.
- Reports: `docs/ARENA5_R1_VS_R2_CALIBRATION.md`, `docs/ARENA6_SIZE_BRACKET.md`,
  `docs/ARENA7_R1_VS_R3.md`, this summary. Branch `box-arenas`, not pushed to main.
- Games: `sparring/bracket_box/lanes/{A,B,B2,E}/` on the Mac, one PGN and one log per
  game with both agents' per-move telemetry; drivers, queue, probes and an incident
  log in `sparring/bracket_size/`.
- Not done: E from the brief (the Stockfish ladder anchor; the box has no wrapper
  staged and head-to-head was the primary evidence), and D (the threshold sweep),
  which ARENA5 made moot.
