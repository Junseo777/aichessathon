# ARENA #2 — R2 head-to-head vs R0, and R2 on the Stockfish ladder

Continues `docs/ARENA1_R0_LADDER.md` and `docs/STOP3_R1_R2.md`. Written to be read cold.

Date: 2026-09-03, 18:31–19:38 local. Box: Apple M1, 8 physical cores, macOS 26.5.2.
Harness: `harness.play` at the competition clock — 120 s + 0.5 s, 300-ply cap,
`harness/rules.py` defaults, unmodified. Two lanes run **in parallel** for one hour
each; see §6 for why that is defensible here and where it isn't.

---

## 1. Headline

**R2 beats R0 head-to-head by +232 Elo (95% CI +6 to +458, p = 0.039).**
That is the first direct measurement of what the 40M shard bought, and it is
positive at the 5% level on 12 games.

**On the Stockfish ladder R2 lands at 2900 (95% 2691–3109) against R0's 2872
on the same two rungs — a +28 difference that this lane has no power to resolve.**

The two results are not in conflict. Lane B's interval comfortably contains
R0 + 232 = 3104. The head-to-head is simply a far more efficient estimator: 12
games of R2-vs-R0 separate the two nets, where 12 games against a common
opponent that draws 83% of the time do not.

| lane | matchup | games | result | score |
|------|---------|-------|--------|-------|
| A | R2 vs R0 | 12 | **+8 =3 −1** | **79.2%** |
| B | R2 vs SF2800 | 6 | +1 =5 −0 | 58.3% |
| B | R2 vs SF3000 | 6 | +0 =5 −1 | 41.7% |

Zero failed games. No crash, flag, illegal move, or init timeout on either side
across 24 games. All 24 PGNs replayed move-for-move through `python-chess`: all
legal from file.

## 2. What this A/B actually isolates

From `docs/STOP3_R1_R2.md` §1: R0 and R2 share the **same value target** (engine
labels, 100% coverage) and differ in **shard size** — 8M rows vs 40M.

So lane A measures the data-volume question, cleanly, with the value target held
constant: **5x more data is worth roughly +232 Elo of playing strength**, against
+0.0318 of val accuracy (0.5210 → 0.5528).

It does **not** measure Part C. The R1-vs-R2 head-to-head that `STOP3_R1_R2.md` §4
asked for — the direct test of whether engine value labels beat outcome labels —
is still unrun. Nothing here should be read as evidence for or against Part C.

## 3. Lane A — R2 vs R0, 12 games, standard start

```
games 12   R2 +8 =3 -1   score 79.2%   (95% Wilson CI 50.9%-93.3%)
implied R2 - R0: +232 Elo (95% CI +6 to +458)
binomial two-sided p = 0.039
terminations: checkmate 9, threefold_repetition 3
```

| | games | W–D–L | score |
|---|---|---|---|
| R2 as White | 6 | +3 =2 −1 | 66.7% |
| R2 as Black | 6 | +5 =1 −0 | 91.7% |

**Openings were not varied, and did not need to be.** An earlier R1 vs R0 run
produced 4/4 identical threefold draws from the standard start, which looked like
determinism but was not: `agent-r0/agent.py` was a **symlink into the repo**, and
`agent.py` resolves its weights with `Path(__file__).resolve().parent / "weights"`.
`resolve()` follows symlinks, so both sides loaded `aichessathon/weights` — R1
played itself. Fixed by making the sparring dirs hold real copies.

This run was verified before launch: each agent's `agent.py` is a regular file, and
each resolves to its own net — R0 → `1cbb39eb…` (`R0_e8_ema`), R2 → `5c433379…`
(`R2_e8_ema`), both matching `weights/CHECKSUMS.txt`. The manifests are identical
and carry no run name, so checksums were the only way to confirm this.

With the bug gone, the standard start produces variety: **11 of 12 games are
distinct move-for-move.** Games 4 and 6 (Najdorf, R0 as White) are byte-identical
replays, so the effective sample is nearer 11 than 12.

## 4. Lane B — R2 vs Stockfish 18, 12 games

`Threads=1`, `Hash=64`, `UCI_LimitStrength`, standard start — deliberately the same
conditions as the R0 bracket in `ARENA1_R0_LADDER.md`, so the numbers are comparable.

| rung | games | score | W–D–L | implied |
|------|-------|-------|-------|---------|
| 2800 | 6 | 58.3% | +1 =5 −0 | 2858 |
| 3000 | 6 | 41.7% | +0 =5 −1 | 2942 |

Joint MLE over the two rungs: **R2 = 2900, 95% likelihood interval 2691–3109.**

Against R0 on the identical two rungs (recomputed from `ARENA1_R0_LADDER.md` §1 —
the published 2868 is a five-rung fit and is not directly comparable):

| | R0 | R2 |
|---|---|---|
| SF2800 | 52.5% (+4 =13 −3, n=20) | 58.3% (+1 =5 −0, n=6) |
| SF3000 | 40.0% (+3 =10 −7, n=20) | 41.7% (+0 =5 −1, n=6) |
| two-rung MLE | **2872** (2758–2984) | **2900** (2691–3109) |
| loss rate | 25% | 8% |
| draw rate | 57% | 83% |

The fit code reproduces the published 2868 exactly from the five-rung data, which
is the check that it implements the same method.

**The draw wall from `ARENA1_R0_LADDER.md` §4 got worse, not better.** R2 drew 10 of 12
(8 by threefold, 2 by insufficient material) and lost 1. The shift away from losses
and toward draws is in the direction a stronger value head predicts, but neither
half is significant at this size — Fisher p = 0.42 on losses, p = 0.17 on draws.
R2 holds these rungs comfortably and converts almost nothing, exactly as R0 did.

## 5. The colour asymmetry does not reproduce

`ARENA1_R0_LADDER.md` §4 flagged R0 at 80.4% as White vs 66.1% as Black over 112 games
and asked for a dedicated run. This run is not that run, but it is evidence:

- **vs Stockfish, R2 is exactly symmetric**: 3.0/6 as White, 3.0/6 as Black.
- **vs R0, the gap runs the other way**: 66.7% White, 91.7% Black (Fisher p = 0.55).

Six games a side settles nothing. But the effect neither reproduced nor held its
sign, which weakly argues against a large stable White bias in `agent.py`. The open
item stands.

## 6. Caveats that bear on the numbers

- **The two lanes ran in parallel.** `ARENA1_R0_LADDER.md` §2 records a previous run
  discarded over CPU contention, so load was measured rather than assumed: 3.48
  load average, 25% idle, each neural agent GIL-bound to ~1 core, ~3.5 busy cores
  of 8. No oversubscription — unlike the earlier incident, where 6 processes fought
  over the same cores. Search here is wall-clock budgeted, so contention costs
  simulations rather than games, and it hits both sides of a pairing symmetrically.
  Per-game wall times were tight and normal: lane A 161–326 s, lane B 256–309 s.
- **Pondering is still asymmetric against Stockfish.** R2 thinks on Stockfish's
  clock; the wrapper does not ponder back. This inflates lane B by an unmeasured
  amount — but it inflated R0's 2872 identically, so the R0-vs-R2 comparison is not
  distorted by it. It does mean neither number is a chess rating.
- **n = 12 per lane.** Lane A clears p < 0.05; nothing in lane B does.
- **Two rungs only, both compressed.** `ARENA1_R0_LADDER.md` §3 already showed adjacent
  rungs scoring identically. Read lane B's interval, not its point estimate.
- **Per-move telemetry is absent from these logs, by design of the harness.**
  `harness/runner.py` does `os.dup2(2, 1)`, so the agent's `print()` goes to a pipe
  and is block-buffered; `Agent.stop()` ends the process with `SIGKILL`, so the
  unflushed buffer is lost. Only `sys.stderr` writes survive. `PYTHONUNBUFFERED`
  would recover it, but was deliberately not set mid-run.
- **All 10 tracebacks in the logs are benign and expected.** Each is the *winner's*
  ponder thread calling MCTS on an already-mated position after the game-ending
  move was returned; `_ponder` catches it. They appear in exactly the 10 checkmates
  won by a neural agent — the 11th was won by Stockfish, which does not ponder.

## 7. What to do with this

1. **Ship R2.** `aichessathon/weights/` already points at `R2_e8_ema`. Lane A is
   direct evidence it is the strongest net trained so far.
2. **R1 vs R2 is still the missing experiment.** It is the only thing that settles
   what Part C bought, and it is one hour of the same harness.
3. **The conversion problem, not the ladder position, is where the Elo is.** R2
   draws 83% against 2800/3000. Both `ARENA1_R0_LADDER.md` §4 and this run point at
   endgame technique rather than search or openings.
4. **Stop measuring on this ladder at 2800/3000.** It cannot resolve differences
   this size. Head-to-head against the previous checkpoint is ~10x more efficient
   per game and is what caught +232 here.

## 8. Artifacts

`sparring/run_1h/` — `laneA.sh`, `laneB.sh`, `analyse.py`, and per-lane directories
holding 24 PGNs, 24 logs, `results.csv`, and `run.log`.
`sparring/run_1h/laneA_varied_partial/` — two games from an abandoned
varied-opening start to lane A, kept for provenance, excluded from every number above.
`sparring/openings.tsv` — the eight book positions that start was built on, unused
in the final run.
