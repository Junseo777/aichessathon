# ARENA #3 — R1 vs R2 and R2 vs R3 at the competition clock, ~100 games each

Continues `docs/ARENA2_R2_VS_R0.md` and `docs/STOP3_R1_R2.md`. Written to be read cold.

Date: 2026-09-04, 02:54–10:52 BST, unattended. Box: Apple M1, 8 cores, macOS 26.5.2.
Harness: `harness.play` at the competition clock, 120 s + 0.5 s, 300-ply cap,
`harness/rules.py` defaults, unmodified. Two lanes in parallel for eight hours, four
agent processes, the parallelism ARENA #2 measured as contention-free here. Search
frozen at `f2a78b8` (first-play urgency 0.25 and the evaluation cache; none of the
`search-lc0` features) in real copies under `sparring/block/agent-r{1,2,3}`; nets by
symlink and confirmed by checksum in every game's init line: R1 `ea52cc6a…`,
R2 `5c433379…`, R3 `a1163b19…` (all `_e8_ema`). Openings: the eight positions in
`sparring/openings.tsv`, cycled, each with both colours. `PYTHONUNBUFFERED=1`, so
every move's simulations, reuse and q survive in the logs.

---

## 1. Headline

**R1 beats R2 by +106 Elo (95% +60 to +157) over 101 games, 64.9%.** R1 is the
Lichess-evals-plus-outcomes net; R2 is the fully Stockfish-labelled one. Same shard,
architecture, hyperparameters and seed: the value target is the only difference. This
is the direct test of Part C that STOP3 §4 asked for, and it comes out the wrong way
round for the pipeline's thesis.

**R2 vs R3 is a coin flip: 49.0% for R2 over 98 games, −7 Elo (−51 to +37).**
value_weight 2.5 bought nothing measurable at the real clock.

Zero failed games in 199. Base net for the Lc0 feature chain by the pre-agreed rule
(higher score, tie keeps R2): **R3**.

## 2. Results

### Lane 1: R1 vs R2

| | games | W-D-L (R1) | score | 95% | Elo (R1 - R2) |
|---|---|---|---|---|---|
| **all** | 101 | +41 =49 -11 | **64.9%** | 58.5-71.2% | **+106** (+60 to +157) |
| R1 as white | 51 | +31 =19 -1 | 79.4% | | |
| R1 as black | 50 | +10 =30 -10 | 50.0% | | |

| opening | games | W-D-L | score |
|---|---|---|---|
| caro_kann | 12 | +3 =8 -1 | 58.3% |
| english_four | 12 | +6 =5 -1 | 70.8% |
| french_winawer | 12 | +5 =7 -0 | 70.8% |
| kings_indian | 12 | +5 =6 -1 | 66.7% |
| qgd_bg5 | 13 | +5 =8 -0 | 69.2% |
| ruy_morphy | 14 | +6 =7 -1 | 67.9% |
| sicilian_najdorf | 14 | +5 =2 -7 | 42.9% |
| slav_accepted | 12 | +6 =6 -0 | 75.0% |

Terminations: checkmate 52, threefold_repetition 42, insufficient_material 6, stalemate 1.
Mean game: 284 s, 113 plies (from 101 PGNs).
R1: mean 609 simulations per move, 2.41 s per move (5688 moves).
R2: mean 649 simulations per move, 2.46 s per move (4801 moves).
Failed games: 0.

### Lane 2: R2 vs R3

| | games | W-D-L (R2) | score | 95% | Elo (R2 - R3) |
|---|---|---|---|---|---|
| **all** | 98 | +19 =58 -21 | **49.0%** | 42.7-55.3% | **-7** (-51 to +37) |
| R2 as white | 49 | +17 =24 -8 | 59.2% | | |
| R2 as black | 49 | +2 =34 -13 | 38.8% | | |

| opening | games | W-D-L | score |
|---|---|---|---|
| caro_kann | 12 | +1 =10 -1 | 50.0% |
| english_four | 12 | +4 =8 -0 | 66.7% |
| french_winawer | 12 | +2 =6 -4 | 41.7% |
| kings_indian | 12 | +1 =6 -5 | 33.3% |
| qgd_bg5 | 12 | +1 =10 -1 | 50.0% |
| ruy_morphy | 14 | +3 =11 -0 | 60.7% |
| sicilian_najdorf | 12 | +4 =2 -6 | 41.7% |
| slav_accepted | 12 | +3 =5 -4 | 45.8% |

Terminations: threefold_repetition 50, checkmate 40, insufficient_material 8.
Mean game: 292 s, 117 plies (from 98 PGNs).
R2: mean 541 simulations per move, 2.39 s per move (5745 moves).
R3: mean 549 simulations per move, 2.36 s per move (5182 moves).
Failed games: 0.



Draws: 48.5% in lane 1 (42 of 49 by threefold), 59.2% in lane 2 (50 of 58). Across all
199 games White scored about 63%; see §4.

## 3. Why R1 wins: a calibration hypothesis, not yet a verdict

The policy heads are near-identical (val_acc 0.5516 vs 0.5528, top prior 0.56 vs
0.55 on these positions). The value heads are not. On 1,718 positions sampled from
the block's own games:

| net | mean \|v\| | share \|v\| > 0.3 | share \|v\| > 0.6 | slope vs R2 | corr vs R2 |
|---|---|---|---|---|---|
| R1 | 0.426 | 51% | 34% | 1.24 | 0.94 |
| R2 | 0.311 | 41% | 20% | 1.00 | 1.00 |
| R3 | 0.316 | 41% | 21% | 0.98 | 0.98 |

The three heads agree on ordering; R1's runs about 25% hotter, as a head trained on
±1 outcomes would. `agent.py`'s draw rule is fixed at ±0.3 (`q_best > 0.3` vetoes a
move that hands the referee a draw claim; `q_best < −0.3` seeks one), so R1 crosses
it in half the positions and R2 in two fifths. R1 declines draws when slightly better
and R2 accepts them. R1 as White went +31 =19 −1; as Black it drew 30 of 50.

Two readings fit: the outcome-labelled net is genuinely stronger and depth-8 engine
values hurt, or the nets are close and the thresholds happen to suit R1's scale. The
old first-play-urgency rule would have handicapped R1 more than R2 (its reduction
scaled with Q), which is consistent with R1 leading only 54% over 14 games under the
old search on 2026-09-03 and 65% here, but 14 games cannot carry that. The
discriminating test is R1 vs R2 with the rule neutralised on both sides, or each
side's threshold scaled to its own spread (R2 at ±0.24); it is queued on the box.
Until it runs, treat "R1 is the best net" as likely and "engine labels hurt" as open.

**Addendum, 2026-09-04 21:30.** The box ran that test (`/workspace/bracket`, phase A,
100 games per arm at the competition clock, R1 vs R2): with the draw rule switched
off on both sides R1 scored **65.5%, +111 Elo (95% +40 to +182)**; with per-net
thresholds (R1 ±0.30, R2 ±0.24) 61.5%, +81. The edge survives both arms, so the
thresholds are not what decides R1 vs R2. The calibration hypothesis above is
refuted for this pairing: the outcome-labelled net is the stronger one under
search, and the depth-8 engine labels are the weaker target. Its bracket, rule off,
~100 games each vs R2: R5 (d128) 48.5%, R4 (d64) 42%, RA (d32) 10%. The size
decision holds at d96, and R5's accuracy does not survive its forward cost. One
loose end from the same run: R1 vs R3 with the rule off came out 50.6% over 84
games (+4, 95% −70 to +78), which sits awkwardly beside R1's +111 over R2 and this
report's R2 ≈ R3; the intervals overlap between +40 and +78, so it is not a
contradiction, but R3 may be better than R2 once the rule is out of the way.

## 4. Other things the data says

- **White scores ~63% at this level**, across both lanes and all four nets (R1 79%,
  R2 50–59%, R3 61% as White). ARENA #1 saw 80/66 against Stockfish. It cancels in a
  colour-balanced match, but on a ladder that assigns colours it is real.
- **The Najdorf is the exception.** R1's only bad opening (42.9%, 7 of its 11 losses),
  and R2 lost it too. Sharp, tactical, both sides below their draw threshold early.
- **R2 and R3 draw each other 59% of the time**, half by threefold. Same policy, same
  calibration, same draw rule: neither declines.
- **Simulations per move** from the telemetry: 541–649 at 2.4 s per move, with tree
  reuse and pondering; the cache and reuse put this above the ~470 a cold 3 s search
  measures.

## 5. Caveats

n ≈ 100 per pairing resolves ±60 Elo. Both lanes shared the machine (symmetric under
a wall-clock budget, per ARENA #2's load measurement). Eight openings only, and each
pairing saw each opening 12–14 times, so an opening-level number is six or seven
games a colour. Pondering is symmetric here and not on the ladder. The search is the
pre-feature `f2a78b8`; the Lc0 features are measured separately in ARENA #4.

## 6. Artifacts

`sparring/block/lane_r1r2/` and `lane_r2r3/`: 199 PGNs, matching logs with per-move
telemetry, `results.csv`, `run.log`. `sparring/block/report.py` regenerates §2;
`report.md` is its output. The calibration probe is `scratchpad/calib.py` (session-local)
and its numbers are in §3.

