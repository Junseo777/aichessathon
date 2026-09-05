# ARENA #9 — R6a, the d128 net trained on R1's target, against R1

Two results pointed at one net that had never been trained: outcome-style labels beat
depth-8 engine labels by ~100 Elo at d96 (ARENA3, ARENA5, ARENA8), and d128 tied d96
under engine labels at the real clock (ARENA6). R6a is d128 trained on R1's target,
the Lichess `[%eval]` where one exists and the game outcome elsewhere, through the
trainer's new `--value-source lichess` switch. This arena plays it against R1, the
ship candidate. Written to be read cold.

Date: 2026-09-04, 22:09–23:35 UTC. Box: RunPod, Ryzen 9 7950X, one physical core per
agent via `taskset`, cgroup quota 13.6 cores, six lanes on cores 0–11, nothing else
running. Harness: `harness.referee.play_match` at the competition clock, 120 s + 0.5 s,
300-ply cap, 60 s init, `harness/rules.py` defaults, unmodified. Code: `aichessathon`
at **f2a78b8** on both sides; draw rule off on both sides (thresholds ±1.00), the
treatment of ARENA5 A1, ARENA6, ARENA7 and ARENA8. Nets, EMA epoch 8, read back from
each agent's `init:` line in every game: R6a `4415d69a…`, R1 `ea52cc6a…`. Eight book
openings with both colours, 50 games per colour.

R6a's provenance (checkpoint `db8b34fc…`, `verify_provenance` PASS): d128, 8 heads,
12 blocks, 2,438,314 params; 40M shard, seed 0, 8 epochs, batch 1024, value weight 1.0,
`value_source lichess`; trained 15:52–19:52 UTC on the box at commit `edd07b7`
(branch `train-value-source`, main `f2a78b8` plus the flag); val accuracy 0.5582
(EMA 0.5582), train-vs-val policy gap −0.042, val value MSE 0.540 on the outcome scale.
Exported on the box by the f2a78b8 exporter; int8 failed the gate (0.91), fp32 plays.

---

## 1. Headline

**R6a loses to R1: 40.5% over 100 games, −67 Elo (95% −136 to +2), while searching
68% as many nodes per move.** A 100-game replay (§5) scored 47.0%; pooled, 43.8% over
200 games, −44 Elo.

```
games 100   R6a +11 =59 -30   score 40.5%   (95% Wilson 31.4%-50.3%)
implied R6a - R1: -67 Elo (95% -136 to +2)
R6a as White: +6 =37 -7, 49.0%      R6a as Black: +5 =22 -23, 32.0%
lanes: 41.2% / 32.4% / 41.2% / 50.0% / 50.0% / 28.1%
terminations: checkmate 41, threefold 51, insufficient material 8   (threefold 51%)
mean game 293 s, 118 plies, longest 341 s
sims/move: R6a 934 (pondered 763), R1 1,378 (pondered 1,243); peak RSS 175 / 246 MB
distinct games 62 of 100; failed games: none; 304 throttled periods (0.6%)
by opening (R6a): Caro-Kann 58%, Najdorf 54%, KID 50%, QGD 50%, Ruy 50%, Slav 31%,
                  English 21%, Winawer 8%
```

R6a has the best validation accuracy of any outcome-trained net (55.8% against R1's
55.2%) and loses. Twenty-three of its thirty losses came as Black.

## 2. Where the simulations went

ARENA6 measured d128 at 0.86× d96's simulations in play and concluded that capacity was
nearly free up to d128, because 3–4 ms of Python per expansion dwarfed the extra
forward cost. That measurement was made with twelve agents on the box. After this
phase, with the box idle, the same probe (`sparring/bracket_size/simrate.py`, one
pinned core, 3 s per book position, fresh tree, empty cache) reads differently:

| net | forward ms | sims/s | forwards/s | cache hits | ms per sim | Python per forward |
|---|---|---|---|---|---|---|
| R1 d96 | 2.30 | 480 | 392 | 19.0% | 2.08 | 0.26 ms |
| R2 d96 | 2.31 | 485 | 389 | 19.1% | 2.06 | 0.26 ms |
| **R6a d128** | **3.57** | **310** | 262 | 14.8% | 3.23 | 0.25 ms |

Idle, the forward pass is 3.4× faster than under load (2.3 ms against 7.8 ms) and the
Python around it is 12× cheaper (0.26 ms against 3.1 ms): contention for shared cache
and memory hits the interpreter far harder than it hits the ONNX kernels. With the
overhead gone, the network is the whole cost and d128's 1.55× forward time becomes
0.65× the simulations, which is what the games showed (0.68×). The ratio ARENA6 found
was a property of a loaded box, not of the net.

The platform runs two agents per machine, each on a dedicated core. That is the quiet
regime, not the loaded one. Read the size trade at 0.65×, not 0.86×.

## 3. Reading it

1. **R6a is not a ship candidate at this clock.** Sixty-seven Elo behind R1 on 100
   games, losing as both colours.
2. **Whether it is the better net per simulation is not settled.** At 0.65× the search,
   the deficit is about 0.6 doublings. If a doubling of search is worth ~100 Elo at this
   budget, the result is consistent with per-simulation parity; if it is worth 50, R6a
   is worse per simulation too. Elo per doubling has never been measured here; R1
   against itself at half the clock would measure it in 100 games and turn every size
   comparison into arithmetic.
3. **d96 is the size for the platform.** On a quiet core d128 pays 35% of its search;
   R5 tied R2 at a 14% cost (ARENA6) and would not at 35%. The full-track primary
   should stay at d96 unless the expansion path gets much cheaper or int8 passes the
   gate, either of which changes the arithmetic in d128's favour.
4. **Validation accuracy mis-ranks nets for play, again.** R6a > R1 on accuracy by 0.6
   points and loses by 67 Elo; R1 < R2 on accuracy and wins by 100. Selection by play is
   the only selection that has worked.
5. **The outcome-label gain did not obviously transfer to d128,** but this arena cannot
   say so on its own: R6a vs R5 at equal search would, and R6a vs R1 at a fixed node
   count would. Neither has been played.
6. **Replays and budget.** 62 of 100 games distinct; ~1,380 simulations per move for R1,
   the box's late-evening rate. As in every arena tonight, the two sides of each game
   shared conditions.

## 4. Decision this supports

**Keep R1 as the ship candidate; do not move the full track to d128.** Sample: 100
games at the competition clock, no failures. What would change it: R6a at a fixed
simulation count beating R1 by more than the search deficit costs, with Elo per
doubling measured, and a cheaper expansion path or a passing int8 export.

## 5. Phase H2 — the replay

Two lanes on cores 0–3, 100 games, 2026-09-05 01:06–05:14 UTC, started as a quiet-box
replay; the other session's four-lane arena and a training run joined the box at
01:43, so it was quiet for its first 37 minutes only.

```
games 100   R6a +19 =56 -25   score 47.0%   (95% Wilson 37.5%-56.7%)
implied R6a - R1: -21 Elo (95% -89 to +47)
R6a as White: +11 =32 -7, 54.0%      R6a as Black: +8 =24 -18, 40.0%
terminations: checkmate 44, threefold 49, insufficient material 6, stalemate 1
sims/move: R6a 1,007 (pondered 866), R1 1,428 (pondered 1,302)   ratio 0.71
failed games: none; 105 throttled periods
```

Softer than phase H but the same sign, at the same 0.7 simulation share. Pooled over the
two phases, 200 games: **R6a +30 =115 −55, 43.8%, −44 Elo (95% about −90 to 0)**.
The interval no longer reaches zero. R6a is behind R1 at this clock.

## 6. Artifacts

`sparring/bracket_box/lanes/H/`: six lanes with `results.csv`, `meta.json`, one PGN and
one log per game with both agents' per-move telemetry; `summary_local.txt`. Agent
`/workspace/bracket/agents/r6a_t1.00`; queue `sparring/bracket_size/queue_h.sh`, log
`queue_h.log`; export and pull `sparring/bracket_size/export_pull_r6a.sh`; run store
`weights/R6a_e8[_ema]/` with checksums in `weights/CHECKSUMS.txt`; checkpoints, history
and logs in `checkpoints/`. Trainer flag: branch `train-value-source`, commits
`edd07b7` and `600cb3a`.
