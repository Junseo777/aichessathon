# ARENA #10 — what is a doubling of search worth?

Every size decision in this project trades network accuracy against simulations per
move, and until now the exchange rate was a guess. ARENA9 needed it: R6a lost by 67 Elo
while searching 0.65× as many nodes as R1, which is consistent with anything from
"worse net" to "better net per node" depending on what a doubling is worth. This arena
measures it with R1 against itself at two search budgets. Written to be read cold.

Box: RunPod, Ryzen 9 7950X, one physical core per agent via `taskset`, cgroup quota
13.6 cores. Harness: `harness.referee.play_match` at the competition clock,
120 s + 0.5 s, 300-ply cap, `harness/rules.py` defaults, unmodified. Code: `aichessathon`
at **f2a78b8** on both sides; net R1 `ea52cc6a…` on both sides; draw rule at the
shipped ±0.30 on both sides (identical heads, so it fires symmetrically). **Pondering
off on both sides** (`_PONDER_NODE_BUDGET = 0`): with pondering on, the side that
thinks less would ponder through the other side's longer think, and the two sides would
do the same total work.

---

## 1. Phase J — halving the time allowance does not halve the search

First design: the "half" side returns half of `_budget_s` per move. Four lanes on cores
4–11, 200 games, 02:17–05:56 UTC, 0 throttled periods.

```
games 200   full +39 =134 -27   score 53.0%   (95% Wilson 46.1%-59.8%)
implied full - half: +21 Elo (95% -27 to +69)
full as White: 51.5%      full as Black: 54.5%
terminations: checkmate 66, threefold 117, insufficient material 16, stalemate 1   (threefold 58%)
sims/move over the whole game: full 1,692, half 1,857
failed games: none
```

The half side searched *more* per move on average. It spends half the time, so its clock
stays fuller, and `_budget_s` grows with the time left; past move 45 its allowance
exceeds the full side's, and the late game holds nearly half of all moves:

| moves | full: sims / think | half: sims / think | half ÷ full |
|---|---|---|---|
| 1–15 | 1,613 / 3.28 s | 867 / 1.83 s | 0.54 |
| 16–30 | 1,529 / 3.39 s | 877 / 2.00 s | 0.57 |
| 31–45 | 1,992 / 2.49 s | 1,649 / 2.00 s | 0.83 |
| 46–60 | 971 / 1.48 s | 1,573 / 1.88 s | 1.62 |
| 61+ | 2,370 / 1.38 s | 4,148 / 1.43 s | 1.75 |

So phase J measured front-loaded against back-loaded time, not a doubling: the side
that searched harder in the opening and middlegame and less in the endgame scored 53%,
not significant. That is mildly interesting for time management and useless for the
question asked. Kept here so the mistake is not repeated: **any budget experiment on
this agent must fix the node count, not the time, because the time formula feeds back
through the clock.**

## 2. Phase J2 — fixed node caps: a doubling is worth about 160 Elo

Second design: `_MCTS.run(..., max_sims=800)` on one side and `max_sims=400` on the
other, the clock deadline left in place as a safety net, pondering off on both sides,
thresholds at the shipped ±0.30. Six lanes on cores 0–11, 200 games, 2026-09-05
10:13–11:42 UTC, nothing else on the box, 0 throttled periods. Every move on every
side hit its cap exactly: mean 800 and 400 simulations per move over 11,400 moves a
side, at about 1.7 s and 0.9 s of thinking.

```
games 200   800-side +99 =89 -12   score 71.8%   (95% Wilson 65.1%-77.5%)
implied 800 - 400: +162 Elo (95% +109 to +215)
800-side as White: +50 =50 -0, 75.0%      as Black: +49 =39 -12, 68.5%
terminations: checkmate 111, threefold 89   (threefold 44%)
failed games: none; distinct games 100+ of 200
```

**Between 400 and 800 simulations per move, a doubling of search is worth about
160 Elo** (95% about 110 to 215). Two things to carry with the number: it was measured
without pondering, so the budgets are lower than the shipped agent's effective ones,
and doublings usually buy less as the budget rises, so treat 160 as the value at the
low end of the range the box plays in and as an upper bound for the box's late-evening
rate of 1,500.

## 3. What the number does to the week's results

With an exchange rate, every arena that compared nets at different search budgets can
be read for what it says about the net itself:

| pairing | search ratio | expected from search alone | observed | net quality per node |
|---|---|---|---|---|
| R6a vs R1 (ARENA9, 200 games) | 0.65× | about −100 | −44 | R6a ≈ +55 |
| int8 R1 vs fp32 R1 (ARENA11) | 1.41× | about +80 | −28 | quantisation ≈ −110 |
| R5 vs R2 (ARENA6, loaded box) | 0.86× | about −35 | −10 | R5 ≈ +25 |
| R4 vs R2 (ARENA6, loaded box) | 1.18× | about +40 | −56 | R4 ≈ −95 |
| RA vs R2 (ARENA6, loaded box) | 1.26× | about +55 | −372 | RA ≈ −425 |

Read with the usual ±70 Elo on each 100-game entry. The picture is consistent: the
d128 nets are better per node and lose on cost; d64 and d32 are worse per node and
their extra search never covers it; int8's rounding costs far more than its speed
returns.

1. **Search speed is the largest lever in the project.** Ten percent more simulations
   is about 22 Elo. That reverses the ARENA9 dismissal of the expansion-path work: on a
   quiet core the Python around the network is about 10% of a simulation, worth about
   20 Elo, and any gain in cache hits, tree reuse or forward cost is worth the same
   rate. It also means the platform's absolute speed, still unread from the validation
   log, sets the whole field's strength.
2. **R6a is the better network per position.** It loses only because d128 costs 35% of
   the search on a quiet core. A d128 forward pass at d96 cost, through a cheaper trunk
   or a quantisation that spares the value head, would put it ahead of R1 by roughly the
   margin R1 has over R2. Nothing available today does that.
3. **Time management barely matters; node count does.** Phase J's front-loaded versus
   back-loaded allocation moved the score by +21 ± 48 Elo at equal total search; phase
   J2's halving moved it by 162.
4. **What is not measured:** the next doubling, 800 against 1,600, which says whether
   the value of search saturates at the box's budget, and the worth of pondering itself,
   which the shipped agent relies on and which doubles the effective budget.

## 4. Decision this supports

**Optimise simulations per move before optimising the network, and read the platform's
budget from the validation log before deciding anything about size.** Sample: 200
games at the competition clock with exact node caps, no failures. What would refine
it: the 800-vs-1,600 doubling, and the same design with pondering on.

## 5. Artifacts

`sparring/bracket_box/lanes/J/` (and `J2/` when played): per lane `results.csv`,
`meta.json`, one PGN and one log per game with both agents' per-move telemetry;
`summary_local.txt`. Agents `/workspace/bracket/agents/r1_full_np`, `r1_half_np`,
`r1_s800_np`, `r1_s400_np`; queues `sparring/bracket_size/queue_j.sh`, `queue_j2.sh`.
