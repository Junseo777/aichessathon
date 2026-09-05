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

## 2. Phase J2 — fixed node caps

PENDING: R1 capped at 800 simulations per move against R1 capped at 400, the clock
deadline left as a safety net, pondering off, six lanes, 200 games; queued behind
ARENA11 (phase K).

## 3. Artifacts

`sparring/bracket_box/lanes/J/` (and `J2/` when played): per lane `results.csv`,
`meta.json`, one PGN and one log per game with both agents' per-move telemetry;
`summary_local.txt`. Agents `/workspace/bracket/agents/r1_full_np`, `r1_half_np`,
`r1_s800_np`, `r1_s400_np`; queues `sparring/bracket_size/queue_j.sh`, `queue_j2.sh`.
