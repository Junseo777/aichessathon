# ARENA #11 — what is a doubling of search worth?

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

## 0. Answer

**Between 400 and 800 simulations per move, a doubling of search is worth +184 Elo
(95% +130 to +238)**: R1 capped at 800 nodes scores 74.3% against R1 capped at 400 over
204 games from 204 distinct positions, pondering off, every move on both sides at its
cap (§5). Two earlier designs are kept below because each failed in a way worth
remembering: halving the time allowance (§1) did not halve the search, and the eight-
opening book (§2) produced sixteen distinct games in two hundred.

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

**Only 16 of the 200 games are distinct.** With fixed caps, no pondering and the same
net on both sides the agent is deterministic, so eight openings with both colours give
sixteen games, each replayed about twelve times. The 71.8% is the mean of sixteen
outcomes, and the Wilson interval above is wrong by a factor of about four: on 16
independent trials it runs from roughly 47% to 88%, which is anything from 0 to about
350 Elo per doubling. Per opening the 800-side scored 100% in three, 50% in four and
26% in one, so the sign is not in doubt and the magnitude is.

So J2 says the sign and little more; §5 is the measurement.

## 3. What the number does to the week's results

At 184 Elo per doubling, every arena that compared nets at different search budgets can
be read for what it says about the net itself. "Expected" is 184 × log2(ratio).

| pairing | search ratio | expected from search alone | observed | net quality per node |
|---|---|---|---|---|
| R6a vs R1 (ARENA9, 200 games) | 0.65× | −114 | −44 | R6a ≈ +70 |
| int8 R1 vs fp32 R1 (ARENA12) | 1.41× | +91 | −28 | quantisation ≈ −120 |
| R5 vs R2 (ARENA6, loaded box) | 0.86× | −40 | −10 | R5 ≈ +30 |
| R4 vs R2 (ARENA6, loaded box) | 1.18× | +44 | −56 | R4 ≈ −100 |
| RA vs R2 (ARENA6, loaded box) | 1.26× | +61 | −372 | RA ≈ −430 |

Read with ±70 Elo on each 100-game entry and ±55 on the exchange rate, and with the
caveat that the rate was measured at 400–800 nodes without pondering; the box's
late-evening games ran at 1,400–1,700 with pondering, where a doubling is likely worth
less. The picture is consistent even so: the d128 nets are better per node and lose on
cost; d64 and d32 are worse per node and their extra search never covers it; int8's
rounding costs more than its speed returns.

1. **Search speed is the largest lever in the project.** Ten percent more simulations
   is about 25 Elo. That reverses the ARENA9 dismissal of the expansion-path work: on a
   quiet core the Python around the network is about 10% of a simulation, worth about
   25 Elo, and any gain in cache hits, tree reuse or forward cost is worth the same
   rate. It also means the platform's absolute speed, still unread from the validation
   log, sets the whole field's strength: a core twice as slow as the box costs everyone
   about 180 Elo of absolute strength, and a net that is cheaper per node gains on one.
2. **R6a is the better network per position, by about 70 Elo.** It loses only because
   d128 costs 35% of the search on a quiet core. A d128 forward pass at d96 cost, through
   a cheaper trunk or a quantisation that spares the value head, would put it ahead of
   R1. Nothing available today does that; ARENA12 shows plain int8 is not it.
3. **Time management barely matters; node count does.** Phase J's front-loaded versus
   back-loaded allocation moved the score by +21 ± 48 Elo at equal total search; a
   halving moved it by 184.
4. **Same-net experiments need a wide book.** With one net on both sides and no
   pondering the agent is deterministic; eight positions give sixteen games however
   many are played. `openings102.tsv`, 102 positions at move 8 of distinct 2400+ games
   from the filtered corpus, is the book to reuse.
5. **Not measured:** the next doubling, 800 against 1,600, which says how fast the
   value of search falls off toward the box's budget, and pondering on against off,
   which the shipped agent relies on.

## 4. Decision this supports

**Optimise simulations per move before optimising the network, and read the platform's
budget from the validation log before deciding anything about size.** Sample: 204
distinct games at the competition clock with exact node caps, no failures, no
throttling. What would refine it: the 800-vs-1,600 doubling from the same book, and the
same design with pondering on.

## 5. Phase J3 — the measurement

The J2 design replayed from `openings102.tsv`: 102 positions at ply 16 of distinct
2400+ games sampled from the filtered Elite corpus (seed 7, one game in fifty, no
checks, no duplicates), each played with both colours across six lanes of 17 positions.
Caps 800 vs 400, pondering off, thresholds ±0.30, 2026-09-05 11:58–13:25 UTC, nothing
else on the box, 0 throttled periods.

```
games 204   800-side +107 =89 -8   score 74.3%   (95% Wilson 67.9%-79.8%)
implied 800 - 400: +184 Elo (95% +130 to +238)
800-side as White: +57 =42 -3, 76.5%      as Black: +50 =47 -5, 72.1%
lanes: 70.6% / 70.6% / 79.4% / 75.0% / 72.1% / 77.9%
terminations: checkmate 115, threefold 81, insufficient material 7, stalemate 1   (threefold 40%)
mean game 144 s, 104 plies
sims/move: 799 and 400 exactly, over 10,600 moves a side; pondered 0
distinct games: 204 of 204; failed games: none
```

The 800 side lost eight games in 204 and won from both colours at nearly the same rate.
J2's 71.8% from sixteen distinct games sits inside this interval, so the flawed run was
not wrong, only uninformative.

## 6. Artifacts

`sparring/bracket_box/lanes/{J,J2,J3}/`: per lane `results.csv`, `meta.json`, one PGN
and one log per game with both agents' per-move telemetry; `summary_local.txt`. Book
`sparring/bracket_box/openings102.tsv` (also `/workspace/bracket/openings102.tsv`).
Agents `/workspace/bracket/agents/r1_full_np`, `r1_half_np`, `r1_s800_np`, `r1_s400_np`;
queues `sparring/bracket_size/queue_j.sh`, `queue_j2.sh`, `queue_j3.sh`.
