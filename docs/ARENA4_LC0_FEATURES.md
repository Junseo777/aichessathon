# ARENA #4 — six Lc0 search features, each measured against the search without it

Continues `docs/ARENA3_R1_R2_R3.md` (the overnight block) and `docs/FINDING_fpu.md`.
Written to be read cold. Branch `search-lc0`, one commit per feature; `main` and the
shipped zip untouched until this report is read.

Date: 2026-09-04/05, unattended overnight run on the Apple M1 (8 cores).
Harness: `harness.play` at the competition clock, 120 s + 0.5 s, 300-ply cap,
`harness/rules.py` defaults. Openings: the eight positions in `sparring/openings.tsv`,
each played with both colours. Two lanes in parallel (four agent processes), the
parallelism ARENA #2 measured as free of contention on this machine.

---

## 1. Method

Every feature is a keyword argument of `chessml.search.MCTS` (or, for the policy
temperature, of `chessml.net.load_fastest`), so any combination can be built from one
code snapshot (`0d91963`) by changing one line of `agent.py`. `sparring/feat/driver.py`
does that: for feature *k* it plays **50 games** of *kept-so-far + k* against
*kept-so-far*, both on the base net chosen by the block (the R2 vs R3 winner; a tie
keeps R2), and **keeps the feature if it scores 50% or better.** Kept features
stack; dropped ones are left out of every later comparison.

The rule was set before any game was played, and 50 games at this clock resolve about
±70 Elo at 95%, so a "kept" verdict means *no evidence of harm and a non-losing
record*, not proof of gain. The keep rule is the user's; the point estimates and
intervals are reported alongside so the reader can apply a stricter one.

The base configuration is the search as it ran in the block (`docs/FINDING_fpu.md`
fix: constant FPU reduction 0.25, evaluation cache), which is what the block's
numbers and ARENA #2 describe.

## 2. The features, in the order tested

| # | feature | switch | what Lc0 does | what changes here |
|---|---|---|---|---|
| 1 | proven results | `proofs=True` | terminal bounds propagate; a node with a proven-lost child is solved | exact values (mate, stalemate, fifty-move, insufficient material) cascade up the path; a proven root stops the search; `_pick` prefers proven wins and refuses proven losses. Derived proofs are valid for one search only, because a repetition claim can void them after the game moves on |
| 2 | smart pruning | `pruning_factor=1.33` | stop when no move can overtake the leader in the remaining time | checked every 32 simulations from the current rate; time-based only, so fixed-count searches and pondering are untouched; the move log says `pruned` |
| 3 | scaled FPU | `fpu_reduction=0.33, fpu_scaled=True` | reduction × sqrt(visited policy mass), 0.33 | a fresh node has no reduction, the reduction grows as its policy is explored; replaces the constant 0.25 |
| 4 | policy temperature | `policy_temperature=1.359` | logits / 1.359 before the softmax | flatter priors; the order of moves is unchanged |
| 5 | root FPU | `root_fpu=1.0` | separate FPU at the root | every root move is tried once before any is repeated; about 30 of ~500 simulations |
| 6 | draw score | `draw_score=0.1` | draws scored inside the tree; WDL contempt | when the root's standing is above +0.3 a drawn leaf is worth −0.1 to the root's side, below −0.3 it is +0.1; proofs unchanged |

Unit evidence, all on the branch: mate in two is proven in under 2,000 simulations and
the search stops on its own; mate in one is proven on the first visit of the mating
move; `_pick` prefers a proven win, refuses a proven loss, and still vetoes a proven
win that hands the referee a claim; pruning fires exactly at the first check when the
inherited lead is unbeatable and never without a deadline; the scaled reduction is 0 on
a fresh node and 0.25·√0.64 after one child with prior 0.64 is visited; the temperature
lowers the maximum prior and raises the minimum without reordering; absolute root FPU
leaves no root move unvisited where the default leaves several; a repetition is worth
−0.1 to a root standing at +0.9 and +0.1 at −0.9. 62 tests pass, ruff and mypy strict
clean, at every commit.

## 3. Results

Two chains, same snapshot, same features, same rule. Chain A on the Mac with base
**R3** (the block's pick); chain B on the RunPod box with base **R1** (the block's
strongest net, added on Junseo's instruction at 15:48), four lanes of 13 games each
pinned to physical cores with `taskset`, the harness copied from this repo because
the box's checkout predates `--fen`. The box runs the net at 2.3 ms per forward
against the Mac's 3.6–4.0 ms, so chain B plays at roughly twice the simulations per
move; the ladder's hardware is unknown, so neither is "the" budget.

### Chain A, base R3 (Mac)

Base net: **R3_e8_ema** (`a1163b19…`), chosen by the block (`docs/ARENA3_R1_R2_R3.md`:
R2 49.0% over 98 games, so R3 by the higher-score rule). Every row is 50 games at
120 s + 0.5 s, two lanes of 25, colours balanced, against the kept configuration at
that point.

| # | feature | kept so far | W-D-L (with) | score | 95% | Elo | draws | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | proofs | none | +5 =39 −6 | 49.0% | 42.5–55.5% | −7 | 78% | **dropped** |
| 2 | smart pruning | none | +15 =27 −8 | 57.0% | 47.8–66.2% | +49 | 54% | **kept** |
| 3 | scaled FPU 0.33 | pruning | +8 =30 −12 | 46.0% | 37.3–54.7% | −28 | 60% | **dropped** |
| 4 | policy temperature 1.359 | pruning | +11 =25 −14 | 47.0% | 37.2–56.8% | -21 | 50% | **dropped** |

**1. Proofs, dropped at 49.0%.** Eleven decisive games in fifty, five to six, all by
checkmate; the rest threefold repetitions but one. As White the feature side went
+0 =21 −4, as Black +5 =18 −2, which is colour noise at n = 25, not a colour effect.
No failures. The only tracebacks in the logs are the known post-mate ponder
`ValueError` on the winner's side, twelve of them, one per checkmate. Nothing here
says proofs hurt; nothing says they help either. On a same-net mirror at this clock
78% of games are drawn, so the 95% interval is ±6.5 points of score and a feature
can register only through conversions, of which there were eleven. Proven results
were designed to matter in exactly those, and did not produce more of them.

**2. Smart pruning, kept at 57.0%** (+15 =27 −8, Elo +49, 95% −15 to +117; as White
+8 =14 −3, as Black +7 =13 −5; no failures). Not significant at 5%, but the mechanism
is visible in the telemetry and points one way. The pruned side stopped 98% of its
searches early and spent 1.41 s per move against the other side's 2.29 s, and still
got *more* simulations per move, 932 against 589: returning the move sooner starts
the opponent's clock sooner, the ponder thread then fills the cache and the reused
subtree for the position that actually arrives, and the next search runs largely on
hits. The banked clock shows up late: from move 40 onwards the pruned side spent
1.68 s per move to the other side's 1.11 s, because `_budget_s` scales with the time
left. That is where games are decided, and this arena had 23 decisive games to the
proofs arena's 11, with the draw rate down from 78% to 54%. Games averaged
241 s against 286 s, so the rest of the chain runs faster than planned.

**3. Scaled FPU (Lc0's 0.33 × √visited policy), dropped at 46.0%** (+8 =30 −12, Elo −28,
95% −90 to +33; as White +5 =14 −6, as Black +3 =16 −6; no failures). Both sides had
pruning, and the telemetry is symmetric (479 vs 458 simulations per move, 1.78 s
per move each), so this is the FPU form alone. The twelve losses are spread over all
eight openings and both colours, which is the profile of a slightly worse search
rather than a blind spot. On this net at ~480 simulations per move the scaled form
starts softer than the constant 0.25 (no reduction on a fresh node) and ends
harder (0.33 once the policy is explored); the probe in `docs/FINDING_fpu.md`
predicted the trade, not its sign. Not significant, and the rule drops it. The
constant 0.25 stays.

**4. Policy temperature 1.359, dropped at 47.0%** (+11 =25 −14, Elo -21, 95%
-91 to +47; as White +6 =12 −7, as Black +5 =13 −7; no failures). The most
decisive arena in chain A (25 of 50), which is what flatter priors do: the search
spends more of its budget on second and third choices, so both sides find and allow
more. It did not translate into a better record. Not significant; dropped by the
rule, and the raw softmax stays.

### Chain B, base R1 (box)

Base net: **R1_e8_ema** (`ea52cc6a…`). 52 games per feature (four lanes of 13),
otherwise as chain A.

| # | feature | kept so far | W-D-L (with) | score | 95% | Elo | draws | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | proofs | none | +2 =48 −2 | 50.0% | 46.2–53.8% | 0 | 92% | **kept** (tie) |
| 2 | smart pruning | proofs | +19 =32 −1 | 67.3% | 60.3–74.3% | +125 | 62% | **kept** |
| 3 | scaled FPU 0.33 | proofs, pruning | +10 =31 −11 | 49.0% | 40.4–57.7% | -7 | 60% | **dropped** |

**1. Proofs, kept at exactly 50.0%** (+2 =48 −2, no failures). Four decisive games
in fifty-two. An R1 mirror at the box's ~1,400 simulations per move is draw-saturated,
so the arena has almost no power and the verdict is the rule's tie-break, not
evidence. Chain A dropped the same feature at 49.0%; both are nulls, and the
difference between "kept" and "dropped" here is one game.

**2. Smart pruning, kept at 67.3%** (+19 =32 −1, Elo +125, 95% +73 to +184; as White +11 =14 −1, as Black +8 =18 −0; two-sided
binomial on the twenty decisive games p < 0.0001; no failures). The same feature scored
57.0% on R3 at the Mac's budget; on R1 at the box's it is the largest effect in either
chain. The mechanism is the one chain A saw, with more room to work: R1's hotter value
head converts when it has time, and pruning gives it time late in the game.
The telemetry differs from chain A in one respect. The pruned side stopped 96% of its
searches and spent 1.13 s per move to the other side's 2.22 s, but this time it made
*fewer* simulations per move, 810 against 1,122: at the box's speed the unpruned
side's longer searches outrun what pondering can pre-fill. It won anyway, 19 to 1,
and the clock is where: from move 40 onwards the pruned side had 1.39 s per move to
1.06 s. So on R1 the gain is time management alone, with no simulation bonus, and it
is larger than on R3. Games averaged 222 s.

**3. Scaled FPU, dropped at 49.0%** (+10 =31 −11, Elo -7, 95% -68 to +54; as White
+4 =22 −0, as Black +6 =9 −11; no failures). Chain A dropped it at 46.0% on R3. Two nets, two
budgets, the same answer: Lc0's form of the reduction is not better than the constant
0.25 here, and both chains say so from inside their intervals rather than by a tie.

## 4. What to ship

SHIP_PENDING

## 5. Artifacts

`sparring/feat/<k>_<name>/{with,without}` — the two agents, each a copy of the
snapshot with its own `agent.py` line and a symlink to the base net.
`sparring/feat/<k>_<name>/lane{1,2}/` — 25 PGNs and logs each, `results.csv`, `run.log`.
`sparring/feat/decisions.txt` — the keep/drop line per feature as the driver wrote it.
`sparring/feat/driver.log` — the driver's timeline.
