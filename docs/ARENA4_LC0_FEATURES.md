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
| 5 | root FPU 1.0 | pruning | +7 =27 −16 | 41.0% | 31.9–50.1% | -63 | 54% | **dropped** |
| 6 | draw score 0.1 | pruning | +11 =30 −9 | 52.0% | 43.3–60.7% | +14 | 60% | **kept** |

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

**5. Root FPU 1.0, dropped at 41.0%** (+7 =27 −16, Elo -63, 95% -131 to +0;
as White +2 =15 −8, as Black +5 =12 −8; no failures). The worst result in either chain, and the
first whose interval only just reaches 50%. Forcing one visit
to each of ~30 root moves costs about 5% of the budget, which is small; the damage is
more likely that every root move then carries a real Q from a single visit, so a bad
move's one lucky evaluation can attract PUCT visits it would never have earned from
its prior. The blind-spot problem it was meant to insure against is already handled
by the first-play-urgency fix. Dropped.

**6. Draw score 0.1, kept at 52.0%** (+11 =30 −9, Elo +14, 95% -47 to +76; as White
+5 =17 −3, as Black +6 =13 −6; no failures). Inside noise both ways. Chain B dropped it on R1 at
48.1%. On R3 the ±0.3 gates fire in two fifths of positions, on R1 in half, and
neither shows a clear effect. Kept by the rule on R3, with no claim behind it.

**Chain A final configuration** (the driver's last line): `fpu_reduction=0.25`,
`pruning_factor=1.33`, `draw_score=0.1`, everything else off, `policy_temperature=1.0`.

### Chain B, base R1 (box)

Base net: **R1_e8_ema** (`ea52cc6a…`). 52 games per feature (four lanes of 13),
otherwise as chain A.

| # | feature | kept so far | W-D-L (with) | score | 95% | Elo | draws | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | proofs | none | +2 =48 −2 | 50.0% | 46.2–53.8% | 0 | 92% | **kept** (tie) |
| 2 | smart pruning | proofs | +19 =32 −1 | 67.3% | 60.3–74.3% | +125 | 62% | **kept** |
| 3 | scaled FPU 0.33 | proofs, pruning | +10 =31 −11 | 49.0% | 40.4–57.7% | -7 | 60% | **dropped** |
| 4 | policy temperature 1.359 | proofs, pruning | +12 =39 −1 | 60.6% | 54.4–66.7% | +75 | 75% | **kept** |
| 5 | root FPU 1.0 | proofs, pruning, temperature | +9 =39 −4 | 54.8% | 48.1–61.5% | +34 | 75% | **kept** |
| 6 | draw score 0.1 | proofs, pruning, temperature, root FPU | +5 =40 −7 | 48.1% | 41.6–54.6% | -13 | 77% | **dropped** |

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

**4. Policy temperature 1.359, kept at 60.6%** (+12 =39 −1, Elo +75, 95% +31 to
+121; as White +6 =20 −0, as Black +6 =19 −1; two-sided binomial on the thirteen decisive games
p = 0.003; no failures). **The first disagreement between the chains**: chain A dropped
the same setting on R3 at 47.0% (+11 =25 −14). The two chains differ in two things at
once, the net (R1's outcome-trained value head against R3's engine-trained one) and
the budget (~1,400 against ~600 simulations per move), so this arena alone cannot
say which one flipped the sign. Both mechanisms are plausible: flatter priors spend
more of the search on second choices, which a hotter value head can rank and a
larger budget can afford. The way to separate them is one 50-game R1 chain at the
Mac's budget, or R3 at the box's. Until then the setting is net-specific: on for R1,
off for R3.

**5. Root FPU 1.0, kept at 54.8%** (+9 =39 −4, Elo +34, 95% -13 to +81; as White
+5 =18 −3, as Black +4 =21 −1; two-sided p = 0.27 on the thirteen decisive games; no failures). The
second disagreement, in the same direction as the first: chain A dropped it on R3 at
41.0%, its worst result. On its own this one is inside noise; together with the
temperature it makes a pattern. Both knobs widen the search at the root, and both
help R1 at ~1,400 simulations per move while hurting R3 at ~600. A wider root costs
depth, and depth is what a small budget cannot spare; a hot value head ranks the
extra candidates more decisively than a cool one. Either explanation fits, and the
same single arena would separate them. Net-specific for now: on for R1, off for R3.

**6. Draw score 0.1, dropped at 48.1%** (+5 =40 −7, Elo -13, 95% -59 to +32; as
White +4 =20 −2, as Black +1 =20 −5; no failures). Twelve decisive games in fifty-two, split five to
seven. R1's hot value head crosses the ±0.3 gates often, so the in-tree contempt was
live in most positions, and it did not help. Dropped.

**Chain B final configuration** (the driver's last line): `fpu_reduction=0.25`,
`proofs=True`, `pruning_factor=1.33`, `root_fpu=1.0`, `policy_temperature=1.359`,
`fpu_scaled=False`, `draw_score=0.0`.

### Disambiguation: net or budget?

Chains A and B disagreed on policy temperature and root FPU, and differed in net and
budget at once. This arena holds the net at R1 and lowers the budget to the Mac's:
R1 with both knobs (temperature 1.359, root FPU 1.0) against R1 without, smart pruning
on both sides, proofs off, at **60 s + 0.25 s** on the box (421 simulations per move
measured, against ~1,400 at the full clock there and ~600 on the Mac). Four lanes of
13 on cores 4–11, 2026-09-05 01:44–02:15 UTC.

| | games | W-D-L (knobs on) | score | 95% | Elo |
|---|---|---|---|---|---|
| all | 52 | +11 =37 −4 | **56.7%** | 49.7–63.8% | **+47** (−2 to +98) |
| knobs as White | 26 | +6 =19 −1 | 59.6% | | |
| knobs as Black | 26 | +5 =18 −3 | 53.8% | | |

No failures; 31 threefold, 15 checkmates, 6 insufficient material.

**Reading.** At a Mac-like budget the knobs are still positive on R1 (+47, the interval
just touching zero), where on R3 at that budget they were clearly negative (47.0% and
41.0%). Across the three R1 arenas, 156 games, they never had a losing record:
60.6% (temperature alone, 1,400 sims), 54.8% (root FPU on top, 1,400 sims), 56.7% (both,
421 sims). So the disagreement was the **net**, not the budget: a wider root suits the
outcome-trained value head, which ranks the extra candidates decisively, and not the
engine-trained one. Consequence: the two knobs are safe to ship with R1 at either
budget and probably worth about +50; they should not ship with R2, R3 or R5.

## 4. What to ship

Side by side, the two chains agree on three features and disagree on two:

| feature | R3 at ~600 sims (Mac) | R1 at ~1,400 sims (box) | verdict |
|---|---|---|---|
| smart pruning | +49, 57.0% | +125, 67.3% (p < 0.0001) | **ship on any net** |
| proofs | −7, 49.0% | 0, 50.0% | null; no reason to carry it |
| scaled FPU 0.33 | −28, 46.0% | −7, 49.0% | drop; the constant 0.25 stays |
| policy temperature 1.359 | −21, 47.0% | +75, 60.6% (p = 0.003) | net- or budget-specific |
| root FPU 1.0 | −63, 41.0% | +34, 54.8% | net- or budget-specific, same direction |
| draw score 0.1 | +14, 52.0% | −13, 48.1% | null either way |

**Pruning is the result of this run.** It is the only feature that won on both nets,
it did so by the largest margins in either chain, and its mechanism is visible in the
telemetry (96–98% of searches stopped early, the saved clock spent after move 40). It
is a clock change, not a search-quality change, so it does not depend on the net or
the budget. It should go to `main` regardless of which net ships.

**The two exploration knobs are net-specific, and the disambiguation arena above
settled it.** Temperature and root FPU widen the root; they help R1 at both the box's
and the Mac's budget (+75, +34, +47) and hurt R3 (−21, −63). With R1 as the shipped
net they are a probable +50 at no measured risk; with an engine-labelled net they
stay off.

**Defaults on `main`** are the set both nets support: `pruning_factor=1.33`, all
else off, temperature 1.0. The box's arm A has since shown R1's edge over R2 survives with the draw rule off
(ARENA #3 addendum), so R1 is the net to plan around. The per-net lines from the
drivers are recorded above for whichever net is chosen; the final step before the zip is one 50-game confirmation of
the chosen line on the chosen net at the Mac's budget.

## 5. Artifacts

`sparring/feat/<k>_<name>/{with,without}` — the two agents, each a copy of the
snapshot with its own `agent.py` line and a symlink to the base net.
`sparring/feat/<k>_<name>/lane{1,2}/` — 25 PGNs and logs each, `results.csv`, `run.log`.
`sparring/feat/decisions.txt` — the keep/drop line per feature as the driver wrote it.
`sparring/feat/driver.log` — the driver's timeline.

Chain B lives on the box under `/workspace/lc0feat/`: `<k>_<name>/{with,without,lane1..4}`,
`decisions.txt`, `driver.log`, and `harness/` (a copy of this repo's harness, because
the box's checkout predates `--fen`). The box scripts are mirrored in
`sparring/feat/box/`.
