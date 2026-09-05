# ARENA #12 — the submission on the Stockfish limit-strength ladder

An absolute reference for the build uploaded on 2026-09-05, measured by the ARENA #1
method on the RunPod box. Written to be read cold.

Date: 2026-09-05, 08:53–10:09 UTC. Box: RunPod, AMD Ryzen 9 7950X (16 physical cores,
13.6-core cgroup quota, 61 GB), lanes on physical cores 0–11, one per side; the R6b
trainer held cores 12–15 (two OMP threads) throughout. The cgroup's `nr_throttled`
counter read 538,463 before the first game and after the last, so the quota never bit. Harness: `/workspace/bracket/repo` pinned to `f2a78b8`, 120 s + 0.5 s,
300-ply cap, python-chess 1.11.2. Opponent: Stockfish 18, the official Linux
`x86-64-vnni512` release build (tar sha256 `91d89e0e…`, equal to the GitHub release
digest), `Threads=1`, `Hash=64`, `UCI_LimitStrength` at the rung, driven by
`sparring/stockfish-agent/agent.py` with the rung baked into its defaults (GPL, outside
the repo on purpose). Openings: the fifteen curated ladder positions in
`sparring/openings_ladder.tsv`, every one played with both colours at every rung.

Agent under test: `aichessathon/submission.zip` built 03:26 BST (md5 `93ce8c23`,
sha256 `94b90480…`) = `main` `bbc55a9`: R1_e8_ema (`ea52cc6a…`) with pruning 1.33,
policy temperature 1.359 and root FPU 1.0, draw thresholds ±0.3. One line changed for
the measurement: `_PONDER_NODE_BUDGET` 100_000 → 0, because the platform suspends the
process while the opponent thinks (rules.md, 2026-09-05), so pondered simulations never
happen there. Every one of the 90 game logs carries the init line with `ea52cc6a…`;
`pondered=0` on all 5,157 logged moves; zero failures.

---

## 1. The number

| rung | games | W–D–L | score | 95% Wilson | rung's own implied rating | as White | as Black |
|------|-------|-------|-------|------------|---------------------------|----------|----------|
| 2800 | 30 | +20 =9 −1 | 81.7% | 64.5–91.6% | 3060 | +9 =5 −1 | +11 =4 −0 |
| 3000 | 30 | +12 =15 −3 | 65.0% | 47.1–79.5% | 3108 | +7 =5 −3 | +5 =10 −0 |
| 3190 | 30 | +1 =17 −12 | 31.7% | 17.9–49.6% | 3056 | +1 =3 −11 | +0 =14 −1 |

Joint maximum-likelihood fit over all 90 games, draws scored 0.5:

**submission = 3077, 95% profile-likelihood interval 2998–3158.**

The three rungs agree with each other to within 52 points (3056–3108), unlike ARENA #1
where adjacent rungs were indistinguishable; the bot is bracketed by 81.7% at 2800 and
31.7% at 3190, so the ladder did not top out. Rung 3190 is the top of the
`UCI_LimitStrength` range but not full strength: Stockfish's own source (`search.h`,
`Skill`) maps `UCI_Elo` 3190 to skill level 18.4 of 20, which means the engine searches
four lines, takes its move from the depth-19 iteration and nudges the choice with a small
random term (`weakness = 120 − 2·level` ≈ 83). Full strength is `UCI_LimitStrength` off,
where the move comes from the deepest iteration reached, depth 25–30 at this clock on one
core. The bot held the depth-19 engine to 17 draws in 30.

Search budget: mean **1,337 simulations per move** (984 for moves 1–19, 1,046 for 20–39,
1,776 for 40–59, 1,672 from 60), the shipped clock formula at work. Mean game 279 s /
118 plies at 2800, 281 s / 122 at 3000, 261 s / 104 at 3190.

Terminations: 2800 checkmate 21, threefold 9; 3000 checkmate 15, stalemate 1,
threefold 14; 3190 checkmate 13, stalemate 2, threefold 15. The single loss at 2800
(ladder12 as White) was a real one: two pawn units against thirteen at the end.

## 2. What this scale is and isn't

Same method as ARENA #1 (R0 = 2868, 2786–2954) and as the reference project's 2,474, so
the numbers are comparable *in method*, not in conditions: ARENA #1 ran on an idle M1 Mac
from the standard start with R0 pondering on a second thread. This run is on the box
(the net forwards at ~2.3 ms per position here, about twice the Mac; Stockfish is faster
on Zen 4 than on the M1), from the curated ladder positions, with pondering off. Read the
rungs rather than the fitted figure when comparing: R0 scored 52.5% at 2800 and 40.0% at
3000 under its easier conditions; the submission scores 81.7% and 65.0% under harder
ones. Not a FIDE or Lichess rating; `docs/DECISIONS.md` §1 on why the scales do not
convert still applies. **The platform's match logs show about 500 simulations per move**
(read from the dashboard on 2026-09-05), so this run searched 2.7 times deeper than the
platform does, 1.4 doublings. If ARENA #10's 21 Elo per doubling holds, the platform-budget
figure sits a few tens of points lower on this scale; the honest correction is to re-run the
ladder with the bot capped at 500 simulations per move (`max_sims=500` at the `_MCTS.run`
call in `agent.py`, the clock left as a safety net), which is item 6 in
`docs/justins-simulations.md`.

## 3. Two things the data flags

**Conversion.** All three stalemates are the bot stalemating Stockfish while far ahead:
bishop and two pawns against a pawn (3000_b game 4, ladder10, ply 177), rook and knight
against a bare king (3190_a game 14, ladder07, ply 225), queen and pawn against a pawn
(3190_b game 6, ladder11, ply 121). Three more threefold draws ended with the bot ahead by
3, 5 and 8 pawn units. Six of 90 games, roughly 3 points of score and 40–60 Elo on this
ladder, are won positions handed back, and rook-and-knight against a bare king at ply
225 says the search cannot find a forced mate it has 100 plies to find. Cheapest fixes,
in order: veto any root move that stalemates when ahead (one legal-move check per
candidate), the moves-left incentive from IDEAS/G7, and the repetition rule's
`q_best` thresholds re-read against the value head's scale in won endings.

**Colour.** The bot scored better as Black at every rung, by 10, 3 and 30 points, and
the White side of these positions scored 45.0%, 48.3% and 35.0% whichever engine held
it. At 3190 the bot lost 11 of 15 as White and 1 of 15 as Black. Two readings fit:
the curated White sides are objectively worse, or the bot plays the White side badly;
ARENA #1 found the opposite sign from the standard start. Per-opening scores over six
games each run from 42% (ladder03, ladder10) to 83% (ladder06); nothing there is
significant on its own. Worth a same-conditions check with the bot on both sides of the
same positions (it is its own control there) before touching anything.

## 4. Caveats

- One trainer shared the box's CPU quota with the twelve lane cores; the throttle
  counter did not move, so neither side lost CPU to it. Sims per move are recorded above
  because the box's rate has drifted between phases on other days (700–1,500 per move).
- Stockfish's `UCI_Elo` rungs are calibrated for a particular hardware and time control
  and shift with both, and the top rung is skill level 18.4, not full strength (§1). A
  fixed-node ladder (`go nodes`) is the hardware-independent form, and a full-strength
  rung is the true ceiling; both are the right extension if this number is to be
  re-measured on another machine.
- 30 games per rung: the joint interval is ±80 points; per-rung intervals are wide.

## 5. Artifacts

Box: `/workspace/ladder/` (queue, lane runner, fit script, agent build with
`BUILD.txt`, `lanes/L/<rung>_<a|b>/` PGNs, per-game logs with both sides' stderr,
`results.csv`, `meta.json`, `summary.txt`). Pulled copy: `sparring/ladder_box/`
(outside git; `pull.sh` refreshes it). Queue log: `queue_ladder.log`.

## 6. Open items

All five are written up as runs for Justin's machine in `docs/justins-simulations.md`
(items 6–11 there), with the exact switches, game counts and what to read.

1. The same ladder with the bot capped at 500 simulations per move, the platform's
   budget (§2). This is the number that matters for the competition.
2. Stalemate veto and mate-finding in trivially won endings (§3), measured on a suite of
   won positions against full-strength Stockfish as the defender, not by arena score.
3. The colour asymmetry (§3): bot vs bot on the same fifteen positions, both colours.
4. Fixed-node rungs (Stockfish 18 at fixed `go nodes`) and one full-strength rung beside
   the UCI_Elo rungs, so the yardstick survives a change of machine and has a true ceiling.
5. R0 and the reference hero on this exact setup, to re-link ARENA #1 and the
   reference's 2,474 to today's conditions.
