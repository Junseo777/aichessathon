# ARENA #13 — step 2: the clock formula, dropped at 44.0% in a measurement a tenth of the platform's speed

Continues `docs/ARENA10_STEP1_REPETITION.md`. Written to be read cold. Branch `ship-chain`;
the change is commit `591845d`, reverted to the old constants in `60b7bdf` after this arena.

Date: 2026-09-05, 09:10 to 11:18 BST, unattended, on the Apple M1. Harness: `harness.play`
at the competition clock, 120 s + 0.5 s, 300-ply cap. Openings: the eight positions in
`sparring/openings.tsv`, both colours, two lanes of 25 (`sparring/feat/arena.sh`). Net
R1_e8_ema on both sides. Driver: `sparring/chain/driver.py`, candidate built from the
branch snapshot with three constants changed; the reference is step 1 itself
(`sparring/step1/with`, the shipped zip's code).

---

## 1. The change

`rated-games/README.md`: the shipped `_budget_s` spends `left / max(14, 46 − move) + 0.4`,
which plays every game down to two seconds (23 s at move 40, 12 s at move 50) and put
every error above 12% win probability on a move made with under a second. Round 14 was
thrown at move 45 with 15.6 s left.

Proposed: `left / max(20, 60 − move) + 0.4`, at least 1.0 s while more than 15 s remain,
the 4 s cap and the `left − 1` guard unchanged. On round 14's own clock trace, spending
the whole budget every move with 10 to 20 ms of platform overhead, it reaches move 45
with 38 s; the old constants reproduce the game's 16.7 s to within 2 s (both are tests in
`chessml/tests/test_agent.py`). The 1 s floor only binds early after a slow start; the
visible effect is 2.8 s instead of 3.6 s per move in the opening, and more from move 40.

## 2. Result

| with | without | W-D-L (with) | score | 95% | Elo | failures |
|---|---|---|---|---|---|---|
| 60 / 20 / 1.0 s | step 1 (46 / 14 / none) | +10 =24 −16 | **44.0%** | 34.1–53.9% | −42 (−114 to +27) | 1, on the reference side |

As White +7 =11 −7 (50.0%), as Black +3 =13 −9 (38.0%); lanes 52% and 36%.
Terminations checkmate 25, threefold 19, insufficient material 4, stalemate 1, flag 1.
By opening: King's Indian 64%, Slav 62%, Caro-Kann 50%, Ruy 43%, Najdorf 38%, QGD 38%,
English 25%, Winawer 12%. Mean game 300 s, longest 374 s.

**The mechanism worked as designed.** Time per move by bracket, with / without:

| moves | 1–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60+ |
|---|---|---|---|---|---|---|
| with | 2.76 s | 2.69 s | 2.58 s | **2.10 s** | **1.42 s** | 0.91 s |
| without | 3.46 s | 3.36 s | 2.80 s | 1.55 s | 0.94 s | 0.72 s |

The candidate banked about 0.7 s a move for the first thirty moves and spent it from
move 40. The six lowest end-of-game clocks in the arena (6.3 to 6.8 s by the agents' own
timing) belong to the reference on five of six, and the one flag was the reference, as
White in lane 1 game 10, with 6.7 s of self-measured time left after 63 moves: about
100 ms a move had gone to harness and process overhead outside the agent's clock.

**The measurement is not the platform's.** Both sides made **19 simulations per move**
(6 to 13 per second; `forward_ms` at init 48 to 200 ms against 3.6 to 4.0 idle). Arena B
the night before ran at roughly 100 per second and the ladder at about 100. The Mac's
load average was 40 to 60 throughout from WindowServer, the Claude desktop renderer,
`mobileassetd` and `modelcatalogd`, none of them arena processes, and it got worse over
the two hours (the candidate's rate fell from 13 to 6 per second). At twenty simulations
a move the search is close to the raw policy and the value head barely enters; the
question the change asks, whether extra seconds late in the game convert at the
platform's depth, was not put.

## 3. Decision

**Dropped**, by the rule set before the games: 44.0%, below 50%. The branch defaults are
back at 46 / 14 / no floor (commit `60b7bdf`); the constants stay lifted to the top of
`agent.py` and the proposed formula's round 14 test runs under a monkeypatch, so the
change is one line away.

What the result is not: evidence against the formula. The interval reaches 53.9%, the
time redistribution is confirmed, and the regime was a tenth of the ladder's search
speed. What would settle it: the same 50 games at 100 or more simulations per second,
on the box or on this Mac when it is quiet, before any other clock change is measured.

## 4. Artifacts

`sparring/chain/2_clock/with/` — the candidate (snapshot `943df1d`+`591845d` with
`CONFIG.txt`); `lane{1,2}/` with 25 PGNs and logs each, `results.csv`, `run.log`;
`summary.txt`. `sparring/chain/chain.log`, `decisions.txt` — the driver's timeline; its
"failures 1" counts the reference's flag. `sparring/chain/positions_z.log` — the
fixed-simulation probe of the README positions used to set the step-3 constants; the
step-3 and step-4 arenas were cut from this chain on Junseo's instruction as duplicates
of `sparring/step1/queue3.sh` (E extension 57.0%, F lower bound running) and the box's
ARENA #9.
