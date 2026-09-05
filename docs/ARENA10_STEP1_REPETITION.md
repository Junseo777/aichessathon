# ARENA #10 — step 1: the referee's repetition rule and deeper opening adoption, on top of pruning

The first change to the shipped agent after the ladder's day one. Written to be read
cold. Branch `ship-step1` at `298ee09`; the upload is `aichessathon-step1/submission.zip`
built from it on 2026-09-05 at 02:05 (5,710,373 bytes unzipped, R1_e8_ema `ea52cc6a…`).

Date: 2026-09-04 20:45 to 2026-09-05 00:21, unattended, on the Apple M1 (8 cores) while
it also ran several Claude Code sessions and, for arena B, a fixed-count position
probe. Harness: `harness.play` at the competition clock, 120 s + 0.5 s, 300-ply cap,
`harness/rules.py` defaults. Openings: the eight positions in `sparring/openings.tsv`,
each with both colours. Two lanes of 25 in parallel (`sparring/feat/arena.sh`,
`summarise.py`). Net R1_e8_ema on every side.

---

## 1. The change

`rated-games/README.md` found that the platform declares a draw as soon as the side to
move *could* claim one (`board.outcome(claim_draw=True)` in `harness/referee.py`), while
`agent.py` only avoided moves that made a *third* occurrence. Round 8 was thrown that
way at +2.2 and round 1 handed over a draw at +0.6. Three edits, one commit:

| edit | where | what |
|---|---|---|
| root rule, winning | `agent.py` `_repeats` | at q > +0.3 avoid any move that makes a **second** occurrence, or lets the fifty-move count reach 98 |
| root rule, losing | `agent.py` `_referee_draws` | at q < −0.3 take a move after which the referee can claim: a third occurrence or the count at 100 by this move or by any reply |
| in the tree | `chessml/search.py` | a second occurrence on a search path is scored 0.0 (was the third), since a shuffling opponent can force the claim from there |
| opening adoption | `agent.py` `_find_in_tree` | the init pre-search tree is searched depth-first (4,096 nodes) for the received position; rated games start six to nine plies in, and the old one-ply match never fired |

`_referee_draws` is checked against `board.outcome(claim_draw=True)` move by move
over a shuffling sequence in `chessml/tests/test_agent.py`. 65 tests pass, ruff and
mypy strict clean. The three mate-finding tests fail on R1 before and after the change
and pass on R3, so they are not a regression signal.

## 2. Results

| arena | with | without | W-D-L (with) | score | 95% | Elo | failures |
|---|---|---|---|---|---|---|---|
| A | step 1 | the pre-pruning ladder zip (`main` f2a78b8, rounds 1 to 14) | +15 =34 −1 | **64.0%** | 57.2–70.8% | +100 (+50 to +154) | 0 |
| B | step 1 | the live upload (21:01, `main` 8de8562: pruning only, rounds 15 on) | +15 =28 −7 | **58.0%** | 49.1–66.9% | +56 (−6 to +122) | 0 |

**A** (20:45–22:26): as White +5 =19 −1 (58.0%), as Black +10 =15 −0 (70.0%);
terminations threefold 28, checkmate 16, insufficient material 6. Mean game 238 s.
This measures pruning and the repetition fixes together against what played rounds
1 to 14. The pruned side stopped 94% of its searches early, spent 1.52 s per move to
the other side's 2.40 s and still made more simulations (442 against 310 per move),
the ARENA #4 mechanism: the opponent's clock starts sooner and the ponder thread
fills the reused subtree. From move 40 the pruned side had 1.71 s per move to 1.15 s.
One caveat from the operator's notes: `with/agent.py` carried `proofs=True` for part of
this arena and was reset before B; proofs were null in both ARENA #4 chains, and
arena D below measures them alone.

**B** (22:32–00:21): as White +8 =11 −6 (54.0%), as Black +7 =17 −1 (62.0%);
terminations threefold 24, checkmate 22, insufficient material 2, fifty-move 1,
stalemate 1. Mean game 255 s. Both sides prune, so this is the repetition rule and
the opening adoption alone. Simulations per move 206 against 132 at the same 2.08 s
per move: the in-tree rule ends a path at the second occurrence without a forward
pass, so a repetition-heavy search runs more simulations for the same time. The
machine was heavily loaded during B (a load average near 40), so the absolute counts
are low; both sides shared the conditions.

## 3. Decision

**Ship step 1.** The rule set before the arena was 50% or better with no failures
against the live build; B is 58.0% with the interval's lower edge at 49.1%, and A
against the older code is significant on its own. The upload is
`aichessathon-step1/submission.zip`; it becomes the reference for the chain that
follows (ARENA #11 onwards: the clock formula, the pick rule, proofs, the d128 net).

What would change it: a loss in the rated games to a forced repetition the new rule
does not see, or a validation failure of the zip on the platform. The zip's `agent.py`
is byte-identical to the worktree's; `search.py` carries the `>= 1` line; the weights
hash to the R1_e8_ema entry in `weights/CHECKSUMS.txt`.

## 4. Artifacts

`sparring/step1/{with,without,live}` — the three agents (step 1, the pre-pruning zip,
the 21:01 upload), each with a symlink to R1_e8_ema.
`sparring/step1/A_vs_ladder/`, `sparring/step1/B_vs_live/` — `lane{1,2}/` with 25 PGNs
and logs each, `results.csv`, `run.log`; `summary.txt`.
`sparring/step1/queue.log` — the timeline. `B_vs_pruning_only_aborted_2games/` was
started against the wrong reference and stopped after two games; ignore it.
