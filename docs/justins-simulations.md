# Arenas to run on Justin's PC

A prompt for a coding agent running on Justin's machine. Copy everything below the line
into the agent's first message, together with the transfer bundle described in section 2.

---

You are running chess-engine arenas for an AI Chessathon entry. The team's own machines
are unusable for this right now: the laptop that measured the last result was thrashing at
a tenth of the competition platform's search speed, and the training box is busy. Your job
is to play a fixed list of 50-game matches on this PC, one at a time, apply a fixed
keep-or-drop rule to each, and write the results up so someone who was not here can read
them cold. Nothing here changes the code except one-line switches that already exist.

## 1. What is being tested, and why

The agent is `agent.py` plus `chessml/` (a small ONNX policy-value net, 1.4M params,
searched by PUCT MCTS). The platform gives each side 120 s + 0.5 s per move on one core,
about 10 ms per simulation. The shipped build ("step 1") is the reference. Each candidate
below is the reference with one switch flipped at the top of `agent.py`:

| priority | candidate | switch | why it needs measuring |
|---|---|---|---|
| 1 | clock formula | `_BUDGET_HORIZON = 60`, `_BUDGET_DIVISOR_FLOOR = 20`, `_BUDGET_FLOOR_S = 1.0` | The shipped formula plays every game down to 2 s and every serious rated-game error came on a move made with under a second. The new one reaches move 45 of the worst game with 38 s instead of 17. It scored 44.0% on the laptop, but both sides were making 19 simulations a move, so the result means nothing. This is the one rerun that matters most. |
| 2 | lower-bound pick | `_LCB_Z = 5.0` | In every replayed rated-game error the most-visited move had a lower q than a rival with at least 20% of its visits. Picking by q minus 5 standard errors chose the engine-approved move in 6 of 7 such positions offline. Never measured at the real clock. |
| 3 | search extension | `_EXTEND_FACTOR = 2.0` | Same signal, different remedy: keep searching up to twice the budget while the visit leader and the best-q candidate disagree. Offline it rarely resolved the disagreement; a single 1x extension scored 57.0% in another queue. |
| 4 | both, if 2 and 3 both pass | `_LCB_Z = 5.0` and `_EXTEND_FACTOR = 2.0` | Only if each passed alone. |
| 5 | mirror calibration | none (reference vs itself) | 50 games of the reference against an identical copy. Tells us this machine's draw rate and how wide the noise is, so the intervals above can be read. Run it only after 1 to 3 are done. |

Do not test the d128 net (it lost 40.5% on the box), and do not test proofs
(`proofs=True`; 49%, 50% and 52% in three arenas, null).

**The keep rule, fixed in advance:** a candidate is kept if it scores 50% or better over
its 50 games with no failure on its own side (crash, illegal move, flag, init failure).
50 games resolve about ±70 Elo, so "kept" means no evidence of harm, not proof of gain.
Report the point estimate and the interval either way.

## 2. What you were given

- `aichessathon/` — the repo checked out at branch `ship-chain`, commit `70f8a38`. The
  code is `agent.py`, `chessml/`, `harness/` (the referee that mirrors the platform's
  clock and protocol; do not edit it), `docs/` (read `docs/ARENA10_STEP1_REPETITION.md`
  and `docs/ARENA11_STEP2_CLOCK.md` for the report format and the story so far),
  `AGENTS.md` (the repo's rules; read it first).
- `weights/R1_e8_ema/model.onnx` and `manifest.json` — the net. Its sha256 must be
  `ea52cc6aac7704abec84b8628a86cbd9df8999ea178bcf75c199a8f245c9ac5a`; check it.
- `sparring/openings.tsv` — eight book openings, one per line, `name<TAB>fen`.
- `sparring/feat/arena.sh` and `sparring/feat/summarise.py` — one arena lane, and the
  scorer. `sparring/chain/driver.py` — the laptop's sequential driver, macOS-specific
  (`caffeinate`, `pgrep`); adapt it or write your own.

## 3. Setup and checks, in this order

1. Python 3.12. In `aichessathon/`: `uv sync --frozen --no-group train`, or a venv with
   `chess==1.11.2 numpy==2.5.2 numba==0.67.0 onnxruntime==1.29.0 torch==2.13.0` (CPU).
2. Link or copy the net: `aichessathon/weights/model.onnx` and `weights/manifest.json`
   from `weights/R1_e8_ema/`. `use-weights.sh` does this if the run store sits beside the
   repo as `../weights/`.
3. `ruff check .`, `mypy`, then `pytest -q chessml -k "not mate and not deadline and not smart_pruning"`
   must be green (64 tests). The excluded six are timing tests and three mate-finding
   tests that fail on this net before and after every change; run them once anyway and
   record the outcome.
4. **Measure the machine.** With nothing else running, play one game of the reference
   against itself with `python -m harness.play --white . --black . --fen "<any opening>"`
   and read the `init:` line (`forward_ms`) and the per-move `sims=` and `t=` from stderr.
   Report forward ms, simulations per second, and the count of physical cores. The
   platform is about 10 ms per simulation; the laptop's good arenas ran at 100 per
   second. If this PC is much faster than the platform, say so in every report; do not
   try to throttle it.
5. Decide the lane count: one lane per two free physical cores (each game is two
   single-threaded agent processes plus a referee). Two lanes is the usual. Never share
   cores with anything else, and keep the machine from sleeping.

## 4. Building a candidate

A candidate directory is `agent.py`, `chessml/*.py` (not `chessml/tests`), and a
`weights/` with the two files. Copy the reference, then change exactly the switch lines
in section 1; `diff -r` against the reference must show only those lines. Write a
`CONFIG.txt` saying what was changed. The reference directory is the unchanged copy.
The switches, with their reference values, are at the top of `agent.py`:

```
_BUDGET_HORIZON = 46
_BUDGET_DIVISOR_FLOOR = 14
_BUDGET_FLOOR_S = 0.0
_CANDIDATE_SHARE = 0.2
_EXTEND_FACTOR: float | None = None
_LCB_Z: float | None = None
```

## 5. Running an arena

50 games per candidate: two lanes of 25, or one of 50. Every opening is played with both
colours; the lane script does this when given the offset (0 for lane 1, 1 for lane 2).
`arena.sh <out-dir> <candidate-dir> <reference-dir> <games> <offset>` writes one PGN
and one log per game and a `results.csv` with `score_with` from the candidate's side.
If bash is not available, write the equivalent in Python: same columns, same opening
rotation (`opening = ((game_number - 1) // 2) % 8`, candidate is White on odd games),
same `harness.play` call with `--fen` and `--pgn`. Then
`python sparring/feat/summarise.py lane1/results.csv lane2/results.csv > summary.txt`.

Run arenas strictly one after another, priority order, and never two at once. Check the
first two games of each arena by hand: both agents print an `init:` line with the net's
sha256, moves print `sims=`, and the game ends with a result line, not a traceback.

## 6. Reporting

For each arena, one file `docs/ARENA<n>_<NAME>.md` in the repo (start at 13; ARENA #12 is
the Stockfish ladder already in `docs/`), in the
format of ARENA10 and ARENA11: what was tested and why, the result table (W-D-L, score,
95% interval, Elo, failures and which side), colour split, terminations, mean game
length, simulations and seconds per move for each side in move brackets (0–19, 20–29,
30–39, 40–49, 50–59, 60+), the machine's speed, and a decision section that says kept or
dropped by the rule and what would change the reading. Commit each report on a branch
`justin-arenas` off `ship-chain`, conventional commit style (`docs: ARENA #12, ...`),
rationale in the body, no co-author trailers. Do not push. When done, hand back a git
bundle of the branch and the `sparring/` output directories (PGNs, logs, results,
summaries) as a zip.

Also keep a running `sparring/justin/decisions.txt`, one line per arena:
`<name>: <n> games, with <score>% (+w =d -l), failures <k> -> KEPT|DROPPED`.

## 7. Things that will go wrong

- The net's `forward_ms` at init decides which ONNX file is used; only `model.onnx`
  exists, so this is informational.
- A game that ends "by flag" or "by crash" on the reference side is not a candidate
  failure, but say which side it was.
- An arena whose per-move simulation counts are far below the machine's idle rate was
  contended; stop, find what else is running, and rerun it. The laptop's 44.0% is
  exactly the artefact to avoid.
- Games from a fixed eight-opening book replay each other; count distinct games and
  say so if fewer than 40 of 50 are distinct.
- The repo's `make zip` is not your job. Nothing you run should touch `main` or any
  upload.

## 8. Added 2026-09-05 after ARENA #12: the Stockfish ladder follow-ups

Read `docs/ARENA12_STOCKFISH_LADDER_SUBMISSION.md` first. It measured the 03:26 upload (R1
with pruning, policy temperature and root FPU) at 3077 (2998–3158) on Stockfish 18's
`UCI_LimitStrength` rungs 2800/3000/3190, 90 games, 120 s + 0.5 s, pondering off, on the
training box. Two things make that number provisional and two findings need their own runs,
so the items below go after items 1 to 5 in priority, except item 6, which comes right after
item 1. Everything in this section is a measurement, not a keep-or-drop: report what you
read and how it moves the ARENA #12 figure.

**Fixed facts you need.** The platform's own match logs show the agent's 5-second init
pre-search at 768–1,070 simulations, against 1,664–2,304 in the box's ladder games, so the
platform runs the net at 0.37–0.51 of the box's speed and the box's 1,337 simulations per
move correspond to about **500 per move on the platform** (490–680); use 500, the low end. The bot is capped at a fixed number of simulations by one
argument at the search call in `agent.py`, `result = _MCTS.run(board, game.key_counts,
deadline, root=root)` → add `max_sims=500`; the clock stays as a safety net. Stockfish's
`UCI_Elo` 3190 is *not* full strength: in `search.h`, `Skill` maps 3190 to level 18.4 of 20,
so the engine searches four lines, takes its move from the depth-19 iteration and nudges
it with a random term; full strength is `UCI_LimitStrength` off. Stockfish's rungs are
calibrated for a time control and a machine and shift with both; `go nodes N` is the
machine-independent form.

| priority | item | setup | why | what to read |
|---|---|---|---|---|
| 6 (run right after item 1) | **ladder at the platform's budget** | reference with `max_sims=500` vs Stockfish 18 at `UCI_Elo` 2800, 3000, 3190; 30 games per rung, the fifteen curated positions in `sparring/openings_ladder.tsv`, each with both colours; then, if the 3190 score is 50% or more, a fourth rung with `UCI_LimitStrength` off | ARENA #12 searched 2.7x deeper than the platform; with the cap, the bot's side no longer depends on this PC's speed, so this is the competition-relevant absolute number | per-rung W-D-L and score, the joint fit from `ladder_fit.py`, and the bot's mean `sims=` per move (must read ~500); compare rung by rung with ARENA #12 §1 |
| 7 | **fixed-node rungs** | Stockfish at `go nodes` N (`Threads=1`, `Hash=64`, `UCI_LimitStrength` off; in the wrapper replace the clock `Limit` with `chess.engine.Limit(nodes=N)`), reference with `max_sims=500`. Probe first: 8 games at N = 200k; above 70% multiply N by 4, below 30% divide by 4, until bracketed; then 30 games at three rungs a factor of 4 apart around the crossover | both sides then have fixed budgets, so the result reproduces on any machine and has no calibration ceiling; it becomes the yardstick for every later build | score per rung and the N where the bot scores 50%; run the 3000 `UCI_Elo` rung in the same session so the two scales can be linked once |
| 8 | **colour check on the curated positions** | reference vs an identical copy on `sparring/openings_ladder.tsv`, every position twice with each colour, 60 games; this can double as item 5's calibration if run instead of it | ARENA #12: the bot scored better as Black at every rung (30 points at 3190) and the White side of these positions scored 35–48% whichever engine held it. Either the pool is lopsided or the bot plays the White side badly against a strong engine; bot vs bot separates the two | the White-side score over the 60 games. Book-opening arenas here gave White ~55–63%. Below 45% means the positions; 55% or more means the bot |
| 9 | **conversion suite, with and without a stalemate veto** | build `_STALEMATE_VETO = True` as a switch (code below). Suite: 20 won positions, the side to move ahead by 5 pawn units or more, in a `suite.tsv` (`name<TAB>fen`): the eight textbook endings KQ v K, KR v K, KRN v K, KBB v K, KBN v K, KQP v KP, KRP v K, KQ v KR, plus twelve middlegame or endgame positions taken from ARENA #12 PGNs where the bot was ahead by 5 or more (`sparring/ladder_box/lanes/L`). The bot with `max_sims=500` plays the side ahead; the defender is Stockfish 18 at full strength (`UCI_LimitStrength` off), one thread. Each position once with the reference and once with the veto candidate, 40 games | ARENA #12 lost six of 90 games to non-conversion: three stalemates with the bot far ahead (rook and knight against a bare king at ply 225) and three threefold draws while ahead. Stalemate is 1–3% of arena games, so a 50-game arena cannot see the veto; a suite can | conversions (checkmate before the 300-ply cap and the fifty-move rule) out of 20 for each build, and the per-move `q=` on the winning side, which tells whether the net even knows it is winning (ARENA #12's KRN v K read q ≈ +0.1 throughout) |
| 10 (only if item 9 converts under 15 of 20 with the veto) | **mate-search fallback** | when the opponent has a bare king or at most three pawn units and the bot is ahead by five or more, run a small iterative alpha-beta mate search (depth 1 to 9 plies, legal moves only, python-chess, capped at 0.5 s) before the MCTS and play a found mate; otherwise fall back to the MCTS pick | the veto keeps the game alive but the value head does not see the mate, so the search cannot steer; a mate solver in the tiny endings is the direct fix | rerun item 9's suite; conversions out of 20; the solver's time per move |
| 11 (last) | **R0 and the reference hero on this ladder** | `weights/R0_e8_ema/model.onnx` (and `baselines/reference-hero` if present) in the reference's `agent.py`, `max_sims=500`, vs `UCI_Elo` 2800 and 3000, 30 games each | ARENA #1's R0 = 2868 and the reference project's 2,474 were measured on a Mac with pondering on and from the standard start; nothing links them to today's conditions | the two per-rung scores beside item 6's, the same fit; the R0-to-submission gap under identical conditions |

**The stalemate veto, exactly.** In `agent.py`, next to the other switches: `_STALEMATE_VETO = False`
(reference) or `True` (candidate). In `_pick`, immediately after `order` is computed and before
any other use of it:

```python
if _STALEMATE_VETO and _material_for_mover(board) > 0:
    kept = []
    for idx in order:
        after = board.copy(stack=False)
        after.push(result.moves[idx])
        if not after.is_stalemate():
            kept.append(idx)
    if kept:
        order = kept
```

`_material_for_mover` already exists in `agent.py` (it feeds the `near_adjudication` rule).
The check is one move push and one legal-move generation per candidate, microseconds. It
does nothing when material is level or the bot is behind, and it never removes the last
legal move.

**Bundle additions for this section.** `sparring/stockfish-agent/agent.py` (the wrapper: reads
`SF_BIN` and `SF_ELO` from the environment, or bake them into the two defaults; `Threads=1`,
`Hash=64`, one stable `game` token per process, do not break that, see ARENA #1 §2),
`sparring/openings_ladder.tsv`, `sparring/ladder_box/ladder_fit.py` (per-rung table, joint
fit, sims per move; lane directories must be named `<rung>_<x>` with opponent directories
named `sf<rung>`), `sparring/ladder_box/ladder_lane.py` (the Linux lane runner used on the
box, pins each side to a core with `taskset`; on another OS write the equivalent loop with
`harness.play --fen`), `docs/ARENA12_STOCKFISH_LADDER_SUBMISSION.md`, and for item 11
`weights/R0_e8_ema/`. Stockfish 18 is the official release build for this machine's CPU;
check `id name Stockfish 18` and `option name UCI_Elo ... min 1320 max 3190` on the `uci`
reply before anything else, and note the build in every report. Stockfish is GPL and must
never enter a candidate directory or the zip.

**Reporting for items 6 to 11.** One `docs/ARENA<n>_<NAME>.md` each, same format, plus this
machine's Stockfish speed (`nps` from a 1 s `go movetime 1000` at `Threads=1` from the
start position) beside the bot's forward time. Items 6 and 7 replace ARENA #12's headline
figure in the decisions file if they disagree with it; say so explicitly.
