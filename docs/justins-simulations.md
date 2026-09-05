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
| 9 | **conversion suite, with and without a stalemate veto** | the switch exists on `main` since `a037773` (`_STALEMATE_VETO`, off by default); on a checkout that lacks it, insert the code below. Candidate = reference with `_STALEMATE_VETO = True`. Suite: 20 won positions, the side to move ahead by 5 pawn units or more, in a `suite.tsv` (`name<TAB>fen`): the eight textbook endings KQ v K, KR v K, KRN v K, KBB v K, KBN v K, KQP v KP, KRP v K, KQ v KR, plus twelve middlegame or endgame positions taken from ARENA #12 PGNs where the bot was ahead by 5 or more (`sparring/ladder_box/lanes/L`). The bot with `max_sims=500` plays the side ahead; the defender is Stockfish 18 at full strength (`UCI_LimitStrength` off), one thread. Each position once with the reference and once with the veto candidate, 40 games. Then, whatever the suite says, 30 full games of the veto candidate against Stockfish at `UCI_Elo` 3000 on the curated positions, the ARENA #12 rung 3000 setup, to see the veto in real games | ARENA #12 lost six of 90 games to non-conversion: three stalemates with the bot far ahead (rook and knight against a bare king at ply 225) and three threefold draws while ahead. Stalemate is 1–3% of arena games, so a 50-game arena cannot see the veto; a suite can | conversions (checkmate before the 300-ply cap and the fifty-move rule) out of 20 for each build, and the per-move `q=` on the winning side, which tells whether the net even knows it is winning (ARENA #12's KRN v K read q ≈ +0.1 throughout); for the 30 games, score and the count of stalemates and threefold draws with the bot ahead by 3 or more, against ARENA #12's 1 stalemate and 1 such threefold at rung 3000 |
| 10 (only if item 9 converts under 15 of 20 with the veto) | **mate-search fallback** | when the opponent has a bare king or at most three pawn units and the bot is ahead by five or more, run a small iterative alpha-beta mate search (depth 1 to 9 plies, legal moves only, python-chess, capped at 0.5 s) before the MCTS and play a found mate; otherwise fall back to the MCTS pick | the veto keeps the game alive but the value head does not see the mate, so the search cannot steer; a mate solver in the tiny endings is the direct fix | rerun item 9's suite; conversions out of 20; the solver's time per move |

**The stalemate veto, exactly** (this is what `main` carries from `a037773`; needed only on an older
checkout). In `agent.py`, next to the other switches: `_STALEMATE_VETO = False` (reference) or `True`
(candidate). In `_pick`, immediately after `order` is computed and before any other use of it:

```python
def _stalemates(board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_stalemate()

# in _pick, right after `order = [int(i) for i in np.argsort(-result.visits)]`:
    if _STALEMATE_VETO and _material_for_mover(board) > 0:
        kept = [idx for idx in order if not _stalemates(board, result.moves[idx])]
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
`harness.play --fen`), `docs/ARENA12_STOCKFISH_LADDER_SUBMISSION.md`. Stockfish 18 is the official release build for this machine's CPU;
check `id name Stockfish 18` and `option name UCI_Elo ... min 1320 max 3190` on the `uci`
reply before anything else, and note the build in every report. Stockfish is GPL and must
never enter a candidate directory or the zip.

**Reporting for items 6 to 10.** One `docs/ARENA<n>_<NAME>.md` each, same format, plus this
machine's Stockfish speed (`nps` from a 1 s `go movetime 1000` at `Threads=1` from the
start position) beside the bot's forward time. Items 6 and 7 replace ARENA #12's headline
figure in the decisions file if they disagree with it; say so explicitly.

## 9. Added 2026-09-05 12:45 from the platform match logs: items 12 to 17

Two corrections before anything in this file is run, then six arenas. Nothing below starts on
the laptop; it is written for this PC.

**Correction A: the harness must suspend the idle agent.** On 2026-09-05 the platform began
suspending the agent process while the opponent moves; the match logs show pondered
simulations per move falling from 199–473 (rounds 10–15) to 7–23 (round 16 on). `main`
mirrors this since `59cd491` (`harness/sandbox.py` sends SIGSTOP between moves). Branch
`ship-chain` predates it, so every arena on it would let both sides ponder, which the
platform no longer allows. Cherry-pick `59cd491` onto the branch (or rebase `ship-chain`
onto `main`) and confirm that a game's `pondered=` reads under 30 before the first arena.
Results with pondering (ARENA #10, the E and D arenas below) are not comparable with results
without it.

**Correction B: the reference in section 1 is not what is on the ladder.** The uploaded zip
(`main` `bbc55a9`, md5 `93ce8c23`) is R1 with pruning 1.33, **policy temperature 1.359 and
root FPU 1.0**, and without the repetition and opening-adoption fixes. `ship-chain`'s
reference has the fixes and neither knob. Item 12 is the arena that resolves this; until it
is run, "kept" against `ship-chain`'s reference says nothing about the ladder build.

**Results that already exist, so they are not repeated** (50 games each unless stated; the
pondering column says which harness):

| arena | sides | result | pondering |
|---|---|---|---|
| A | fixes + pruning vs pre-pruning zip | 64.0%, +100 Elo | yes |
| B (ARENA #10) | fixes + pruning vs pruning only | 58.0%, +56 Elo | yes |
| D | proofs on top of B's winner | 52.0%, +14 Elo | yes |
| E | search extension 1.0x on top | 57.0%, +49 Elo (+17 =23 −10) | yes |
| ARENA #11 | clock 60/20/1.0 | 44.0%, contended laptop, void | mixed |
| F | LCB pick, z = 0.6 on q − z/√visits | pending, `sparring/step1/F_lcb_vs_step1/summary.txt` | no |
| box H (ARENA #9) | R6a d128 with R1's target vs R1, 100 games | 40.5%, −67 Elo | yes |
| box I | R7a d128 value blend vs R1, 100 games | 51.0%, +7 Elo, at 72% of R1's simulations | yes |

| priority | candidate | switch or build | why it needs measuring | what to report |
|---|---|---|---|---|
| 12 (before items 2–5) | **the ship question on the live build** | `ship-chain` reference with `load_fastest(..., policy_temperature=1.359)` and `MCTS(..., root_fpu=1.0)` added, vs the uploaded zip unpacked as the opponent directory; suspend harness; `sparring/openings_ladder.tsv`; 50 games | The fixes scored 58% against pruning-only with pondering on. The ladder then lost a half point to exactly the bug they fix: round 19, our search at +0.46 to +0.57 from move 39, five queen checks, threefold declared with 53 s on our clock. The live build has two knobs the reference lacks, so the fixes have never been measured on top of what actually plays, nor without pondering | the usual table, plus, from the game logs, every threefold where the losing side's `q=` was above +0.3 at the repetition; a kept result means the fixes go into the next zip on top of the live knobs |
| 13 | **pruning guard** | one line in `chessml/search.py`, `MCTS._cannot_be_overtaken`, before the rate estimate: `if sims < 0.25 * root.total: return False` (fresh simulations must be at least a quarter of the tree's visits before the stop rule may fire); vs the reference of item 12 | The rule compares the visit lead, which includes the reused subtree's inherited visits, with a rate measured on new simulations only. In round 17 it stopped move 28 after 32 new simulations on a reused tree of 888 and move 37 after 32 on 1,266, both with over a minute in hand; 4 of 48 searches ended under 200 simulations with 60 s or more left. With pondering gone the reused tree is our own previous search, so this fires on the ladder every game | the table, plus per side from the logs: searches under 200 simulations with more than 60 s left, median simulations per move, mean seconds per move by bracket. The guard should cost time on easy moves only; if seconds per move rise by more than 20% in the 0–19 bracket, say so |
| 14 (after 13) | **cap 8 s while the clock is healthy** | in `_budget_s`: `cap = 8.0 if left > 60.0 else 4.0` and `min(cap, budget, left - 1.0)`; keep everything else, including pruning; vs item 12's reference. Different from item 1: that reshapes the curve, this only lets a hard move run long while the clock is rich | The pruning build ends games with 30 to 58 s unused (rounds 15–20: 36, 10, 31, 58, 53, 16 s) and the 4 s cap bound five times in round 17. Against this: the three round-18 slips replayed from 150 to 2,500 simulations never change move, so the return is the round-14 kind of error (flips at 250) rather than the round-18 kind. Cheap, so measure it | the table, seconds per move by bracket, clock left at the end for each side, and how many moves hit the new cap |
| 15 (instead of, or before, item 3) | **extension at 1.0x** | `_EXTEND_FACTOR = 1.0`; vs item 12's reference | Item 3 proposes 2.0x. The only measured value is 1.0x: 57.0% with pondering, firing on 26% of moves at 5.0 s mean against a 3.4 s budget. Without pondering the tree at move start is smaller, so disagreement may be more frequent and the time cost larger; 1.0x is the safer first point | the table, the share of moves that extended and their mean seconds, and clock left at the end |
| 16 | **R7a, the d128 value-blend net, vs R1** | `weights/R7a_e8_ema/model.onnx` (sha256 `297ec269e63e…`, now in the run store with `weights/CHECKSUMS.txt`) in a copy of item 12's reference, vs the same reference on R1; (a) 100 games at the real clock on the suspend harness, (b) 100 games with `max_sims=500` on both sides | It tied R1 on the box (51.0%, 100 games) while getting 72% of R1's simulations per move, the first net that has; R6a with R1's plain target lost 40.5%. Its forward is 1.6x R1's on Mac-like cores (DECISIONS §2: d128 9.55 ms vs d96 5.97 ms), so (a) says whether the tie survives the platform's budget and (b) whether the net is better per simulation | both tables; the mean `sims=` per side in (a). R1 has twenty ladder games of evidence behind it, so R7a replaces it only on a clear win in (a), 55% or better, not a tie |
| 17 (only if 16 passes) | **the two knobs on R7a** | `policy_temperature=1.359` alone, then `root_fpu=1.0` alone, each vs R7a without it, 50 games | ARENA #4's disambiguation arena found both knobs net-specific: kept on R1, lost on R3. A new net inherits nothing | the two tables; switch a knob off for R7a if it loses |

**Reading the logs.** Every `harness.play` game log prints one line per move for each side:
`move N: <uci> sims=<n> reused=<n> pondered=<n> q=<v> t=<s>s[ pruned][ extended]`. The
regex `^  move (\d+): (\S+) sims=(\d+) reused=(\d+) pondered=(\d+) q=([-+\d.]+) t=([\d.]+)s( pruned)?( extended)?`
recovers the telemetry the items above ask for. The platform's own match logs for rounds
10 to 20 (`rated-games/*.log`, same format under OUTPUT plus a per-move clock table) are the
baseline: median 480 simulations per move on the pruning build, 33 of 34 to 66 of 70
searches stopped early, forward 4.5 to 6.2 ms across ten machines.

**Bundle additions for this section.** `weights/R7a_e8_ema/` and `weights/R7a_e8/` with
their `CHECKSUMS.txt` lines, `rated-games/*.log`, and commit `59cd491` from `main`.

## 10. Running on the training box instead of this PC

Any item in sections 1, 8 or 9 may run on the team's rented box rather than here, and
items 6 to 8 are better there: the box is the machine ARENA #12 was measured on, so
rung-for-rung comparisons need no speed caveat. Move a run there when this PC cannot give
each lane two free physical cores for the hours it needs, when an item would take more
than a day here, or when a result must sit beside ARENA #12. Access is by SSH key; Junseo
adds your key and gives you the host and port. Nothing about the box's address belongs in
this file or in a report.

**What the box is.** A RunPod container on an AMD Ryzen 9 7950X: 32 vCPUs visible, but the
cgroup grants **13.6 cores and 61 GB**. Read `/sys/fs/cgroup/cpu.max` and
`/sys/fs/cgroup/memory.max`; never trust `free`, `nproc` or `lscpu`, which report the host.
Physical cores are 0–15 with hyperthread siblings 16–31; pin one side of a game to one
physical core with `taskset -c N` and never share a core between two agents. A trainer
usually holds cores 12–15 and the GPU; leave them. `nr_throttled` in
`/sys/fs/cgroup/cpu.stat` is the check that the quota did not bite: read it before and
after a run and report both. The box has no `rsync` and no `unzip`: copy with `scp`, extract
with `python -m zipfile -e`. Its git checkout cannot fetch (the deploy key is rejected), so
code arrives by `scp` or a git bundle, never by `git pull` there.

**Sharing it.** Other sessions queue work there in `tmux` sessions and gate on a clear box.
Before launching anything: `tmux ls`, then `pgrep -af "harness/runner.py"` to see live games
and `taskset -cp <pid>` to see who holds which core, and the marker files
`/workspace/bracket/{PHASE,DONE,ABORT}_*` and `/workspace/ladder/{PHASE,DONE,ABORT}_ladder`.
Follow the convention exactly: your queue script waits until no `harness/runner.py` process
exists, launches at most six lanes on cores 0–11 with the lane drivers on 12–15, logs to
`<dir>/queue_<name>.log` with UTC timestamps, and writes `PHASE_<name>` at start and
`DONE_<name>` or `ABORT_<name>` at the end so the others can wait on you. Never kill or
detach another session's tmux, never launch while games run, and if the box stays busy for
more than four hours, run the item here instead and say so in the report.

**What is already there, under `/workspace/ladder/`.** Stockfish 18
(`stockfish/stockfish-ubuntu-x86-64-vnni512`, the release build for that CPU, tar sha256
`91d89e0e…`); rung agents `agents/sf2800`, `sf3000`, `sf3190` and `agents/sf_full`
(`UCI_LimitStrength` off, smoke-tested); `openings_ladder.tsv`; `ladder_lane.py`, the lane
runner used for ARENA #12 (`--candidate`, `--opponent`, `--cand-core`, `--opp-core`,
`--openings`, `--offset`, `--max-games`, `--hours`, `--out`, `--base-ms`, `--increment-ms`;
`--repo` defaults to the symlink `repo` → `/workspace/bracket/repo`, the harness pinned at
`f2a78b8`); `ladder_fit.py`; `build_agent.sh <zip>`, which unpacks a submission zip into
`agents/sub_np`, sets the ponder budget to 0, verifies the one-line diff and plays a 10 s
smoke game; `queue_ladder.sh`, the queue that ran ARENA #12, to copy for a new item; and
`agent_main_a037773.py`, the current `main` agent with the veto switch, for building the
item 9 candidate. The venv is `/workspace/aichessathon/.venv/bin/python` (python-chess
1.11.2, onnxruntime, numpy). The pinned harness there does not suspend an idle agent, so
every candidate built on the box carries `_PONDER_NODE_BUDGET = 0`, which `build_agent.sh`
does and which matches the platform. The net is `/workspace/weights/R1_e8_ema/model.onnx`
(check the sha256).

**Building a candidate there.** A candidate directory is `agent.py`, a `chessml` symlink
to `repo/chessml`, and `weights/` (`model.onnx` + `manifest.json`). Start from
`agents/sub_np` (the ARENA #12 candidate) and change only the switch lines; for the veto,
start from `agent_main_a037773.py` instead and set `_PONDER_NODE_BUDGET = 0` and
`_STALEMATE_VETO = True`. `diff` against the base must show only those lines; keep the diff
in a `BUILD.txt` beside the agent. Speed there: the net forwards in ~2.4 ms on an idle core,
about 2–2.7x the platform, so use the `max_sims=500` cap for every bot side (section 8).

**Results.** Keep each item under `/workspace/justin/<item>/` with the lanes, `results.csv`,
per-game logs and the summary, pull them to this PC with `scp -r` into `sparring/justin/`,
and write the ARENA report here as in section 6. The box is shared storage on a network
filesystem; do not leave anything there that is not a result.
