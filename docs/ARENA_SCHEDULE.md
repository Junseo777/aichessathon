# Arena schedule and handoff

Written to be read cold by an agent picking this up. All times UTC. Baseline:
**2026-09-05 17:00 UTC**. Estimates come from measured rates, not guesses; the
measurement is stated beside each so you can re-derive them if a machine differs.

## 1. Where things stand

| | |
|---|---|
| Branch | `main`, arena work committed directly (Justin's instruction; the brief's §6 says branch-and-bundle, overridden) |
| Done | **item 12** — 50 games, 54.0% (+7 =40 −3), failures 0, **KEPT**. `docs/ARENA13_ITEM12_SHIP_QUESTION.md` |
| Running | **item 1** (clock 60/20/1.0) on Justin's PC, 2 lanes, started 16:49, ends **~18:26** |
| Blocked | items 6–9 needed the box; as of 22:30 UTC 6, 7 and 16 are withdrawn and 8, 9 run on this PC (section 3, the box) |

## 2. The two machines

**Justin's PC** — Intel i7-9700K, 8 physical cores, no hyperthreading, Windows.
Net forwards **3.93 ms**. Stockfish 18 AVX2 at **0.70 Mnps** (Threads=1), installed
at `C:\Chess project\tools\` — outside the repo, it is GPL.

**Two lanes, not three.** Measured with `sparring/justin/probe.py`: idle 1088–1152
pre-search sims and ~533 sims/move; two lanes 992–1056; three lanes 800–992 with
forward spiking to 6.8 ms. At a fixed clock contention does not lengthen games, it
quietly buys less search — which is what voided ARENA #11 — so the probe is the only
way to see it. Rate: **50 games = 1.7 h** (234 s/game, two lanes).

**The box** — RunPod, Ryzen 9 7950X, cgroup grants 13.6 cores and 61 GB. Net forwards
2.3–2.8 ms. Six lanes on cores 0–11, drivers on 12–15. With `max_sims=500` games run
~150 s, so **50 games ≈ 25 min on six lanes**.

## 3. Timeline

### Justin's PC (sequential, two lanes)

| item | what | start | end |
|---|---|---|---|
| 1 | clock 60 / 20 / 1.0 | 16:49 | **18:26** |
| 13 | pruning guard (`sims < 0.25 * root.total`) | 18:26 | 20:10 |
| 14 | cap 8 s while clock > 60 s | 20:10 | 21:55 |
| 15 | extension 1.0x | 21:55 | 23:40 |
| 2 | LCB pick, z = 1.5 **and** z = 5.0 (100 games) | 23:40 | 03:05 |
| 3 | extension 2.0x | 03:05 | 04:50 |
| 5 | mirror calibration | 04:50 | 06:35 |
| 4 | *conditional*, only if 2 and 3 both pass | 06:35 | 08:20 |

### The box — amended 2026-09-05 22:30 UTC: items 6, 7 and 16 withdrawn, 8 and 9 move to the PC

The plan below (six lanes from ~23:00) is superseded. A new net, **R8** (R1's recipe with
the policy head trained on Stockfish's MultiPV top-four lines), beat R1 **61.5%** in 100
games at the platform setting (500 sims a side, no pondering; `lanes/U0/summary.txt` on the
box). Consequences, per `docs/justins-simulations.md` section 11:

| item | was | now |
|---|---|---|
| 6 | ladder at `max_sims=500` on R1 | **deferred** to the final build, which will not be R1 |
| 7 | fixed-node rungs on R1 | **deferred**, same reason |
| 8 | colour check, 60 games | keep, **on this PC** after the PC track |
| 9 | conversion suite ± veto | keep, **on this PC** after the PC track |
| 16 | R7a vs R1 | **withdrawn** — R7a tied R1, R8 beats R1; 17 goes with it |

The box is booked back to back after S by Junseo's queues **P2** (R8 vs R1 repeat, R8's
knobs, R8b), **P3** (R8 at 16 epochs vs R8), **gpu2/gpu5** (R8c, then R8g = the MultiPV
policy on the pure engine value target) and **g8** (R8g vs R8). It looks clear for minutes
between their phases, for example ~03:15–04:00 between P3 and g8; a launch in such a gap
holds the cores for hours and delays the R8g arena. The gate below therefore also requires
`DONE_g8`. Nothing of Justin's runs on the box before that, and after it only what the
table above still wants there, which is nothing.

**Everything non-conditional finishes ~06:35 Sunday 6 Sept**, PC track being the long
pole. With conditionals 4, 10 and 17, ~09:30.

Both estimates assume unattended running and no failures. Budget slippage: one failed
arena costs its own length again, so a bad night is ~09:00 rather than 06:35.

## 4. Gate before touching the box

Poll every 60 s, need three clear minutes in a row (`sparring/justin/wait_for_box.sh`
does this):

- `/workspace/bracket/DONE_p` exists, or `ABORT_p`
- no `queue_s.sh` process, or `DONE_s` / `ABORT_s` exists
- `/workspace/bracket/DONE_g8` exists, or `ABORT_g8`, or it is past 2026-09-06 12:00 UTC
  (added 22:30 UTC; the script checks it as `g8_clear`)
- no `harness/runner.py`, `bracket.py --lane`, `ladder_lane.py --lane`

Then: at most six lanes, one agent per physical core on 0–11, drivers on 12–15,
everything under `/workspace/justin/<item>/`, `queue_<item>.log` with UTC timestamps,
`PHASE_justin_<item>` at start and `DONE_justin_<item>` / `ABORT_justin_<item>` at the
end. Read `nr_throttled` from `/sys/fs/cgroup/cpu.stat` before and after and report
both; it is cumulative since boot, so the absolute value means nothing. Never touch
another session's tmux. Cap every bot side `max_sims=500`. Do **not** use
`/workspace/weights/R1_e8_ema/model.fused.onnx` — the main-branch agent does not load it.

## 5. Four things that will bite you

**Windows cannot suspend the idle agent.** `harness/sandbox.py` uses SIGSTOP, which
Windows lacks, so `SUSPEND_IDLE` is false. Do not edit `harness/`. Instead set
`_PONDER_NODE_BUDGET = 0` on **both** sides (Correction C) and confirm `pondered=0`
in the logs. Every candidate here carries it, recorded in its `CONFIG.txt`.

**Per-move telemetry is lost from arena logs on Windows.** `Sandbox.stop()` kills the
child before joining the reader threads and `TerminateProcess` discards the pipe
buffers, so `stderr_tail` is empty. `sparring/justin/telemetry/sitecustomize.py`
recovers it: on `PYTHONPATH`, Python imports it at interpreter startup in every agent
subprocess and tees stdout and stderr to `tel/<game>/agent_<side>_<pid>.log`.
`lane.py` sets it per game. Verified: `pondered=0` across 331 moves.

**Report the median, not the mean.** Median is 416 sims/move; the mean is 872 and the
max 54,080. The net carries a 60,000-entry transposition cache, so in a repetitive
endgame nearly every evaluation is a cache hit and simulations become almost free — a
handful of long draws dominate any average. A mean-based bracket table implies the
machine is four times faster than it is.

**Only 38 of 50 games were distinct in item 12** (§7's threshold is 40). Fifteen
openings with both colours over 50 games is 3.3 games per position, and equal engines
from a fixed position repeat. Count distinct games in every report; the interval is
optimistic when they are fewer.

## 6. Keep rule

A candidate is kept at **50% or better over its 50 games with no failure on its own
side** (crash, illegal move, flag, init failure). 50 games resolve about ±70 Elo, so
kept means *no evidence of harm*, not evidence of gain. Report the point estimate and
the interval either way, and append one line to `sparring/justin/decisions.txt`:

```
<name>: <n> games, with <score>% (+w =d -l), failures <k> -> KEPT|DROPPED
```
