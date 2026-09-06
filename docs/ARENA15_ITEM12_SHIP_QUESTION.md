# ARENA #15 — item 12: the repetition and opening fixes on the live build, kept at 54.0%

Continues `docs/ARENA14_STOCKFISH_LADDER_SUBMISSION.md`. Written to be read cold.
Branch `justin-arenas` off `main` `535c045` (code identical to `691a904`; the two
commits between them are docs only).

Date: 2026-09-05, 14:39 to 16:20 UTC, unattended, on Justin's PC. Harness:
`harness.play` at the competition clock, 120 s + 0.5 s, 300-ply cap. Openings: the
fifteen curated positions in `sparring/openings_ladder.tsv`, both colours, two lanes
of 25 (`sparring/justin/lane.py`, the Python equivalent of `arena.sh` required by §5
because this is Windows). Net R1_e8_ema on both sides, sha256 verified
`ea52cc6aac7704abec84b8628a86cbd9df8999ea178bcf75c199a8f245c9ac5a`.

---

## 1. What was tested

Correction B's definition: the reference is the ladder build **plus** the two fixes,
and the opponent is what is actually on the ladder.

- **with** = `main` `691a904`: R1, pruning 1.33, policy temperature 1.359, root FPU 1.0,
  **and** the repetition and opening-adoption fixes.
- **without** = `bbc55a9` (md5 `93ce8c23`), the uploaded zip: the same knobs, no fixes.

Both sides carry exactly one further line, Correction C (below). Each is a verified
one-line diff from its git base; the diffs are in `sparring/justin/item12/{ref,sub}/CONFIG.txt`.

The fixes are the opening-adoption walk (`_find_in_tree`, `_ADOPT_NODE_LIMIT = 4096`)
and the repetition handling. They scored 58% against pruning-only in ARENA #10, but with
pondering on and against a build without the two knobs, so they had never been measured
on top of what actually plays.

## 2. Correction C: pondering off in the agents, not the harness

`harness/sandbox.py` suspends the idle agent with SIGSTOP. Windows has no SIGSTOP, so
`SUSPEND_IDLE` is false there and both sides would ponder — the non-comparable kind of
result. The harness was not edited. Instead `_PONDER_NODE_BUDGET = 100_000 -> 0` on
**both** sides.

Verified rather than assumed. Reading the agent process directly:

```
init: {'d_model': 96, ..., 'forward_ms': 3.927, 'sha256': 'ea52cc6aac...'}
init: opening pre-search sims=1216
move 1: e2e4 sims=64 reused=1216 pondered=0 q=+0.05 t=0.25s pruned
```

`pondered=0`, and `reused=1216` confirms subtree reuse still works, which is the
platform's behaviour. In `chessml/search.py` the ponder thread breaks at the top of its
loop when `expanded >= node_budget`, so at 0 it does no work at all.

## 3. Result

| with | without | W-D-L (with) | score | 95% | Elo | failures |
|---|---|---|---|---|---|---|
| `691a904` + fixes | `bbc55a9` uploaded zip | +7 =40 −3 | **54.0%** | 47.9–60.1% | +28 (−15 to +71) | **0** |

As White +3 =20 −2 (52.0%), as Black +4 =20 −1 (56.0%). Both lanes 54.0%.

Terminations: threefold repetition 34, checkmate 10, insufficient material 6.
Mean game 235 s, longest 320 s, shortest 105 s. Mean length 109.8 plies, median 104.

By opening: ladder03/04/12/13/14 75%, ladder01/02/05/06/08/09/10/15 50%,
ladder07/11 25%. All fifteen positions were played.

## 4. Machine

Intel i7-9700K, **8 physical cores, no hyperthreading**, Windows. Net forwards in
**3.93 ms**, against the platform's 4.5–6.2 ms across ten machines
(`rated-games/*.log`). **This PC is roughly 1.2–1.6x the platform's speed**, so both
sides searched deeper here than they will on the ladder. Two lanes, four cores in use,
nothing else running.

## 5. What this arena could not measure

**Per-move telemetry is unavailable on Windows.** `Sandbox.stop()` calls
`process.kill()` before joining the reader threads, and Windows `TerminateProcess`
discards pipe buffers that POSIX preserves, so `stderr_tail` comes back empty — checked
on a 6-ply game, the log is 80 bytes. The agent emits the lines correctly; they are lost
between the child and the log.

So this report has no `sims=`/`q=`/seconds-per-move brackets, and **item 12's specific
ask — every threefold where the losing side's `q=` was above +0.3 at the repetition —
could not be answered.** That needs a POSIX box. It is the one part of the item left
undone, and it is the part that would say *whether the fixes actually addressed round
19's failure mode* rather than merely that the build is not worse.

For the baseline it would have been compared against, from the platform's own logs:

| round | termination | max q, last 10 moves | median sims/move |
|---|---|---|---|
| round-19-gijs-smit | **threefold** | **+0.570** | 384 |
| round-14-pgn | stalemate | −0.020 | 184 |
| the other nine | checkmate | — | 480–755 |

## 6. Caveats

**Only 38 of 50 games are distinct** (§7's threshold is 40). Twelve are replays of
another game in the set, so the effective sample is smaller than 50 and the 47.9–60.1%
interval is optimistic. Fifteen openings with both colours over 50 games gives 3.3 games
per position, and equal engines from a fixed position repeat.

**82% of games were drawn** (40 of 50), 34 of them by threefold. Two builds this close
drawing four games in five is unsurprising, but it flattens the signal: a 50-game arena
resolving ±70 Elo resolves nothing at all when the draw rate is this high.

## 7. Decision

**KEPT.** 54.0% ≥ 50% with zero failures on either side, so the rule is satisfied and the
fixes go into the next zip on top of the live knobs.

The honest reading is weaker than the number: +28 Elo with an interval spanning −15 to
+71 is *no evidence of harm*, not evidence of gain, and the draw rate and the 38-of-50
distinctness both argue for treating it that way. What would change the reading is a run
on a POSIX machine that recovers the per-move `q=` at each threefold — that would say
whether the fixes prevent round 19's specific loss, which is the question worth answering.
