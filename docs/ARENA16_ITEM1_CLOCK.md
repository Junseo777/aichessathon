# ARENA #16 — item 1: the clock formula, dropped at 47.0% on a clean machine

Continues `docs/ARENA15_ITEM12_SHIP_QUESTION.md`, and reruns
`docs/ARENA13_STEP2_CLOCK.md`, whose 44.0% was voided as contended. Written to be
read cold. Branch `justin-arenas`.

Date: 2026-09-05 20:57 to 2026-09-06 00:26 UTC, unattended, on Justin's PC. Harness:
`harness.play` at the competition clock, 120 s + 0.5 s, 300-ply cap. Openings: the
fifteen curated positions in `sparring/openings_ladder.tsv`, both colours, two lanes
of 25 (`sparring/justin/lane.py`). Net R1_e8_ema on both sides, sha256 verified
`ea52cc6aac77…`.

---

## 1. What was tested

Reference is `main` `691a904` — R1 with pruning 1.33, policy temperature 1.359, root
FPU 1.0, and the repetition and opening-adoption fixes. The candidate is that with
three constants changed, a verified three-line diff:

```
_BUDGET_HORIZON        46  -> 60
_BUDGET_DIVISOR_FLOOR  14  -> 20
_BUDGET_FLOOR_S       0.0  -> 1.0
```

Both sides additionally carry `_PONDER_NODE_BUDGET = 0` (Correction C), since Windows
has no SIGSTOP and the harness cannot suspend the idle agent. Verified in the arena,
not merely in a probe: **`pondered=0` across all 6,649 moves**.

This is the rerun the brief calls the one that matters most. ARENA #13 measured 44.0%
on a laptop making 19 simulations a move — a tenth of platform speed — and was voided.

## 2. Result

| with | without | W-D-L (with) | score | 95% | Elo | failures |
|---|---|---|---|---|---|---|
| clock 60 / 20 / 1.0 | reference 46 / 14 / 0.0 | +10 =27 −13 | **47.0%** | 37.6–56.4% | −21 (−88 to +44) | **0** |

As White +3 =14 −8 (**40.0%**), as Black +7 =13 −5 (**54.0%**).

Terminations: checkmate 23, threefold repetition 18, insufficient material 8,
stalemate 1. Mean game 255 s, median 259 s. Mean length 133.0 plies, median 131.

By opening: ladder04 and ladder08 75%; ladder01 and ladder10 62%; ladder02, 05, 07,
13, 14 50%; ladder03 and ladder09 38%; ladder11, 12, 15 25%; **ladder06 0%**.

## 3. The mechanism worked; it just did not win

Median simulations and seconds per move, by bracket. Medians, not means — the net
carries a 60,000-entry transposition cache, so repetition positions are nearly free
and a handful of long draws drag the mean from 416 to 872 sims a move.

| moves | 0–19 | 20–29 | 30–39 | 40–49 | 50–59 | 60+ |
|---|---|---|---|---|---|---|
| **with** sims | 512 | 480 | 576 | 608 | 608 | 512 |
| **without** sims | 608 | 576 | 576 | 688 | 672 | 448 |
| **with** secs | 1.80 | 1.86 | 1.98 | 1.83 | 1.71 | **1.30** |
| **without** secs | 2.20 | 2.13 | 1.94 | 2.14 | 1.81 | **1.09** |

The candidate spends **less** per move everywhere up to move 59 — 1.80 s against 2.20 s
in the opening — and **more** past move 60, 1.30 s against 1.09 s. That is exactly what
the change was designed to do: a larger divisor conserves early so the endgame is not
played at two seconds. The curve moved as intended and the result still went backwards.

So the reading is not "the mechanism failed to engage". It engaged, and spending less
time in the first sixty moves to buy time after move 60 was a bad trade at this
strength. The reference's extra ~100 simulations a move through the middlegame appear
to be worth more than the candidate's extra 0.2 s in the endgame.

## 4. Against ARENA #13

| | ARENA #13 | ARENA #16 |
|---|---|---|
| score | 44.0% | **47.0%** |
| machine | contended laptop, ~19 sims/move | clean, 480–688 sims/move |
| pondering | mixed | off both sides, verified |
| openings | 8 book positions | 15 curated |
| verdict | void | **DROPPED** |

47.0% sits inside ARENA #13's interval and points the same way. The earlier number was
not measuring the clock change — it was measuring a starved machine — but it happened
to reach the right conclusion. **The change is not better, and now we know that rather
than suspect it.**

## 5. Machine

Intel i7-9700K, 8 physical cores, no hyperthreading, Windows. Net forwards **3.93 ms**,
against the platform's 4.5–6.2 ms, so this PC is roughly **1.2–1.6× platform speed**;
both sides searched deeper here than they will on the ladder.

Two lanes, not three. Measured with `sparring/justin/probe.py`: idle 1088–1152
pre-search sims; two lanes 992–1056; three lanes 800–992 with forward spiking to
6.8 ms. At a fixed clock contention does not lengthen games, it quietly buys less
search — which is what voided ARENA #13 — so a probe is the only way to see it. For an
arena *about* the clock, three lanes was not worth 30 minutes.

## 6. Caveats

**38 of 50 games distinct**, under §7's threshold of 40. Twelve are replays, so the
effective sample is below 50 and the 37.6–56.4% interval is optimistic.

**The colour split is wide**: 40.0% as White against 54.0% as Black, a 14-point gap on
25 games a side. Item 8 exists to ask whether that is the curated positions or the bot,
and this is a second instance of the pattern ARENA #14 saw on the ladder.

**A hang cost 2h40m** on the first attempt: both lane drivers wedged inside
`subprocess.run` on a `harness.play` child that no longer existed, a Windows
handle-inheritance stall, with 0-byte game logs. The box gate log ruled out machine
sleep — 215 consecutive polls, no gap over 3 minutes. `lane.py` now passes
`timeout=1200` and records `void by timeout`, so a stall costs one game rather than a
night. The run reported here is a clean restart, not the wedged attempt.

## 7. Decision

**DROPPED.** 47.0% is below 50%, with zero failures on either side, so the rule drops
it and the shipped clock formula stands.

What would change the reading: the mechanism does what it claims, so a smaller step —
the horizon at 52 rather than 60, or the floor without the divisor change — might keep
the endgame improvement without surrendering the middlegame. Nothing here says the
*direction* is wrong, only that this size of step costs more than it returns.
