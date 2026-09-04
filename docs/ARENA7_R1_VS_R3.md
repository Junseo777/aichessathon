# ARENA #7 — R1 vs R3, the two ship contenders

R1 beat R2 three times over (ARENA3 on the Mac, ARENA5 on the box); R3 tied R2
(ARENA3). The two had never met. Written to be read cold.

Date: 2026-09-04, 13:08–14:19 UTC, the box's last idle window before the stop. Box:
RunPod, Ryzen 9 7950X, one physical core per agent via `taskset`, hyperthread siblings
idle. Harness: `harness.referee.play_match` at the competition clock, 120 s + 0.5 s,
300-ply cap, 60 s init, `harness/rules.py` defaults, unmodified. Code: `aichessathon`
at **f2a78b8** on both sides, draw rule off on both sides (thresholds ±1.00), the same
treatment as ARENA5 A1 and the whole of ARENA6. Nets, EMA epoch 8, read back from each
agent's `init:` line in every game: R1 `ea52cc6a…`, R3 `a1163b19…`. Six lanes, 14
games each, eight book openings with both colours, 42 games per colour.

---

## 1. Headline

**A tie: R1 50.6% over 84 games, +4 Elo (95% −70 to +78).**

```
games 84   R1 +20 =45 -19   score 50.6%   (95% Wilson 40.1%-61.0%)
R1 as White: +14 =26 -2, 64.3%      R1 as Black: +6 =19 -17, 36.9%
White scored 63.7% over all games
lanes: 42.9% / 50.0% / 53.6% / 60.7% / 50.0% / 46.4%
terminations: checkmate 39, threefold 40, insufficient material 4, fifty-move 1   (threefold 48%)
mean game 290 s, 115 plies; sims/move R1 1,529 (pondered 1,253), R3 1,509 (1,278)
peak RSS 236 / 212 MB; failed games: none
distinct games: 64 of 84
by opening: QGD 70%, Caro-Kann 60%, Slav 60%, Winawer 50%, Ruy 50%, KID 46%, Najdorf 42%, English 30%
```

## 2. Reading it

1. **R1's 100-Elo edge over R2 does not carry to R3.** R2 and R3 differ only in
   `value_weight` (1.0 vs 2.5) and were level head-to-head (ARENA3, 49.0% for R2 over
   98 games, −51 to +37), yet R3 holds R1 where R2 did not. The intervals allow a
   consistent story (R3 ≈ R2 + 30–50, R1 ≈ R2 + 100, so R1 vs R3 ≈ +50–70, whose
   57–60% is inside this run's 40–61%), but the point estimates say R3 is the better
   opponent for R1 than its record against R2 suggests. R3 has the best value MSE of
   the d96 nets (0.0138 against R2's 0.0152); whether that is the mechanism is not
   identified here.
2. **Colour decides these games.** White scored 63.7%, against 55–58% in every
   other pairing on this box tonight, and R1 lost only 2 of 42 games as White but 17
   of 42 as Black. Two nets this close, from a fixed book, play into the same lines:
   only 64 of 84 games are distinct, with one Slav line replayed four times. The
   effective sample is nearer 64 than 84, and the interval above is optimistic.
3. **Simulation budget was about double the earlier phases'.** Both sides averaged
   ~1,500 simulations per move here against ~700 in ARENA5 A1 and ~880 in ARENA6,
   because the box ran faster as the day went on (no cgroup throttling at any point;
   the host's load is not ours to control). Both sides of every game shared the same
   conditions, so the pairing is fair, but first-play urgency and the draw-seeking
   collapse are budget-dependent, so this result and ARENA5's were not measured at the
   same search depth.

## 3. Decision this supports

**R1 remains the ship candidate, on the strength of three independent 100-game wins
over R2; R3 is the only alternative, and on this evidence it is not worse than R1.**
Sample: 84 games (64 distinct), no failures. What would settle it: 200+ more R1 vs R3
games from a wider opening set than these eight (they replay), with colours balanced
and the budget recorded, before the 11 September lock. If R3 is chosen instead, the
Lc0 feature chain on the Mac (ARENA4) was built on R3 by its pre-agreed rule, so the
search work transfers directly; R1 would need the chain re-measured on it.

## 4. Artifacts

`sparring/bracket_box/lanes/E/`: six lanes with `results.csv`, `meta.json`, one PGN
and one log per game with both agents' per-move telemetry; `summary.txt`.
Queue: `sparring/bracket_size/queue_extra.sh`.
