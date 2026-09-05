# ARENA #8 — is R1's edge over R2 the value head's scale?

R1 (outcome-labelled) beats R2 (depth-8 engine-labelled) by about 100 Elo (ARENA3,
ARENA5). The calibration probe (ARENA5 §3) found R1's value head 30% hotter than R2's
(slope 1.3, correlation 0.93) while the search constants, `c_puct` 1.5 and the
first-play reduction 0.25, are fixed numbers on the value scale. This arena asks
whether R2 loses because its head speaks too quietly for those constants, or because
its labels teach something different. Written to be read cold.

Date: 2026-09-04, 14:57 UTC onwards. Box: RunPod, Ryzen 9 7950X, one physical core per
agent via `taskset`, cgroup quota 13.6 cores. Harness: `harness.referee.play_match` at
the competition clock, 120 s + 0.5 s, 300-ply cap, 60 s init, `harness/rules.py`
defaults, unmodified. Code: `aichessathon` at **f2a78b8** on every side. Nets, EMA
epoch 8, read back from each agent's `init:` line in every game: R1 `ea52cc6a…`,
R2 `5c433379…`. Draw rule off on both sides (thresholds ±1.00), the ARENA5 A1 treatment,
so the rescale reaches only the search.

---

## 1. Headline

**Rescaling R2's value head by 1.3 does not close the gap.** Three arms of 100 games,
the same night, the same code, the same budget: R1 scores 62.0% against R2 scaled
linearly, 60.0% against R2 scaled in tanh space, and 59.5% against the plain R2 played
as a control. The three are indistinguishable.

| opponent | games | R1 W–D–L | R1 score | 95% CI | implied Elo | phase |
|---|---|---|---|---|---|---|
| R2, v' = clip(1.3 v) | 100 | +34 =56 −10 | **62.0%** | 52.2–70.9% | +85 (+15 to +155) | F |
| R2, v' = tanh(1.3 atanh v) | 100 | +33 =54 −13 | **60.0%** | 50.2–69.1% | +70 (+1 to +139) | G |
| R2 plain, same-night control | 100 | +34 =51 −15 | **59.5%** | 49.7–68.6% | +67 (−2 to +136) | G |
| R2 plain, ARENA5 A1, ~700 sims/move | 100 | +40 =51 −9 | 65.5% | 55.8–74.1% | +111 (+40 to +182) | earlier |

If the scale were the mechanism, R1 should have fallen toward 50–55% in the rescaled
arms and stayed higher in the control. Both rescales landed on the control. What R1's
labels teach is not recoverable by turning R2's head up.

## 2. Method

`sparring/bracket_size/patch_value_scale.py` builds a real copy of the pinned agent
whose `chessml/net.py` rescales every value the network returns before the search or
the cache sees it; priors, move lists and weights are untouched. Two forms:

- **lin**, phase F: v' = clip(1.3 v, −1, 1). The literal rescale by the probe's slope.
  One of the six verification positions, a won rook ending, clips from +0.82 to +1.00.
- **tanh**, phase G: v' = tanh(1.3 · atanh v). Slope 1.3 at zero, bounded, ordering
  preserved; what a head trained on a 1.3× hotter target would look like. +0.82 → +0.90.

The builder evaluates six positions through the patched and the unpatched `chessml` in
two subprocesses and refuses the agent on any mismatch above 1e-6; both agents passed
(small values scale by exactly 1.3, priors identical). One 10 s smoke game per agent
against R1 preceded the phase; the init line names the scale on exactly one side.

Phase F: two lanes of 50 games on cores 0–3 (offsets 0 and 4 into the eight-opening
book, opposite first colours, so 50 games per colour). Two lanes rather than six because
the other session's Lc0 feature chain held cores 8–15 all evening; 12 pinned agents plus
one training run on cores 4–7 stayed inside the 13.6-core quota (1,051 throttled
100 ms periods over the four hours, 0.7%).

Phase G, queued to start once the feature chain released its cores: the tanh arm,
three lanes, 100 games, and a **same-night control**, R1 vs plain R2, three lanes,
100 games, so the baseline is measured at tonight's simulation budget rather than
borrowed from ARENA5.

## 3. Phase F in detail

```
games 100   R1 +34 =56 -10   score 62.0%   (95% Wilson 52.2%-70.9%)
implied R1 - R2x1.3: +85 Elo (95% +15 to +155)
R1 as White: +19 =26 -5, 64.0%      R1 as Black: +15 =30 -5, 60.0%
lanes: 2 x 50, colours balanced
terminations: checkmate 44, threefold 54, insufficient material 2   (threefold 54%)
mean game 288 s, 115 plies, longest 360 s
sims/move: R1 1,611 (pondered 1,421), R2x1.3 1,615 (pondered 1,447); peak RSS 249 / 223 MB
distinct games: 54 of 100
failed games: none; init-line sha mismatches: 0
by opening (R1): English 75%, Winawer 75%, Ruy 75%, Slav 67%, KID 64%, Najdorf 62%,
                 Caro-Kann 42%, QGD 33%   (n = 12-14 each)
```

## 4. Reading it

1. **Scale is not the mechanism.** Both rescales landed on the same-night control to
   within 2.5 points, three independent 100-game samples. A partial effect of a few
   points cannot be excluded at this sample; the hypothesis that most of the 100 Elo was
   the search constants under-reading a quiet head is.
2. **So the labels carry different information.** With the policy heads near-identical
   (ARENA3 §3), the value head is the whole difference, and the difference is in what
   the target teaches, not how loud it is. The Lichess `[%eval]` rows R1 saw are not the
   explanation either: on the 3.53M rows carrying both a Lichess eval and a depth-8
   label they correlate at 0.974 and agree in sign 99.2% of the time, so R2 had nearly
   the same values on those rows. What remains is the 36.5M rows where R1 saw a game
   result and R2 a depth-8 evaluation.
3. **Conversion, again.** 54% of games ended by repetition with the rule off and a
   rescaled head, 50% with the tanh head and 46% in the control, the same band as every
   other d96 pairing. The rescale did not change how often R2 lets won positions drift.
4. **Replays.** 54, 52 and 65 of 100 games are distinct in the three arms: both nets
   are near-deterministic from a fixed book, so lanes over eight openings replay lines.
   Effective samples are nearer 55–65 and the intervals are optimistic. Same caveat as
   ARENA7; a wider book is the fix.
5. **Budget.** Both sides ran ~1,600 simulations per move all night, about 2.3× ARENA5
   A1's 700, because the box was fast. Every game's two sides shared conditions, and the
   control removes the budget from the comparison between arms; §6 discusses the
   control against ARENA5.

## 5. Decision this supports

**Do not look for R2's missing Elo in the search constants; it is in the training
target.** Sample: 300 games at the competition clock, three arms of 100, no failed
games, the control measured the same night. Downstream: the d128 net trained tonight
with R1's target (R6a, the `--value-source lichess` run, ARENA9) is the right next
candidate, not a rescaled R5; the 100M shard needs no engine labels unless the blend
(R7) earns them; a value-scale sweep on R1 itself remains untested and is not the
priority.

## 6. Phase G in detail

Started 19:17 UTC once the feature chain had released cores 8–15; six lanes on cores
0–11, drivers on 12–15; 13 throttled periods over the phase. The R6a trainer overlapped
cores 4–7 for the first 35 minutes (until 19:52), so lanes g_tanh_z and g_ctrl_x ran a
lower budget for their first games; both sides of each affected game shared it.

### Tanh arm — R1 vs R2 with v' = tanh(1.3 atanh v)

```
games 100   R1 +33 =54 -13   score 60.0%   (95% Wilson 50.2%-69.1%)
implied R1 - R2 tanh: +70 Elo (95% +1 to +139)
R1 as White: +13 =31 -6, 57.0%      R1 as Black: +20 =23 -7, 63.0%
lanes: 60.3% / 61.8% / 57.8%
terminations: checkmate 46, threefold 50, insufficient material 4   (threefold 50%)
mean game 287 s, 112 plies; sims/move R1 1,724 (pondered 1,413), R2 1,515 (1,301)
distinct games 52 of 100; failed games: none
by opening: Slav 96%, Najdorf 79%, Winawer 75%, Caro-Kann 71%, English 50%, KID 50%,
            Ruy 46%, QGD 8%
```

### Control — R1 vs plain R2, thresholds ±1.00 both sides (ARENA5 A1 replayed)

```
games 100   R1 +34 =51 -15   score 59.5%   (95% Wilson 49.7%-68.6%)
implied R1 - R2: +67 Elo (95% -2 to +136)
R1 as White: +20 =24 -6, 64.0%      R1 as Black: +14 =27 -9, 55.0%
lanes: 60.3% / 58.8% / 59.4%
terminations: checkmate 49, threefold 46, insufficient material 5   (threefold 46%)
mean game 297 s, 124 plies; sims/move R1 1,611 (pondered 1,437), R2 1,573 (1,367)
distinct games 65 of 100; failed games: none
by opening: English 88%, Slav 82%, Najdorf 71%, Caro-Kann 62%, KID 54%, Ruy 54%,
            Winawer 42%, QGD 21%
```

Pooled over the three arms, 300 games: R1 60.5%, +101 =161 −38.

### The control against ARENA5

The same pairing under the same treatment scored 65.5% for R1 at ~700 simulations per
move (ARENA5 A1, 06:10 UTC) and 59.5% at ~1,600 (this control, 19:17 UTC). Six points
on two 100-game samples is not significant (the difference's standard error is about
seven points), but the direction is the one the first-play-urgency analysis predicts:
more search lets the weaker head recover a little. Treat R1's edge over R2 as
60–65% across budgets, and record the budget with every future pairing.

## 7. Artifacts

`sparring/bracket_box/lanes/{F,G_tanh,G_ctrl}/` (pulled from the box's
`/workspace/bracket/lanes/`): each lane with `results.csv`, `meta.json` (commit, cores, net sha256s), one PGN and one
log per game with both agents' per-move telemetry; `summary_local.txt` from
`sparring/bracket_size/analyse.py`. Agents: `/workspace/bracket/agents/r2_v13lin_t1.00`
and `r2_v13tanh_t1.00`, built by `patch_value_scale.py`; queue
`sparring/bracket_size/queue_scale.sh`, log `queue_scale.log`; smoke games
`sparring/bracket_box/smoke/scale_*`.
