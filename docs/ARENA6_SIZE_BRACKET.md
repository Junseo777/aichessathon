# ARENA #6 — the size bracket at the competition clock

R4 (d64), R5 (d128) and RA (d32) each head-to-head against R2 (d96), the fast track's
STOP #4 question: does capacity bought with simulations survive the real clock? Written
to be read cold.

Date: 2026-09-04, 09:01–13:08 UTC. Box: RunPod, Ryzen 9 7950X, one physical core per
agent via `taskset`, hyperthread siblings idle, cgroup CPU quota 13.6 cores. Harness:
`harness.referee.play_match` at the competition clock, 120 s + 0.5 s, 300-ply cap,
60 s init, `harness/rules.py` defaults, unmodified. Code: `aichessathon` at
**f2a78b8** on every side. Nets, EMA epoch 8, exported on the box the same morning
and read back from each agent's `init:` line in every game log: R2 `5c433379…`,
R4 `455a100d…`, R5 `f51a2614…`, RA `616a5fc0…`. Zero mismatches against lane metadata.

All four nets share the 40M shard, seed 0, 8 epochs, batch 1024, value_weight 1.0 and
100% depth-8 engine value labels; only `d_model`, `n_heads` and `n_blocks` differ.

---

## 1. Headline

**Nothing beats d96, and the small nets are not close.** With the draw rule off on
both sides (ARENA5 §5), against R2 at the real clock:

| candidate | params | val acc | games | W–D–L vs R2 | score | 95% CI | implied Elo vs R2 |
|---|---|---|---|---|---|---|---|
| **R5 d128** | 2.44M | 0.5622 | 100 | +22 =53 −25 | **48.5%** | 38.9–58.2% | **−10** (−78 to +57) |
| R4 d64 | 629k | 0.5375 | 100 | +17 =50 −33 | 42.0% | 32.8–51.8% | −56 (−125 to +12) |
| RA d32 | 118k | 0.4859 | 100 | +2 =17 −81 | 10.5% | 5.9–18.0% | −372 (−482 to −263) |

R2 d96, 1.38M params, val acc 0.5528, is the opponent in every row. Each pairing:
four lanes over two phases (B: 2 × 28 games, B2: 2 × 22), 50 games per colour, no
failed games in 300.

R5's val accuracy (0.5622 against R2's 0.5528) is real, and it costs little search;
R5 is inside the noise of R2. R4 gives up accuracy for almost no extra search and
loses. RA, the reference's own size, is beaten by 370 Elo: the ~5x simulation
multiplier the timing table promised never arrives.

## 2. Why the multiplier does not arrive

The per-move telemetry (`sims=` in every move line, both sides, all games) gives the
simulation rate each net actually achieved in the same games as its opponent:

| pairing | candidate sims/move | R2 sims/move in the same games | ratio | predicted from forward time |
|---|---|---|---|---|
| RA d32 vs R2 | 1,106 (pondered 933) | 880 (633) | **1.26x** | ~5x |
| R4 d64 vs R2 | 1,043 (877) | 887 (704) | **1.18x** | ~1.8x |
| R5 d128 vs R2 | 770 (569) | 893 (713) | **0.86x** | ~0.6x |

Means over every move of every game in the pairing (4,950–5,690 moves per side);
`pondered` is the mean number of simulations banked on the opponent's clock. For
comparison, R1 vs R2 with the same treatment (ARENA5, A1) ran at 715 vs 693.

The forward-cost table in DECISIONS §2 (M1, one thread) predicts d32 ≈ 5x, d64 ≈ 1.8x,
d128 ≈ 0.6x of d96's simulations. Measured in play: 1.26x, 1.18x, 0.86x. The reason is
that the forward pass is only part of a simulation. Measured on an idle pinned core of
this box while the lanes ran (`sparring/bracket_size/simrate.py`, fresh tree, empty
cache, 3 s from each book position, medians):

| net | forward ms | sims/s | forwards/s | cache hits | ms per sim | non-forward ms per expansion |
|---|---|---|---|---|---|---|
| R2 d96 | 7.76 | 105 | 92 | 9% | 9.5 | 3.1 |
| R4 d64 | 7.74 | 145 | 124 | 12% | 6.9 | 0.4 |
| R5 d128 | 10.75 | 79 | 68 | 9% | 12.7 | 3.9 |
| RA d32 | 0.58 | 263 | 217 | 19% | 3.8 | 4.0 |
| R1 d96 | 8.29 | 110 | 99 | 10% | 9.1 | 1.8 |

Every expansion pays for legal-move generation, `featurize`, move encoding, board
copies and the transposition key in Python before the network is asked anything: about
3–4 ms per expansion here. A 0.6 ms network therefore runs at ~4 ms per simulation,
not 0.6, and a 7.8 ms network at ~9.5 ms: the ratio is 2.5x in isolation and shrinks
further in real games, where reuse and pondering favour the tree that already exists.
(d64's forward time equals d96's on this box; on the M1 it was 3.37 vs 5.97 ms. This
box's absolute timings are unstable and were not relied on; the ratios are.)

Two consequences. The "parameters cost simulations" trade in DECISIONS §2 is far
flatter than the forward-time table implied, so **capacity is nearly free up to d128
at this clock**, and the lever that would actually buy simulations is the Python cost
per expansion, not a smaller network. And the reference's 116k model with ~1,900
simulations per move does not exist on this runtime; RA got ~1,000.

## 3. Results in detail

### R5 d128 vs R2 — 100 games

```
R5 +22 =53 -25   score 48.5%   (95% Wilson 38.9%-58.2%)   implied -10 Elo (-78 to +57)
R5 as White: +12 =29 -9, 53.0%      R5 as Black: +10 =24 -16, 44.0%
White scored 54.5% over all games
lanes: 39.3% / 53.6% / 47.7% / 54.5%
terminations: checkmate 47, threefold 49, insufficient material 4   (threefold 49%)
mean game 287 s, 110 plies; sims/move R5 770, R2 893; peak RSS 156 / 166 MB
distinct games 89 of 100 (eleven replays across lanes; the most deterministic pairing)
by opening: Caro-Kann 71%, QGD 67%, Slav 54%, Ruy 50%, KID 46%, Winawer 43%, English 33%, Najdorf 29%
```

### R4 d64 vs R2 — 100 games

```
R4 +17 =50 -33   score 42.0%   (95% Wilson 32.8%-51.8%)   implied -56 Elo (-125 to +12)
R4 as White: +11 =25 -14, 47.0%     R4 as Black: +6 =25 -19, 37.0%
White scored 55.0% over all games
lanes: 35.7% / 46.4% / 43.2% / 43.2%
terminations: checkmate 50, threefold 46, insufficient material 4   (threefold 46%)
mean game 291 s, 114 plies; sims/move R4 1,043, R2 887; peak RSS 175 / 139 MB
distinct games 98 of 100
by opening: Slav 58%, Caro-Kann/Winawer/QGD 50%, Najdorf 43%, Ruy 33%, KID 29%, English 21%
```

### RA d32 vs R2 — 100 games

```
RA +2 =17 -81   score 10.5%   (95% Wilson 5.9%-18.0%)   implied -372 Elo (-482 to -263)
RA as White: +1 =10 -39, 12.0%      RA as Black: +1 =7 -42, 9.0%
White scored 51.5% over all games
lanes: 8.9% / 17.9% / 13.6% / 0.0%
terminations: checkmate 83, threefold 15, insufficient material 2   (threefold 15%)
mean game 278 s, 99 plies; sims/move RA 1,106, R2 880; peak RSS 166 / 116 MB
distinct games 93 of 100
by opening: Ruy 29%, Najdorf 18%, QGD 12%, Caro-Kann/KID 8%, English/Winawer 4%, Slav 0%
```

RA is not drawing its way to 10%: 83 of its 100 games ended in checkmate, 81 of them
against it. Its threefold rate (15%) is a third of the others' because it rarely
reaches a position it can hold. Peak RSS is 116–175 MB on every side, far under 2 GB.

## 4. Decision this supports

**Keep d96 or move to d128; do not go smaller.** R5 is within noise of R2 after
100 games and pays only ~15–25% of simulations for +0.9 points of validation accuracy;
R4 and RA lose, RA decisively. Sample: 300 games at the competition clock, 100 per
pairing, no failed games, both colours balanced, rule off both sides. For the full
track this means d128 is a legitimate primary (R6 at d128 rather than d96 is
defensible, and its extra capacity is what a 100M shard would feed); the anchor
question DECISIONS §2 asked is answered against the reference's size.

What this does not settle: whether d128 is actually *better* than d96 (the interval
straddles zero; 400 games would resolve ±35 Elo), and how much a faster expansion
path would move every one of these numbers, since it is the binding cost.

## 5. Caveats

- One box, one night, six concurrent lanes on separate pinned cores.
- Forward and simulation rates on this box vary between runs on the same core
  (shared host, frequency and cache pressure); both sides of every game shared the
  same conditions at the same time, and the in-game rates above come from the games
  themselves.
- R2 played all three pairings with the rule off; its A1 form against R1 (ARENA5)
  was measured the same way, so the two reports are comparable.
- A few games are move-for-move replays of another game from the same book position
  (listed per pairing above).
- The box runs Python 3.14 against the platform's 3.12; onnxruntime 1.29, numpy 2.5
  and python-chess 1.11 match.

## 6. Artifacts

`sparring/bracket_box/lanes/B/` and `lanes/B2/` (pulled from the box), each lane with
`results.csv`, `meta.json`, one PGN and one log per game with both agents' per-move
telemetry; `summary.txt` per phase and the combined summary in `lanes/B_all/`.
Simulation-rate probe: `sparring/bracket_size/simrate.py`. Driver and queue:
`sparring/bracket_size/`.
