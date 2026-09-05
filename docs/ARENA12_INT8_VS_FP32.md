# ARENA #12 — int8 R1 against fp32 R1

Every trained checkpoint has failed the exporter's int8 gate, 99% top-move agreement
with fp32, so only fp32 has ever played. The gate is a policy, not a measurement of
strength: int8 runs the forward pass faster, and on a quiet core the forward pass is
about 90% of a simulation's cost (ARENA9 §2). This arena asks whether the extra search
buys more than the rounding costs. Written to be read cold.

Date: 2026-09-05, 06:00–07:25 UTC. Box: RunPod, Ryzen 9 7950X, one physical core per
agent via `taskset`, six lanes on cores 0–11. Harness: `harness.referee.play_match` at
the competition clock, 120 s + 0.5 s, 300-ply cap, `harness/rules.py` defaults,
unmodified. Code: `aichessathon` at **f2a78b8** on both sides; the same net R1 on both
sides; draw rule off both sides (±1.00); pondering on, as shipped. Eight book openings
with both colours, 50 games per colour.

## 1. The int8 export

`sparring/bracket_size/int8_export.py` applies the exporter's own conversion
(onnxruntime `quantize_dynamic`, QInt8 weights) to `R1_e8_ema/model.onnx` with the gate
switched off, into an agent directory holding `model.int8.onnx` alone, so `load_fastest`
has nothing else to choose. The run store is untouched. Measured on one pinned core
before the games, on 256 positions from random playouts:

| | fp32 | int8 |
|---|---|---|
| top-move agreement with fp32 | 1.000 | **0.977** |
| largest value-head difference | 0 | 0.185 |
| forward pass | 2.20 ms | **1.50 ms** (1.47×) |
| file size | 5.68 MB | 1.72 MB |

The agreement is not the 0.89–0.91 the exporter reported for every checkpoint. The
exporter's parity check feeds random tensors drawn from {0, 0.5, 1} at fixed rates,
not chess positions; on real positions the quantised policy head agrees 97.7% of the
time. The value head's error, up to 0.18 on a scale whose typical magnitude is 0.39,
is the larger distortion. Both init lines in every game confirm which file played:
`chosen: model.int8.onnx` at 1.50 ms on one side, `model.onnx` at 2.19 ms on the other.

## 2. Result

**int8 does not beat fp32: 46.0% over 100 games, −28 Elo (95% −96 to +40), while
searching 1.41× as many nodes per move.**

```
games 100   int8 +15 =62 -23   score 46.0%   (95% Wilson 36.6%-55.7%)
implied int8 - fp32: -28 Elo (95% -96 to +40)
int8 as White: +11 =30 -9, 52.0%      int8 as Black: +4 =32 -14, 40.0%
terminations: checkmate 38, threefold 60, insufficient material 2   (threefold 60%)
mean game 290 s, 117 plies
sims/move: int8 2,149 (pondered 1,931), fp32 1,520 (pondered 1,337)   ratio 1.41
peak RSS 305 / 236 MB; distinct games 66 of 100; failed games: none
by opening (int8): KID 62%, English 58%, Caro-Kann 54%, QGD 50%, Winawer 46%,
                   Najdorf 46%, Slav 42%, Ruy 12%
```

Contention: 4,754 throttled periods during the phase, about 9%, from another session's
training run and arena that shared the box; both sides of every game shared it.

## 3. Reading it

1. **The speed does not pay for the noise.** 41% more search is about 0.5 doublings.
   Whatever a doubling is worth (ARENA10), int8 gave that up and 28 Elo more. With a
   ±68 Elo interval this is not a proof that int8 is worse, but it is no evidence that
   it is better, and shipping needs evidence.
2. **The value head is the likely culprit, not the policy.** 97.7% top-move agreement
   is high; a value error of 0.18 is half of a typical evaluation, and the search
   averages values into every decision. Quantising the trunk and leaving the value head
   in fp32, or static quantisation with calibration on real positions, are the untested
   variants.
3. **Fix the gate's inputs.** The exporter's parity check should featurise real
   positions, not random tensors, whatever threshold it keeps; today it reports a
   number that means nothing.
4. **Do not ship both files.** `load_fastest` picks the faster file at init; on this
   box that is int8, which plays worse. Until an int8 export wins an arena, the zip
   must carry fp32 alone, which is what `make zip` does today.

## 4. Decision this supports

**Keep shipping fp32.** Sample: 100 games at the competition clock, no failures. What
would change it: an int8 variant that keeps the value head in fp32 scoring 50% or
better against fp32 over 100 games, or ARENA10 showing a doubling is worth so much that
a retest with more games is justified.

## 5. Artifacts

`sparring/bracket_box/lanes/K/`: six lanes with `results.csv`, `meta.json`, one PGN and
one log per game; `summary_local.txt`. Agent `/workspace/bracket/agents/r1_int8_t1.00`
(int8 sha `a090d234…`); queue `sparring/bracket_size/queue_k.sh`, log `queue_k.log`;
`bracket.py` gained a fallback so it can checksum an int8-only agent.
