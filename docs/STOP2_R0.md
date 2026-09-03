# STOP AND REPORT #2 — R0

Status report from the Stage 2/3 worker (Part A box). Written to be read cold:
everything needed to act on it is below, with no reference to prior chat.

Date: 2026-09-03. Box: RunPod, EU-RO-1, 1x RTX 4090, Ryzen 9 7950X (16 physical
cores / 32 threads), 500 GB network volume at `/workspace`.

**Memory: the container cgroup is capped at 61 GB.** `free` reports the host's
124 GB and is misleading; `/sys/fs/cgroup/memory.max` is 60,999,999,488. An
earlier revision of this document said 124 GB. Budget 61 GB, shared with
whatever else is running on the box.

Pipeline code is pushed to branch `pipeline` (commits `17a32e6`, `0aaf157`),
based on `51ac5d2`. Not yet merged.

---

## 1. R0 — the first trained checkpoint

`d96 x b12`, 1,381,050 params, 8 epochs over the fully-labelled 8M shard,
value_weight 1.0, batch 1024, AdamW lr 1e-3 wd 1e-4, cosine to zero, BF16,
grad clip 1.0, EMA decay 0.999, seed 0.

| epoch | train_pol | val_pol | gap | val_acc | W / B | val_mse | ema_acc |
|-------|-----------|---------|---------|---------|-------------|---------|---------|
| 1 | 2.3241 | 1.7826 | +0.5415 | 0.4418 | .4416/.4419 | 0.0373 | 0.4450 |
| 2 | 1.6799 | 1.6132 | +0.0667 | 0.4788 | .4790/.4786 | 0.0351 | 0.4888 |
| 3 | 1.5632 | 1.5427 | +0.0204 | 0.4955 | .4954/.4955 | 0.0254 | 0.5046 |
| 4 | 1.4993 | 1.5037 | -0.0045 | 0.5062 | .5068/.5057 | 0.0231 | 0.5143 |
| 5 | 1.4514 | 1.4771 | -0.0257 | 0.5125 | .5121/.5130 | 0.0228 | 0.5176 |
| 6 | 1.4123 | 1.4589 | -0.0466 | 0.5172 | .5175/.5168 | 0.0216 | 0.5200 |
| 7 | 1.3815 | 1.4501 | -0.0685 | 0.5202 | .5205/.5198 | 0.0207 | 0.5208 |
| 8 | 1.3636 | 1.4496 | -0.0860 | 0.5210 | .5212/.5209 | 0.0205 | 0.5211 |

- **Val top-1 52.10%**, above the expected 40-55% band.
- **Colour split balanced to three decimals at every epoch.** No mirror bug.
- **Validation saturated, not truncated**: epoch 8 gained +0.0008 while the
  train/val gap widened to -0.0860. 8 epochs is the right horizon for 8M.
- Loader **26,960 samples/s** solo, 21,800 once Part C competed for CPU.
  Target was >= 20k. RAM-resident loader, not memmap.

Artifacts: `R0_e8.pt`, `R0_e8_ema.pt`, `R0_history.json`, and fp32 ONNX for
both. On the box at `/workspace/checkpoints` and `/workspace/weights`.

## 2. ONNX export and the parity line

```
R0_e8      fp32 EXACT | int8 argmax agreement 0.953 | int8 value err 0.1050
R0_e8_ema  fp32 EXACT | int8 argmax agreement 0.922 | int8 value err 0.0916
```

Both below the 0.99 gate, so **`model.int8.onnx` was deleted for both before
any game was played**. fp32 is 5,681,715 B, well inside the 50 MB budget.

Two findings worth carrying forward:

- **EMA quantises worse than raw** (0.922 vs 0.953), which is the opposite of
  the usual expectation.
- The **value error matters more than the argmax**. 0.10 on a [-1,1] scale is
  injected at every one of ~700 search nodes; the argmax figure understates
  how bad int8 would be under PUCT.

`train/export_onnx.py` deletes all three output files when the strict int8
check fails, including the good fp32 model. The export was therefore driven by
calling `export()` and `verify_parity(strict_int8=False)` directly rather than
through the CLI. **Worth fixing in that script** so a failing int8 check keeps
the fp32 artifact.

## 3. Part C on the 8M shard — 100% coverage

6,550,441 unique positions (81.9% of rows), Stockfish 18, MultiPV=4,
`Limit(nodes=25000)`, 16 workers, 214 minutes at 509 pos/s. All 8,000,000 rows
filled via the Z join.

Assertions (12 passed, 0 failed):

```
1   position-before-move: policy decodes to a legal move
2   mirror: own pieces in planes 0-5, no back-rank pawns
3   no truncation: X bit-identical to featurize_int8
4   value sign: White mates -> +1 at final white-to-move row
5   determinism: 13/13 files byte-identical on rebuild
6   engine-value sign: mean|won +0.2442, corr +0.5056, n=8,000,000
6b  eval attachment: 4 hand-checked plies incl. post-swing row
6c  stockfish-vs-lichess corr = 0.9846 over 709,088 rows labelled by both
6d  Y_*_engine == Y_*_engine4[:,0] byte-exact, 8,000,000 rows
6e  MultiPV rows monotone / legal / distinct, 15,000 rows
6f  human move within engine top-4: top 81.7% (top-1 45.8%, n=400,145)
                                    mid 75.3% (top-1 39.9%, n= 99,855)
                                    all 80.4% (top-1 44.6%)
7   plane and label invariants, 500,000 rows
8   subset: all 8,000,000 Z of the 8M shard match the 40M prefix
```

6c at 0.985 says our node-limited evals agree with Lichess's deeper ones in
scale. 6f's 81.7% on the top tier sits inside the predicted 70-85%, ruling out
a mirror or encoding bug in the new MultiPV columns.

## 4. Part C configuration sweep

100,000 identical positions per config. Depth configs carry a 150,000-node
safety ceiling.

| config | wk | pos/s | medNodes | p90Nodes | medDepth | mixed | disagree |
|--------|----|-------|----------|----------|----------|-------|----------|
| nodes=25000 | 16 | 534 | 25,015 | 25,029 | 10 | 0.675 | 0.209 |
| nodes=10000 | 16 | 1,200 | 10,005 | 10,012 | 8 | 0.688 | 0.222 |
| depth=10 | 16 | 323 | 39,258 | 80,613 | 10 | 0.003 | 0.001 |
| depth=12 | 16 | 142 | 112,209 | 150,129 | 12 | 0.229 | 0.065 |
| depth=9 | 16 | 548 | 21,678 | 45,100 | 9 | 0.001 | 0.000 |
| **depth=8** | 16 | **857** | 12,263 | 25,962 | 8 | **0.000** | **0.000** |
| nodes=25000 | 32 | 359 | 25,015 | 25,029 | 10 | 0.671 | 0.207 |
| nodes=10000 | 32 | 857 | 10,005 | 10,012 | 8 | 0.686 | 0.220 |
| depth=10 | 32 | 208 | 39,196 | 80,442 | 10 | 0.003 | 0.001 |
| depth=12 | 32 | 88 | 112,083 | 150,128 | 12 | 0.228 | 0.064 |

**16 workers beats 32 in every config** — one process per physical core;
SMT contention plus 32x16 MB of hash costs more than the extra threads return.

**Chosen for the 40M pass: `depth=8`, MultiPV=4, 16 workers.** Rule (a) selects
depth=9 (median 21,678 nodes, nearest 25k among depth configs clearing
0.75 x 534 = 400); 548 < 700 so rule (c) fires and takes one step shallower.
Against the other rule-(c) option, `nodes=10000`: same median depth 8, but
**0.0% mixed depths against 68.8%**, for 29% less throughput.

`depth=12` is worse than `depth=10` on mixed depths (0.229 vs 0.003) because
its p90 hits the 150k ceiling exactly — the ceiling binds and reintroduces
mid-iteration stops.

## 5. Arena smoke test — R0 loses

```
. vs baselines/reference-hero, 20 games, base 5000 ms
+2 =12 -6, score 40.0%
terminations: checkmate 8, threefold_repetition 12
```

The export and arena path work end to end: the agent loads `model.onnx`,
pre-searches the opening, and plays legal games. As a strength result, two
things need flagging honestly:

- **12 of 20 games ended in threefold repetition (60%).** That is high enough
  to be a behavioural issue rather than a strength signal, and deserves its own
  investigation.
- **40% is a real early data point against the d96 size choice**, arriving
  before RA was scheduled. It is heavily confounded: R0 saw only 8M positions
  (a fifth of 40M, a twelfth of the 100M target), and at a 5 s base the 116k
  hero gets roughly 5x the simulations d96 does. No size conclusion should be
  drawn until R2 at 40M on a realistic clock.

## 6. Corpus state

| item | state |
|------|-------|
| Raw archives | 54 GB. 6 Elite months (2025-06..11), 2 monthlies (2026-06/07) |
| Filtered | Elite 1,307,125 games / 110.9M positions; monthly 2026-07 7,394,952 games / 518.3M positions |
| 8M shard | 8,000,000 rows, tier 80.0/20.0, 8.86% Lichess eval coverage, 100% Stockfish coverage |
| 40M shard | 40,000,000 rows from 481,397 games, tier 80.0/20.0, 8.83% Lichess eval coverage |
| 100M shard | not built |
| Tau corpus | not built |

Eval yield per tier, measured: **Elite 0.00% both tiers** (the database strips
all annotations at source — 0 `[%eval]` and 0 `[%clk]` in 6,273 sampled games);
**monthly 2026-07: top 42.23%, mid 15.17%**. 2026-06 is downloaded but
deliberately unfiltered.

Shards interleave on two levels, tier and source, both deterministic and
weighted by measured filtered volume. The source level is load-bearing: without
it a tier would drain Elite first and the shard would contain zero Lichess
evals, leaving 6c with nothing to cross-check.

## 7. In flight

**Part C on the 40M shard**, `depth=8`, 16 workers: 31,362,459 unique positions
(78.4% of rows), ~878 pos/s, ETA ~9 hours from 23:19 UTC 2026-09-02.

All 31.4M are being relabelled rather than reusing the 6.55M already done at
`nodes=25000`. That costs ~2 extra hours and keeps the 40M corpus internally
homogeneous; mixing two labelling regimes inside one training set is the kind of
silent inconsistency that is expensive to discover later. The 8M shard keeps its
25k-node labels, so each shard is self-consistent.

## 8. Addendum — R1, and two OOM kills

R1 (d96 x b12, 40M shard, Lichess evals only, the A/B baseline against R2's
full engine coverage) was launched, killed, fixed, killed again, and is now
running. Confirmed as the right run by its loader line: `engine-labelled 8.84%`.

Both kills were the 61 GB cgroup cap, recorded as `oom_kill 2` in
`memory.events` with `memory.peak` at 61,004,386,304.

- **First kill, during loading.** `RamSplit` did `np.asarray(shard.X)[idx].copy()`.
  The fancy index already returns a fresh array, so the copy allocated a second
  full-size buffer while the first was live: 53.8 GB doubled to ~107 GB.
  Harmless at 8M (~10 GB), fatal at 40M.
- **Second kill, at the first training step**, after dropping the copy. 52.7 GB
  of train rows plus 1.1 GB val plus ~6 GB of Part C plus the CUDA context
  still exceeds 61 GB.

Fixed in `5d67ce1` by not holding X in RAM at all. `train/loader.py` adds
`BlockShuffledSplit`: X stays a memmap, read in contiguous 262,144-row blocks
whose order is shuffled each epoch, sixteen blocks buffered and shuffled
together. 5.6 GB resident against 52.7 GB on disk, sequential reads rather than
random, shuffling across 4.2M rows. Splits under 12 GB still load to RAM, so
the 8M path is unchanged. This is the brief's Plan A, arriving earlier than
planned; bit-packing is still wanted for 100M.

Alignment was verified rather than assumed: with `rng=None` the k-th sample
must be split row `row_of[k]`. Over 20,480 samples, zero X mismatches and zero
policy mismatches. A misalignment here would have produced a normal-looking
learning curve.

R1 now runs at 30.3 GB of the 61 GB cap, 18,635 samples/s, ~35 min/epoch,
~4.7 h for 8 epochs.

`train/export_onnx.py` also fixed in `5d67ce1`: an int8-agreement failure now
removes only `model.int8.onnx` and keeps fp32 plus the manifest; an fp32 parity
failure still deletes everything and raises.

## 9. Open items

1. **R0 is not on the submission side.** `R0_e8_ema.onnx` needs to land in
   `weights/model.onnx` with a manifest of `{"d_model":96,"n_heads":4,
   "n_blocks":12}`. fp32 only, do not quantise. This is a human action; the
   ladder opens Sept 4 and until it happens there is no trained checkpoint on
   the submission side.
2. **Assertion 3 cannot catch a stale encoder.** It compares a shard against
   whichever copy of `chessml/encoding.py` is on the box, so an outdated copy
   agrees with itself perfectly. The current shards were separately verified
   against `origin/main`'s encoder — 60,000 sequential plus 30,000 random rows
   per shard, zero mismatches, with ~2,000 ep-square rows per sample exercising
   the changed branch. That check should become part of the suite.
3. **Bit-packing for the 100M shard** is still required; the block-shuffled
   loader solves 40M but re-reads 52.7 GB per epoch.
4. **R2 is gated** on the 40M Part C pass (~7.5 h remaining at time of writing).
5. `pipeline` branch is unmerged.

**Resolved since first writing:** the 60% threefold rate in R0's arena is
`agent.py:200-208` working as designed — `q_best < -0.3` makes the agent seek a
referee draw claim, and against a stronger opponent R0 sits below that
threshold most of the game. 40% is a floor inflated by draw-seeking, not
evidence against d96.
