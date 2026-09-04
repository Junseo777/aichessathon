# STOP AND REPORT #3 — R1, R2, and Part C on the 40M shard

Continues `docs/STOP2_R0.md`. Written to be read cold. Covers everything since
R0 was trained on the 8M shard.

Date: 2026-09-03 16:30 UTC. Branch `pipeline`, based on `51ac5d2`, unmerged.

---

## 1. Headline

Part C finished on the 40M shard at 100% coverage, and R1 and R2 both trained
to completion. The A/B they were built to settle came back near-null, and the
reason is structural rather than a measurement problem — see section 4.

| run | shard | value target | val acc | ema acc | val_pol | val_mse |
|-----|-------|--------------|---------|---------|---------|---------|
| R0 | 8M | engine (100%) | 0.5210 | 0.5211 | 1.4496 | 0.0205 |
| R1 | 40M | Lichess only (8.84%), outcome elsewhere | 0.5516 | 0.5517 | 1.3363 | 0.5382 |
| R2 | 40M | engine (100%) | **0.5528** | **0.5528** | **1.3331** | 0.0152 |

R3 (`value_weight=2.5`, otherwise identical to R2) is training now.

## 2. Two OOM kills, and a number that was wrong

R1 was launched, killed, fixed, killed again, and only then ran. Both kills
were the container memory cap.

**The box reports 124 GB via `free` but the cgroup caps at 61 GB.**
`/sys/fs/cgroup/memory.max` = 60,999,999,488; `memory.peak` reached
61,004,386,304; `memory.events` records `oom_kill 2`. `free` reports host
memory, not the container's allowance. `docs/STOP2_R0.md` originally stated
124 GB and has been corrected. **Budget 61 GB, shared with everything else on
the box.**

Cause of the first kill: `RamSplit` did `np.asarray(shard.X)[idx].copy()`. The
fancy index already returns a fresh array, so the copy allocated a second
full-size buffer while the first was live — 53.8 GB doubled to ~107 GB at 40M.
Harmless at 8M (~10 GB).

Cause of the second, after dropping the copy: 52.7 GB of train rows plus 1.1 GB
val plus ~6 GB of Part C plus the CUDA context still exceeds 61 GB.

**Fix (`5d67ce1`):** X is no longer held in RAM. `train/loader.py` adds
`BlockShuffledSplit` — X stays a memmap, read in contiguous 262,144-row blocks
whose order is shuffled per epoch, sixteen blocks buffered and shuffled
together. 5.6 GB resident against 52.7 GB on disk, sequential reads rather than
random, shuffling across 4.2M rows. Splits under 12 GB still go to RAM, so the
8M path is unchanged. This is the brief's Plan A, arriving earlier than planned.
Bit-packing is still wanted for the 100M shard, which will not fit this way
either.

Alignment was verified before trusting it: with `rng=None` the k-th sample must
be split row `row_of[k]`. Over 20,480 samples, **zero X mismatches and zero
policy mismatches**. A misaligned loader produces a normal-looking learning
curve, so this check is not optional.

Throughput after the fix: 20,500 samples/s with Part C competing, **25,000-28,700
once Part C finished**. Target was >= 20k.

## 3. Part C on the 40M shard — complete

`depth=8`, MultiPV=4, 16 workers, ~880 pos/s.

```
rows            40,000,000
unique          31,362,459   (78.4% of rows)
rows labelled   40,000,000   coverage 100.0%
unique labelled 31,362,459   coverage 100.0%
```

All 31.4M unique positions were labelled fresh at `depth=8` rather than reusing
the 6.55M already done at `nodes=25000`, so the 40M corpus is internally
homogeneous. The 8M shard keeps its 25k-node labels; each shard is
self-consistent, the two are not mixed.

## 4. The A/B came back near-null, and why

R1 and R2 are identical in every respect except the value target: same config,
same seed, same shard, same 8 epochs. R2 had 100% engine coverage, R1 had 8.84%.

**R2 beat R1 by +0.0012 val accuracy — 0.5516 to 0.5528.** Effectively nothing.

This is not a measurement failure, it is what the experiment was always going
to show. **Part C only replaces the value target.** The policy target is the
human move in both runs, so policy accuracy could not move much, and did not.

The `val_mse` column cannot be used to compare them either: R1's is measured
against game outcomes (+-1, high variance) and R2's against engine values
(smooth, small magnitude). 0.5382 vs 0.0152 is two different questions, not a
35x improvement.

**So the value of Part C is still unmeasured.** The brief's thesis is that the
value head is what ~700 PUCT simulations amplify, so a better value head can be
worth real Elo at identical policy accuracy. Nothing in the loss curves can
confirm or refute that. What would:

- an R1 vs R2 head-to-head arena (the direct test), or
- scoring both models' value heads against the *same* criterion — engine value
  on the held-out split — which R1 was not trained on but can be evaluated
  against.

Neither has been run yet. **Do not conclude Part C was wasted from the +0.0012;
that number cannot carry the claim.** Equally, do not conclude it helped.

## 5. Exports and int8

Every checkpoint fails the 0.99 int8 gate, so `model.int8.onnx` is deleted on
every export and fp32 ships. fp32 parity is exact in all six cases.

| checkpoint | int8 argmax | int8 value err |
|------------|-------------|----------------|
| R0_e8 | 0.953 | 0.1050 |
| R0_e8_ema | 0.922 | 0.0916 |
| R1_e8 | 0.918 | 0.0004 |
| R1_e8_ema | 0.879 | 0.0004 |
| R2_e8 | 0.914 | 0.1970 |
| R2_e8_ema | 0.922 | 0.1744 |

Two observations. The failure is consistent across sizes and training regimes,
so int8 is not viable for this architecture at d96 — it is not a
one-checkpoint accident. And the two failure modes move independently: R1's
value error is 0.0004, 250x better than R0's, while its argmax agreement is
*worse*. Quantisation damages the policy head and the value head separately.

`train/export_onnx.py` used to delete `model.onnx`, `model.int8.onnx` and
`manifest.json` together whenever `verify_parity` raised, so every export
destroyed the fp32 model it had just produced. Fixed in `5d67ce1`: an
int8-agreement failure removes only `model.int8.onnx` and keeps fp32 plus the
manifest; an fp32 parity failure still deletes everything and raises.

## 6. Checkpoint provenance

Checkpoints previously recorded only `{"config", "model"}` — nothing about what
produced them. They now carry a `provenance` block: run name, epoch, seed,
value weight, batch, lr, git commit, the shard path with a SHA-256 of its
`meta.json` and row count, that epoch's metrics, UTC timestamp, torch version
and host. Backward compatible; `export_onnx.py` reads `config` and `model` only.

R0, R1 and R2 predate this and have no provenance block. Their evidence is the
per-epoch checkpoints plus `*_history.json`, which is a verifiable chain: each
`R*_eN.pt` must reproduce its recorded row in the history when evaluated on the
val split.

Worth recording for the "did you train this" question, since it is checked
after the fact: the checkpoint is 171 tensors matching `train/model.py`
exactly, including `blocks.N.attn.rel_bias (4, 225)` — the learned 15x15
relative-position geometry bias. No public chess network has that
parameterisation; Lc0, Maia and Stockfish NNUE cannot be loaded into
`ChessNet`. A plagiarised model cannot produce this graph.

## 7. Corpus and box state

| item | state |
|------|-------|
| 8M shard | 8,000,000 rows, 100% Stockfish coverage (`nodes=25000`) |
| 40M shard | 40,000,000 rows, 100% Stockfish coverage (`depth=8`), tier 80.0/20.0 |
| 100M shard | not built |
| Tau corpus | not built |
| Spend | $13.00 of $150 as of 09:22 UTC Sept 3 |

Box: 1x RTX 4090, 32 vCPU, **61 GB cgroup cap**, 500 GB volume at `/workspace`.
Pod 1 only; a second pod was discussed but not provisioned.

## 8. Open items

1. **R0/R1/R2 are still not on the submission side.** `weights/` on Junseo's
   machine is the random-init export. R2 is the current best: drop
   `R2_e8_ema/model.onnx` in as `weights/model.onnx` with its `manifest.json`.
   fp32 only, do not quantise. The ladder opens Sept 4. *(Done 2026-09-03:
   `R2_e8_ema` is active.)*
2. **Part C's value is unmeasured.** See section 4. Needs an arena or a
   common-criterion value comparison before the run matrix leans on it.
3. **Bit-packing for the 100M shard.** The block-shuffled loader handles 40M
   but re-reads 52.7 GB per epoch, and 100M will not fit the 61 GB cap.
4. **Assertion 3 cannot catch a stale encoder** — it compares a shard against
   whichever copy of `chessml/encoding.py` is on the box. Both shards were
   separately verified against `origin/main`'s encoder (60,000 sequential plus
   30,000 random rows each, zero mismatches, ~2,000 ep-square rows per sample),
   but that check is not in the suite.
5. **The box idled ~3.5 hours** between R2 finishing and R3 starting, roughly
   $2.70. Nothing watches for a finished run and starts the next one.
6. `pipeline` branch is unmerged. *(Merged in `e77ee46`.)*
