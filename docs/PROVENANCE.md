# Provenance: how to check these weights are ours

The rules permit engine-annotated training data but require that any model we
ship is one we trained, and say provenance is checked after the fact. This
document is the evidence and, more usefully, the instructions for testing it
yourself rather than taking our word for it.

Nothing here is a proof in the mathematical sense. What it does is make
fabrication far more expensive than simply training the model would have been.

## The one-command check

```
uv run --group train python -m train.verify_provenance weights/model.pt
```

Add `--shard <path>` to also re-evaluate the checkpoint and confirm it still
produces the metrics recorded inside it.

Three things are checked.

## 1. Structure — a foreign network cannot pass

`ChessNet` in `train/model.py` is a square-token transformer: 21-plane input at
`SCALE = 50`, 64 tokens, pre-norm blocks, a learned 15x15 relative-position
bias per attention head, a 4,672-logit policy head shaped as 64 x 73, and a
scalar `tanh` value head. At `d96 x b12` that is exactly **171 tensors and
1,381,050 parameters**.

The distinguishing tensor is `blocks.N.attn.rel_bias`, shape `(n_heads, 225)` —
the geometry bias. No public chess network has this parameterisation. Lc0 nets,
Maia, and Stockfish's NNUE are all different graphs with different tensor names
and shapes; none of them can be loaded into `ChessNet`, and none of them can be
exported to the ONNX graph this repository produces.

The verifier rebuilds the architecture from the checkpoint's own config and
compares tensor names and shapes exactly. A checkpoint carrying someone else's
weights fails with named missing and unexpected tensors, not a vague warning.

## 2. Provenance block — what the file says made it

Checkpoints written after `69e34ee` carry a `provenance` block: run name, epoch,
seed, value weight, batch, learning rate, git commit, the shard path with a
SHA-256 of its `meta.json` and its row count, that epoch's metrics, a UTC
timestamp, torch version, and host.

R0, R1 and R2 predate this and carry only `config` and `model`. Their evidence
is section 3.

## 3. Reproduction — the part that is hard to fake

Every run keeps a checkpoint at **every epoch**, not just the last, alongside
`<run>_history.json`. So there is a chain to check rather than a single claim:

```
load R2_e3.pt, evaluate on the validation split
  -> must reproduce val_pol 1.3925, acc 0.5360, val_mse 0.0185
```

and the same for all eight epochs, each matching its own recorded row, with the
sequence forming a plausible monotone descent. Reproducing that without having
trained the model would require producing eight separate networks that each hit
a specific score on a specific held-out set — which is harder than training it.

The validation split is not arbitrary either. It is chosen **by game** via a
hash of the game's identity (`build_shard.py: game_key`, `split_of`), so it is
reproducible by anyone who regenerates the corpus, and it cannot be gamed by
choosing a favourable split after the fact.

## 4. The corpus is reproducible from public data

- Sources: Lichess Elite database (2025-06 .. 2025-11) and the Lichess standard
  monthly archive for 2026-07. Both public, both CC0.
- `pipeline/filter_pgn.py` applies the filters; `provenance/` holds the filter
  reports with exact game and position counts per tier.
- `pipeline/build_shard.py` is deterministic: same sources, same weights, same
  target produces a byte-identical shard. This is assertion 5 in
  `pipeline/validate.py`, and it was verified across all 13 shard files.
- Training uses a fixed seed (0).

So the whole chain — public archives, filter, shard, labels, training — can be
re-run. GPU nondeterminism means the weights will not be bit-identical, but the
learning curve and final metrics will land in the same place.

## 5. Raw artifacts

`provenance/` contains, committed and unedited:

- `logs/` — stdout from every pipeline stage and training run, with timestamps
  and throughput
- `history/` — per-epoch metrics for every run
- `filter_*.json` — corpus composition per tier

These are cheap to keep and awkward to fabricate consistently, because they have
to agree with each other, with the checkpoints, and with the shard manifests.

## What this does not prove

It does not prove the *data* was untampered with, only that the model came from
the pipeline in this repository. It does not prove no human intervened between
epochs. And a determined faker with enough GPU time could produce a consistent
chain — but at that point they would have trained a model, which is the thing
the rule asks for.
