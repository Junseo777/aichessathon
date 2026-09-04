# Provenance: how to check these weights are ours

The rules permit engine-annotated training data but require that any model we
ship is one we trained, and say provenance is checked after the fact. This
document is the evidence and, more usefully, the instructions for testing it
yourself rather than taking our word for it.

Nothing here is a proof in the mathematical sense. What it does is make
fabrication far more expensive than simply training the model would have been.

## The one-command check

```
uv run --group train python -m train.verify_provenance ../checkpoints/R2_e8_ema.pt
```

The argument is a training checkpoint, `<run>_e<epoch>.pt` or its `_ema` twin.
Checkpoints live in the run store, not in this repository: `../checkpoints/`
beside it on a dev machine, `/workspace/checkpoints` on the training box.
`weights/` holds only the ONNX graph exported from one, which this check
cannot read; the export's `manifest.json` names the checkpoint it came from and
that file's SHA-256 (for exports made after this was added — R0, R1 and R2's
manifests predate it, and `weights/CHECKSUMS.txt` in the run store is the
record for those).

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

R3 carries the block but names `git_commit da3ba48`, a commit without the
provenance code: the box ran that code uncommitted, and it reached `main` as
`69e34ee`. Read `git_commit` as the last commit checked out on the box when the
run started, not as proof of the exact code that ran. Later runs (RA from
`f08a856` on the box's `pipeline` branch) were started from a commit that
includes it.

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
  monthly archive for 2026-07. Both public, both CC0. `pipeline/acquire.sh` also
  fetches the 2026-06 monthly (28 GB); it was downloaded for a cross-check and
  never filtered, so nothing from it is in any shard. `MONTHLIES=2026-07
  pipeline/acquire.sh` skips it.
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

`provenance/` contains, committed as the stages wrote them:

- `logs/` — stdout from every pipeline stage and training run, with timestamps
  and throughput. One file is filtered: `acquire.log` keeps the stage's own
  timestamped lines and its closing directory listing, and drops curl's
  progress meter, which was the other 148 KB of the 150 KB file on the box.
  `tr '\r' '\n' < acquire.log | grep -E '^\[[0-9:]+\]|^total|^d|^-rw'` on the
  original reproduces the committed copy byte for byte.
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
