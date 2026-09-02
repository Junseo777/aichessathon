# Pipeline brief — Stages 2 and 3

The working spec for the training-data pipeline and the training runs. It lives in
the repo so that a change to the code and a change to the spec land in the same
commit. `docs/DECISIONS.md` carries the reasoning; this file carries the contract
and the plan.

**Roles.** Justin runs the rented box: data, labels, training. Junseo runs
integration, search, arena and submission. Deliverables from the box are shards on
the volume and `.pt` checkpoints; nothing large moves between machines.

**Event.** Ladder opens Sept 4 (hourly rated rounds), 13-round Swiss over LOCKED
builds Sept 11, live final Sept 12. Final candidate checkpoints by Sept 9; the 11th
is the lock, not the target. A model that exists on the 4th beats a better one that
exists on the 9th.

---

## 0. Workflow rules

**Pull before every session, without exception.**

```
cd /workspace/aichessathon && git pull --rebase origin main && uv run pytest chessml/tests -q
```

`chessml/encoding.py` is the contract between shards and the shipped agent, and it
is still being corrected. It changed on Sept 2 (`54cf344`, the en passant plane).
**A shard built with an outdated encoder is silently wrong and the assertions cannot
catch it**, because assertion 3 compares the shard against the pipeline's own copy
of the encoder: an outdated copy agrees with itself perfectly. Any shard built
before a change to `chessml/encoding.py` must be regenerated.

**Push pipeline code to a branch**, not `main`:

```
git checkout -b pipeline && git add <files> && git commit && git push -u origin pipeline
```

Junseo reviews and merges. Never commit shards, PGNs, weights, or the Stockfish
binary; `weights/` and `data/` are gitignored and `/workspace/tools` is outside the
repo.

**Everything on the box runs inside tmux** (`tmux new -s work`) and lives on
`/workspace`. Anything outside the persistent volume is deleted when the pod stops.

**Never benchmark inference speed on the box.** All timing targets assume one weak
CPU core; many-core numbers are 5-10x optimistic and would push us toward a model
that flags. If a timing is ever needed: `taskset -c 0`, all thread counts 1.

**Engines at inference are banned** (retroactive disqualification). Engine
evaluations as training labels are confirmed permitted. Stockfish may label data on
the box; the binary and its network files never enter the repo or the submission.

---

## 1. The one hard rule

Board-to-array conversion exists and is verified. Import it; never reimplement it:

```
chessml/encoding.py
    featurize_int8(board)   -> (21, 8, 8) int8      the canonical encoding
    encode_move(move)       -> int in [0, 4672)
    mirror_move(move)       -> the move in the rotated (black-to-move) frame
    decode_move(idx, board) -> move                 for validation
    transposition_key(board)
```

`featurize_int8` already mirrors the board when black is to move; label moves must
be passed through `mirror_move` before encoding. Do not change `SCALE`.

---

## 2. Shard format

Memmapped binaries in one directory:

| File | Type | Shape | Contents |
|---|---|---|---|
| `X.bin` | int8 | (N,21,8,8) | `featurize_int8` of the position BEFORE the move |
| `Y_policy.bin` | int32 | (N,) | `encode_move` of the human move played, rotated frame |
| `Y_value.bin` | int8 | (N,) | game result +1/0/-1 from the side-to-move's perspective |
| `Y_value_engine.bin` | float16 | (N,) | engine value, side-to-move's perspective; NaN if unlabelled |
| `Y_policy_engine.bin` | int16 | (N,) | the engine's best move, `Y_policy_engine4[:, 0]` |
| `Y_policy_engine4.bin` | int16 | (N,4) | MultiPV moves, rotated frame; -1 where absent |
| `Y_value_engine4.bin` | float16 | (N,4) | their win-probabilities; NaN where absent |
| `Y_depth_engine4.bin` | int8 | (N,4) | their depths; -1 where absent |
| `Y_value_lichess.bin` | float16 | (N,) | the `[%eval]` value, kept so 6c can compare |
| `tier.bin` | int8 | (N,) | 0 = top (2400+), 1 = mid |
| `value_source.bin` | int8 | (N,) | 0 = none, 1 = Lichess `[%eval]`, 2 = our Stockfish |
| `Z.bin` | uint64 | (N,) | `chess.polyglot.zobrist_hash` of the stored position |
| `fen.txt` | text | N lines | sidecar, so labelling never re-parses archives |
| `fen_offset.bin` | uint64 | (N,) | byte offset of each row's line in `fen.txt` |
| `split.bin` | int8 | (N,) | 0 = train, 1 = validation |
| `meta.json` | | | n_samples, tier_counts, eval_coverage, seed, source_files |

The authority on this table is `ARRAYS` in `pipeline/shard.py`; it is the schema the
writer, the labeller, the validator and the trainer all share.

`Y_value_engine` is MultiPV column 0, so everything reading the scalar value target
is unchanged.

Validation split: hold out ~2% **by game**, not by position — positions within a
game are correlated and a by-position split leaks. Use the same game-hash rule for
every shard so the 40M validation games are also validation games at 100M.

---

## 3. Sources, filters, volume

**Sources** (both CC0): Lichess Elite (nikonoel) and Lichess standard monthly.

**Filters** (the reference's recipe, kept whole): both Elos parse; `|EloA - EloB| <=
400`; initial time control >= 180 s; `Termination == "Normal"`; result in
{1-0, 0-1, 1/2-1/2}; >= 10 plies; standard chess only. Dedup priority Elite over
monthly.

**Tier mix** 80% from 2400+, 20% from 1900-2400, none below. We want strength, not a
model of human play across levels.

**Interleave sources within each tier, not just tiers.** Measured: Elite carries
**0.00% `[%eval]`** (it strips all annotations) and alone supplies 110.9M positions,
so a shard filled tier-by-tier drains Elite first and contains zero Lichess evals,
which makes assertion 6c unrunnable. Weighted round-robin over (tier x source) by
filtered volume. The emission pattern must depend only on weights and history, never
on target size, so a 40M shard stays a byte-identical prefix of the 100M one.

**Volume:** 8M smoke shard (validation + first checkpoint), then 40M (~54 GB, the
ladder model and the size bracket), then 100M (~126 GB, generated Sept 4 while the
bracket trains). Same seed and sampling throughout so labels join across shards by
`Z.bin` and nothing is computed twice.

**Measured yields (Sept 2).** Elite, six months: 1.31M games kept of 1.71M seen,
110.9M positions, `[%eval]` 0.00% in both tiers. Lichess monthly 2026-07, 100M games
seen: top tier 281k games / 22.9M positions / 42.2% eval yield; mid tier 7.11M games
/ 495.5M positions / 15.2%. Interleaved 8M shard: 80.0/20.0 tier split, ~7.8% of rows
carrying a Lichess eval.

---

## 4. Labels

Human game outcomes are a poor label for position quality — a 2400 loses won
positions constantly, so "White eventually won" says little about whether White was
winning *here*. MCTS leans on the value head to decide what to explore, so value
quality converts directly into search strength. This is the largest lever available.

**Lichess `[%eval]` (free, ~7.8% of rows).** Centipawns from White's perspective, or
mate scores. Conversion: `v = 2/(1 + exp(-0.00368208 * cp)) - 1`; mates to +/-0.995;
perspective via python-chess `score.pov(board.turn)`. Set `value_source = 1`.

> **The off-by-one trap.** The `[%eval]` on move m evaluates the position AFTER m.
> A stored position's value comes from the node whose position IS that position —
> the parent of m's node. No statistical test catches a one-ply shift, because
> adjacent positions correlate strongly. Assertion 6b is the only guard.

**Stockfish self-labelling, MultiPV=4 (the whole label supply in practice).** Elite's
0% eval yield means this is not a top-up. 25k nodes per position with four lines
(about the depth 10k single-PV gave the top line), one engine process per core,
`Threads=1`, `Hash=16`, engines kept resident. Deduplicate by `Z` first (~30-40%
fewer positions), join back onto every row sharing that `Z` in both shards, set
`value_source = 2`. Store the four moves, their win-probabilities and their depths.

> **The fourth line is not a luxury — it is the soft policy target.** It is what
> replaces the policy half of distillation, which a scalar eval cannot supply. If
> throughput forces a cut, cut nodes, never MultiPV: the extra lines cannot be
> recovered without relabelling.

There is no off-by-one risk here (the stored FEN is evaluated directly), but
assertion 6 still applies to newly filled rows, plus 6c.

---

## 5. Assertions

A predecessor project lost years to these. All are silent; none throws on its own.
Keep them in a standalone validation script and run it on every shard.

1. **Position before move.** Decode `Y_policy` against the stored position; assert
   the move is LEGAL there, over every row of a large sample. The most important
   check in the project.
2. **Mirror.** For black-to-move rows, planes 0-5 hold BLACK's pieces. Planes 0 and
   6 are all-zero on board rows 0 and 7 — pawns on back ranks mean a mirror bug.
3. **No truncation.** Stored `X` rows are bit-identical to `featurize_int8`
   recomputed from the source position. (Only meaningful against a current
   `encoding.py`; see the pull rule.)
4. **Value sign.** A miniature where White mates stores `Y_value = +1` at the final
   white-to-move position.
5. **Determinism.** Regenerating with the same seed is byte-identical.
6. **Engine-value sign.** Over rows where the side to move went on to win and an
   eval exists, `mean(Y_value_engine)` is clearly positive, and
   `corr(Y_value_engine, Y_value) > 0`.
6b. **Engine-value attachment.** Hand-verify three (position, value) pairs at known
   plies from an embedded PGN, including one right after a large eval swing.
6c. **Source agreement.** On positions labelled by both sources (deliberately label
   ~50k Lichess-covered positions), `corr` between them `> 0.9`.
7. **Invariants** over >=100k random rows: planes 0-16 in {0,50}; planes 5 and 11
   each sum to exactly 50 per row; plane 19 == 50; plane 16 sums to 0 or 50;
   `Y_policy` in [0,4672); `Y_value` in {-1,0,1}; `Z` nonzero; every
   `Y_policy_engine` entry is -1 or a legal move in the stored position.
8. **Subset.** Every `Z` in the 40M shard appears in the 100M shard, and overlapping
   rows agree byte-for-byte.

---

## 6. Training

The model is defined in `train/model.py` (`ChessNet` + `Config`). **Do not modify the
architecture** — its components come from a full ablation study and its size from
measured single-core timings. You are writing the trainer.

**Loss.**

```
policy_loss  = cross_entropy(policy_logits, Y_policy)     # full 4672-way, no masking
value_target = Y_value_engine where present else Y_value  # masked select; never drop rows
value_loss   = mse(value_head, value_target)
loss         = policy_loss + VALUE_WEIGHT * value_loss
```

**Schedule.** AdamW lr 1e-3, weight decay 1e-4, batch 1024, BF16, gradient clipping
at norm 1.0, cosine decay to zero **across the epochs actually run**. 8 epochs over
40M. Shuffle every epoch, fixed seeds. EMA of weights (decay 0.999), saving both raw
and EMA checkpoints every epoch:

```
torch.save({"config": asdict(cfg), "model": state_dict}, path)
```

Keep every epoch on the volume — the last epoch is not automatically the best under
search, and selection happens by play on Junseo's side.

**The loader is the whole battle.** The reference stalled at ~2k samples/s on
disk-bound loading: 8 epochs over 40M would be 44 hours. Measure throughput in the
first five minutes of every run. Memmap with block-granularity shuffling may suffice
at 40M; at 100M bit-pack into RAM (12 piece planes as bitboards + ep square +
castling nibble + three scalar planes ~= 101 B/row, so 100M ~= 10 GB resident) and
unpack per batch on the GPU. If you pack, assert `unpack(pack(x))` equals
`featurize_int8` output bit-exactly over a large sample first. Target >= 20k
samples/s.

**Configs.**

```
d32  = Config(d_model=32,  n_heads=4, n_blocks=8)   ~0.12M   the reference's size
d64  = Config(d_model=64,  n_heads=4, n_blocks=12)  ~0.63M
d96  = Config(d_model=96,  n_heads=4, n_blocks=12)  ~1.38M
d128 = Config(d_model=128, n_heads=8, n_blocks=12)  ~2.44M
```

**Run matrix.** Fast track on 40M: R0 (d96, smoke shard) → R1 (d96, Lichess evals
only, the A/B baseline for full coverage) → R2 (d96, fully labelled, the ladder
model) → R3 (d96, value_weight 2.5) → R4 (d64) → R5 (d128) → RA (d32 anchor). Full
track on 100M at the chosen size: R6 (primary) → R7 (0.5/0.5 engine/outcome value
blend) → R8 (MultiPV soft policy target; R8b keeps a human component at alpha 0.5)
→ R9 (tau corpus) → R10 (on-distribution fine-tune).

R1 vs R2 measures what full engine coverage is worth. R2 vs R3 settles value weight.
R2/R4/R5/RA settle size — and RA is a real test, not a formality: if a 116k net with
~1,900 searches per move beats d96 with ~700, that is the model that ships.

**Gates on every run.** Val top-1 policy accuracy 40-55%, tracked **split by the
original side to move** (a collapse on one colour is a mirror bug, not a training
problem — stop and report). Val **value MSE** per epoch alongside accuracy; value
quality is what search amplifies and accuracy alone is the wrong headline number.
The final train-vs-val gap: converged and flat means capacity-saturated, still
climbing means data-limited.

---

## 7. Stop-and-report points

1. **8M shard built** — assertion results, tier counts, eval yield per tier, disk
   used, elapsed time. Wait for confirmation before the 40M run.
2. **R0 trained** — curves and val accuracy. `scp` the checkpoint immediately; do
   not wait for better models.
3. **Part C coverage on 40M** — final coverage number, throughput, and R1 results.
4. **Fast track complete** — all checkpoints, plus R2's per-epoch train and val
   loss. This gates the full track: Junseo arenas the bracket and picks the size,
   and R2's gap says whether 100M earns its cost.
5. **Final, Sept 8-9** — every checkpoint (raw and EMA), per-run curves, val accuracy
   split by colour, val value MSE, train/val gap, loader throughput, wall time; plus
   final coverage on both shards, tier counts, tau corpus location and size, and
   which of R7-R10 ran.

If behind, cut from the bottom: R10, R8, R7, then R9. Never cut the assertions, the
smoke shard, R0-R2, RA, or the Stockfish labelling.

---

## 8. Teardown

Confirm Junseo has loaded every checkpoint before touching the pod. Leave the
filtered PGNs, FEN sidecars, shards, tau corpus and label files on the volume
through Sept 12. Terminate the pod (a stopped pod still bills); the volume persists
independently. Junseo deletes the deploy key. Confirm nothing under
`/workspace/tools` ever entered the repo: `git status --ignored`.
