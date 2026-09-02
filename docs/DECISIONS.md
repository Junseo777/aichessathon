# Design decisions

Every material decision in the project so far, with the reasoning and the evidence
behind it. Where a number was measured it says so; where it is an estimate it gives
a range. Decisions are listed in dependency order: each section assumes the ones
above it.

Reference project throughout: `pbaer/neural-chess` (MIT). Its README and code were
read in full; its measured results are cited as "the reference measured".

---

## 0. The constraints everything else follows from

| Constraint | Value | Consequence |
|---|---|---|
| Compute at play time | 1 dedicated CPU core, no GPU, 2 GB RAM | model size is paid for in search depth |
| Clock | 120 s + 0.5 s/move | ~2.5 s of thinking per move, sustainably |
| Language/runtime | Python 3.12, torch-CPU, numpy, python-chess, onnxruntime preinstalled | interpreter overhead is a first-order cost |
| Init | 60 s before the clock starts | load, warm up, probe hardware, pre-search for free |
| Process model | one process per game, alive between moves, pondering allowed | history tracking and pondering are possible |
| Failure modes | illegal move, crash, OOM, flag: all full losses | reliability is worth as much as strength |
| Draws | referee auto-claims threefold and fifty-move; ply 300 adjudicated on material | the agent must track repetition itself |
| Rules | no Stockfish/Lc0/Maia or wrappers at inference; source must be judge-readable; engine-derived TRAINING labels confirmed permitted | Stockfish may label data, never play |
| Event | ladder Sept 4-11 seeds a 13-round Swiss on locked builds Sept 11; live final Sept 12 | ten days total; execution risk dominates |

The single most important fact in this table: **parameters cost simulations.** A
network is evaluated once per position the search visits. Every design decision
about size, runtime and search is a consequence of that.

---

## 1. Architecture: what we kept from the reference and why

**Decision: the reference's v3.1 square-token transformer, unchanged.**

- 21-plane input (own/opponent pieces, castling, en passant, clocks, side, repetition)
- board mirrored for black so one set of weights plays both colours
- 4,672-logit AlphaZero move encoding with legal-move masking
- pure attention over the 64 squares, no conv stem, learned 2D relative-position
  ("geometry") bias, pre-norm blocks, FFN x4
- joint policy head and scalar tanh value head

**Why:** the reference ran a full ablation campaign on exactly this architecture and
found it tight. Removing the geometry bias collapsed strength (77% -> 5% vs
Stockfish-easiest); removing the positional embedding cost 17 points; halving heads
cost 19; weight-sharing was fatal; the conv stem was dead weight. Every remaining
component earns its keep. Re-litigating any of that on a ten-day timeline would be
spending days to rediscover a published result.

**What we discarded:** the reference's self-imposed *principles* (human games only,
no engine labels, no computed features, single forward pass by default). Those were
that author's experimental constraints, not rules. Their results are therefore a
floor for this architecture, not a ceiling.

---

## 2. Model size: why ~1.4M parameters and not the reference's 116k

This is the decision most worth scrutinising, so it gets the full argument.

### The challenge

The reference's 116k model measures ~1,572 Elo one-shot and ~2,474 at 300 MCTS
simulations. The strongest pure-Python engines are commonly quoted around 2,300-2,400.
So: isn't 116k already enough, and why spend anything on a bigger model?

### Three things wrong with the premise

**a) The 2,474 and the 2,300-2,400 are on different scales.** The reference's Elo is a
maximum-likelihood fit against Stockfish running `UCI_LimitStrength` at calibrated
rungs. Numbfish's ~2,300 is a Lichess bot rating. The competition ladder is a third
scale, relative to whatever the other entrants build. None of these convert to each
other reliably; Stockfish's limit-strength rungs in particular are known to be
poorly calibrated against human/online Elo. "2,474 > 2,400 therefore we win" does
not follow from these numbers.

**b) "Strongest Python engine" is the wrong ceiling.** The competition preinstalls
onnxruntime and torch. Inference runs in compiled C++ through ORT, not in the Python
interpreter. Any team can do what we are doing. The pure-Python-era strength ceiling
does not bound this field. (And the ~2,300 Numbfish figure is achieved with
Stockfish's own NNUE weights, which are banned here.)

**c) The reference's 2,474 used a value head we could not have copied - but can now
beat.** That number is for the *distilled* 116k model: a 37M-parameter teacher
trained on 72M unique positions, then distilled into the student. Distillation was
the reference's only legal route to a good value head. We have a better one: engine
evaluations.

How much that distillation was worth is *not* measured anywhere in the reference,
and this document previously implied it was. The reference's only published deltas
are h2h 0.527 against its own undistilled twin (~+19 Elo) and +6.7pp on its
Stockfish ladder. The 1,572-to-2,474 curve was measured on the distilled model
alone; no curve exists for the twin. Its README attributes the gain to the value
head, but its ablation (`eval/v3/run_distill.py`) swept only the teacher/human mix
and the softmax temperature - never policy-only against value-only - so that
attribution is an interpretation. Notably, pure teacher beat every mix with human
labels and T=3 collapsed to h2h 0.38, which is hard to explain if the teacher's
*policy* distribution were inert. See section 8.

### The actual argument for going bigger

**Measured (this machine, M1, one thread, ONNX, best-of-5):**

| Config | Params | fp32 ms | int8 ms | sims / 2.5 s (int8) |
|---|---:|---:|---:|---:|
| d24 x b8 | 72k | 1.13 | - | ~1,900 |
| d32 x b8 (the reference's) | 118k | 1.13 | 1.00 | ~1,900 |
| d48 x b8 | 248k | 1.63 | - | ~1,300 |
| d64 x b8 | 426k | 2.25 | - | ~1,000 |
| d64 x b12 | 629k | 3.37 | 2.50 | ~890 |
| d96 x b8 | 930k | 3.96 | 2.60 | ~860 |
| **d96 x b12** | **1.38M** | **5.97** | **3.77** | **~700 (768 measured in a live game)** |
| d128 x b12 | 2.44M | 9.55 | 5.74 | ~410 |
| d192 x b12 | 5.42M | 17.6 | 9.26 | ~260 |
| d256 x b20 | 15.9M | 47.0 | - | ~53 |

Two shapes in that table decide the question:

1. **Below ~400k parameters, latency is overhead-bound, not compute-bound.** d24 and
   d32 cost the same. Going from 72k to 426k parameters (6x) costs only 2x latency.
   A tiny model is *latency-inefficient*: you give up strength and get almost no
   speed back. The reference never measured this because it never deployed on a CPU
   clock.

2. **The search benefit is front-loaded.** The reference's own curve: 10 sims is
   *worse* than one-shot (1,310 vs 1,572), then a steep climb to 2,474 at 300. Beyond
   a few hundred simulations the curve flattens. So the ~1,900 sims the 116k model
   would get on our core are mostly spent on the flat part; the ~700 the 1.4M model
   gets are still on the steep part.

Putting the two together: moving from 116k to 1.4M costs about 1.4 doublings of
simulations from the flattest region of the curve (estimate: -60 to -100 Elo) and
buys 3.6 doublings of parameters (the reference's ladder implies +60-90 per doubling
at small scale: estimate +200-300 one-shot). The reference also found that "search
effectiveness scales with base model strength" - a stronger base extracts *more* per
simulation - so the parameter gain compounds under search rather than merely adding.

Above ~2.4M the trade reverses: d192 gives up simulations from the steep region
(~260) for +90-130 base, roughly a wash, and it gets worse from there. **The optimum
is a band, roughly 600k-2.4M, and inside that band the configs are within the
error bars of each other.**

### A reason that only applies to us

The reference found its 116k model **capacity-saturated on human labels**: train and
validation converged and plateaued; more data changed nothing. A saturated network
cannot absorb a richer signal. We are training on engine evaluations, which are far
richer than game outcomes. A 116k net would leave most of that signal on the table;
a 1.4M net has room for it. Our labels are the specific reason to want more capacity
than the reference needed.

### Where this argument is weak, honestly

- Every Elo figure above is an *estimate* built on the reference's measurements plus
  extrapolation. The sims curve beyond 300 is extrapolated.
- The reference's 116k hero was trained with the tau recipe (soft-policy histogram,
  averaged value) and distillation from a 37M teacher. Those are *label-quality*
  advantages, not training-time ones: a 116k run takes hours, not months (the months
  were the whole project). We match or beat it on the value side with engine labels,
  and the MultiPV policy target (section 8) closes the policy side; both are in the
  run matrix rather than in a checkpoint, so at the time of writing this is a plan,
  not a measurement.
- Engine labels would also help a 116k model.

### The decision, and the guard on it

Ship a model from the 600k-2.4M band, primary d96 x b12, **with the final choice made
by arena games between trained candidates rather than by this argument.** The run
matrix trains d64, d96 and d128 for that reason.

**Change made by this review:** the bracket should also include the reference's own
d32 x b8 as a cheap anchor (it trains in ~20 minutes on the same data). If a 116k
model with engine labels and ~1,900 simulations beats d96 with ~700 in a 400-game
arena, that is the model that ships, and the argument above was wrong at our
training budget. Cheap to test; expensive to assume.

---

## 3. Search: PUCT MCTS, not alpha-beta, not batched

**Decision: AlphaZero-style PUCT with the network's own priors and values.**

**Why not alpha-beta with a learned evaluation:** the network is the leaf cost either
way, so alpha-beta gets the same ~700 evaluations per move. At branching ~30 that is
depth 2-3 full-width, perhaps 4-5 with aggressive pruning - and it needs the policy at
interior nodes for ordering, so it saves no network calls. Minimax also backs up the
*maximum* of a noisy evaluation, amplifying value-head noise; PUCT averages it.
Alpha-beta wins when evaluation is cheap enough for 10^5-10^6 nodes (NNUE-class);
one Python core at 4 ms per evaluation is the opposite regime. Numbfish (Sunfish +
NNUE + Numba) reaches ~35k nodes per move - 50x ours - but only because its
evaluation is ~1,000x cheaper per node, and an NNUE of that quality needs billions
of training positions we do not have.

**Why not batched-leaf MCTS:** measured. Batch-8 costs 6-8x batch-1 on this CPU
(<=20% per-position amortisation, vs 5-10x on a GPU). Not worth virtual-loss
decorrelation and the complexity.

**Search-engineering decisions (all measured):**

| Choice | Instead of | Measured cost avoided |
|---|---|---|
| numpy per-edge arrays, vectorised PUCT argmax | Python loop over child dicts | 0.065 ms -> 0.018 ms per select |
| `board.copy(stack=False)` per simulation | `board.copy()` | 0.204 ms -> 0.003 ms |
| incremental zobrist-count dict for repetition | `can_claim_threefold_repetition()` per node | 0.407 ms per call at 60-ply history |
| terminal status cached on the node | recomputed per visit | n/a (correct because the tree never merges transpositions) |

The reference's `mcts.py` does all three of the expensive things; ported naively it
would spend more per simulation on bookkeeping than on the network. Total measured
overhead after fixes: ~0.23 ms per simulation, versus 3.77 ms of network.

**Implemented since (all strength-neutral-or-better by construction):**

| Change | Why it cannot hurt | Measured |
|---|---|---|
| exact evaluation cache, keyed on transposition + quantised clock planes, stackless boards only | a hit is bit-identical to the forward pass it replaces | fewer forward passes per search |
| subtree reuse across moves | inherited visits are real search work on the same positions | 63-322 visits inherited per move vs a random opponent |
| repetition checked at visit time, not cached at expansion | a reused subtree stays correct as game history grows | - |
| pondering on the opponent's clock (section 4) | uses a core that would otherwise idle; stopped before any own-move work | ~790 simulations per two-second opponent turn |
| opening pre-search during init | runs before the clock starts | 1,206 simulations banked before move one |

**Deferred (need a trained model to tune, so not guaranteed non-negative yet):**
first-play urgency reduction, c_puct re-sweep at our simulation count (the
reference's 1.5 was tuned at 50 sims on a 37M model), uncertainty-aware time
allocation, model-gated opening book, draw contempt.

---

## 4. Pondering

**Decision: search during the opponent's clock. Implemented.**

The contract states each agent keeps its dedicated core after `get_move` returns and
that pondering is allowed. Both agents run on the same machine but on separate
dedicated cores, so this uses cycles that would otherwise idle and costs the opponent
nothing.

Shape: not the UCI "predict one reply, ponderhit or restart" pattern. A background
thread keeps expanding the whole tree from the position after our move; whatever the
opponent plays, the next search starts from that child with its visits intact. No
prediction, no miss penalty. Constraints honoured: one thread only, stopped and
joined before any own-move work (a join timeout disables pondering for the game
rather than trusting a possibly-running thread), node budget of 100k so the tree
cannot approach the 2 GB limit. Combined with subtree reuse this roughly doubles
effective simulations: estimated +100-150 Elo, to be confirmed by arena once a
trained model exists.

---

## 5. Inference runtime

| Decision | Alternative | Why |
|---|---|---|
| onnxruntime | torch | measured 1.1-1.8x faster at batch 1 at every size; faster import; lower RSS |
| ship int8 **and** fp32, pick at init by timing | pick one | the competition CPU is unknown; the probe measured 3.77 vs 5.97 ms here and chose int8 |
| opset 17 export | newer | supported by any onnxruntime since 2022 |
| strict int8 gate only for trained checkpoints | always | random-init logits are near-uniform and quantisation flips their argmax freely; the >=99% bar is only meaningful once trained |
| never import torch at play time | - | the agent depends on the exported graph alone; training-environment drift cannot reach it |
| deadline checked every simulation | fixed sims | a slow core self-corrects to fewer sims instead of flagging |

---

## 6. Encoding, and the one deliberate change from the reference

**Decision: one shared featurization, defined on an int8 grid (SCALE = 50), used by
the shard writer, the trainer and the agent.**

The reference stores planes with `x.astype(np.int8)` (`src/v2/dataset.py:241`),
which truncates the fractional clock and repetition planes to integers - but its
inference `featurize()` emits the fractional values. Its trained models never see at
play time what they saw in training. That skew produces no error; it just costs
strength. Our encoding makes it impossible: `featurize_int8` is canonical, the float
the model sees is that grid divided by SCALE, and eleven parity tests pin it (mirror
equivalence, encode/decode round-trips over every legal move including
underpromotions, en passant and castling, quantisation grid, invariants).

**We then shipped the same class of bug ourselves, and the fix is the interesting
part.** Plane 16 (en passant) was set from `board.ep_square`, which a live board
populates after every double push. The referee hands `get_move` a `board.fen()`, and
python-chess omits the en passant square from a FEN unless a capture is actually
legal - so a shard built from pushed boards carried a plane-16 bit on roughly a tenth
of rows that the agent can never see at play time. Exactly the reference's failure:
no error, just strength paid away quietly. Plane 16 is now keyed on
`has_legal_en_passant()`, which agrees under every FEN convention because all of them
carry the square when a capture is legal, and a test pins
`featurize(Board(b.fen())) == featurize(b)` over every test board. Found by the
pipeline's own bit-exact round-trip assertion, which is the argument for having
written those assertions before any data existed.

Plane order, rotation convention and move encoding are otherwise identical to the
reference, so its code can be cross-checked line by line.

---

## 7. Data

| Decision | Why |
|---|---|
| Lichess Elite + Lichess monthly, CC0 | licence-clean; the reference validated the sources |
| the reference's filter recipe kept whole | validated; each filter removes a known noise source |
| **40M positions first; 100M for the final runs if the data says so** | 40M trains in hours and yields a good model, which is what the Sept 4 ladder needs. Whether a 1.4M-2.4M net is still data-limited at 40M is empirical, not settled: the reference's 116k saturated at 72M unique positions and ours is 12x larger, so it plausibly is. The per-run train-vs-val gap decides. With the build lock on Sept 11, a 100M shard (~126 GB, ~11 h per training run) is affordable for the final candidates |
| **tier mix 80/20/0 instead of 65/25/10** (fallback if 2400+ volume is short) | we want strength, not a model of human play across levels; weaker games teach moves we do not want imitated |
| **interleave SOURCES within each tier, not just tiers** | measured: Elite carries 0% `[%eval]` (it strips all annotations) and alone supplies 110.9M positions, so a shard filled tier-by-tier drains Elite first and contains zero Lichess evals - which makes assertion 6c unrunnable. Weighted round-robin over (tier x source) by filtered volume; the pattern depends only on weights and emission history, never on target size, so a 40M shard stays a byte-identical prefix of the 100M one |
| 8M smoke shard first | surfaces a pipeline bug in one hour instead of after a 40M run |
| validation split **by game** | positions within a game are correlated; a by-position split leaks |
| `Z.bin` zobrist column | lets labels be joined later without regenerating 54 GB |
| FEN sidecar | lets Stockfish label positions without re-parsing archives |
| filtered PGNs kept | input to the shard, the tau corpus, and any future label join |
| shard generated by importing `chessml.encoding` | the single hard rule; see section 6 |

**Measured yields** (Sept 2): Elite, six months, 1.31M games kept of 1.71M seen,
110.9M positions, `[%eval]` yield 0.00% in both tiers. Lichess monthly 2026-07,
100M games seen: top tier 281k games / 22.9M positions / 42.2% eval yield; mid tier
7.11M games / 495.5M positions / 15.2%. Interleaved 8M shard: 80.0/20.0 tier split,
~7.8% of rows carrying a Lichess eval. Elite's 0% is the fact that makes Part C the
whole label supply rather than a top-up.

**Measured on the box** (Sept 2, 16 physical cores / 32 threads, Ryzen 9 7950X;
`/workspace` is MooseFS over FUSE): filter 20k games/s; shard build 17.8k rows/s;
Stockfish labelling **509 positions/s at 25k nodes, MultiPV=4, 16 workers**. The 8M
and 40M shards both land at exactly 80.0/20.0, Part C reached 100% coverage on 8M,
and assertion 8 is verified - all 8,000,000 Z of the 8M shard match the 40M prefix.

That 509/s is **below the 700/s floor** the labelling amendment set for the 40M pass,
and the floor assumed 32 physical cores rather than 16 with SMT. The fallback is a
cheaper per-position search (fewer nodes, or a depth limit), never fewer MultiPV
lines. Which config the 40M pass uses is decided by the sweep in
`pipeline/sweep_labels.py`, not by this document.

**Assertions the pipeline must pass** (a predecessor project lost years to these):
position stored before the move; mirror correctness incl. no pawns on back ranks;
bit-exact round-trip against `featurize_int8`; value sign on a constructed mate;
determinism; engine-value sign and correlation; a hand-verified `[%eval]` attachment
check (the correlation test cannot detect a one-ply shift); invariants over 100k rows.

---

## 8. Labels: where the Elo actually is

**Decision: engine evaluations as the value target, on every position.**

Human game outcomes are a poor label for position quality: a 2400 player loses won
positions constantly, so "White eventually won" says little about whether White was
winning *here*. MCTS leans on the value head to decide what to explore, so value
quality converts directly into search strength. This is the reference's largest
weakness (its own words: the value head is "signal-limited") and the organisers have
confirmed engine-derived training labels are permitted.

Three sources, in order of cost:

1. **Lichess `[%eval]` annotations** - free, already in the archives, on a fraction of
   games. The reference strips them; we keep them. Conversion uses Lichess's own
   centipawn-to-win-probability mapping (`2/(1+exp(-0.00368208 cp)) - 1`), mates
   clamped to +/-0.995, perspective via python-chess `score.pov()`. The eval on move
   m evaluates the position *after* m, so a stored position's value comes from the
   preceding node - an off-by-one that no statistical test catches, hence the
   hand-verified assertion.
2. **Stockfish self-labelling on the rented box's CPUs, MultiPV=4** - fills coverage
   to ~100% and is the whole label supply in practice, since Elite measured 0%
   `[%eval]`. 25k nodes per position with four lines (~the depth 10k single-PV gave
   the top line), one engine per core, deduplicated by zobrist first (~30-40% fewer
   positions), joined back by `Z.bin`. Stored per position: the four moves, their
   win-probabilities, and their depths. Column 0 is the scalar value target, so
   everything reading `Y_value_engine` is unchanged. Roughly 2x the single-PV cost,
   on CPU, parallel to GPU training. **The fourth line is not a luxury: it is the
   soft policy target** (section 8, "what replaces distillation"). If throughput
   forces a cut, cut nodes, never MultiPV - the extra lines cannot be recovered
   without relabelling.
3. **The tau recipe** (stretch) - aggregate by unique position: mean outcome, human
   move histogram as a soft policy target, count^0.5 sampling. The reference measured
   **+79 Elo at identical parameters** from this alone.

### Why not distillation, and what replaces it

The first version of this section said engine evaluations "supersede" distillation.
That was half right, and the half that was wrong is the more important half.

**Distillation replaced two targets; engine evals replace one.** The reference's
teacher labelled every position with top-32 policy logits *and* a value
(`src/v3/teacher_label.py`). A Stockfish eval is a scalar. It dominates the value
half - a near-zero-variance estimate from a ~3,500-strength engine against a
denoised human-outcome estimate from a ~2,600 teacher - and replaces nothing on the
policy side. Our policy target would have stayed human one-hot, with the tau
histogram collapsing to one-hot outside the opening anyway (the reference's 100M
rows deduplicated to 72M unique, so most positions occur once).

**MultiPV narrows that gap from a stronger source. It does not close it, and it is
not equivalent to the teacher's policy.** Four lines with win-probabilities,
softmaxed, are a soft policy distribution produced during a labelling pass we were
running anyway. Three ways it differs from top-32 teacher logits, all of them
against us except the first:

- *Source strength:* Stockfish at 25k nodes against a ~2,600-strength imitation
  model. The one unambiguous improvement.
- *Breadth:* four lines against thirty-two. Everything outside the top four takes
  zero mass, where the teacher ranked down to 32. The student's softmax will not
  learn literal zeros, but the tail shape is gone. This is one reason R8b keeps a
  human component (alpha 0.5) rather than going pure-teacher as the reference did.
- *Shape:* the teacher encoded what a strong imitator would plausibly play; Stockfish
  encodes objective quality, and in forcing positions collapses toward one-hot -
  which is the target we already had. The gain is concentrated in positions where
  several moves are close.

Run R8 tests it. Until then this is the least-evidenced claim in this document, and
"our policy target now matches the 37M teacher" is not a claim it supports.

**What engine evals structurally cannot give, and we accept:** the value head should
estimate the outcome when *our* net at ~700 sims plays on; Stockfish estimates it
under near-perfect play. `2/(1+exp(-0.00368208 cp))-1` is a function of cp alone, so
a +3.0 needing six only-moves and a +3.0 that is trivial technique receive identical
targets. It cannot express sharpness; a teacher's value head, being a function of
the position, can. The error points the wrong way for a searching agent -
overconfidence steers PUCT into lines it cannot hold. Mitigation is the outcome
blend (R7), not a fix.

**Cost, corrected.** The "~3 sequential GPU-days" previously cited here is the
reference's from-scratch 37M teacher, not our cost: the size bracket already trains
d128, and the top-32 format is ~2.5 GB at 40M rows, so a student run is hours. The
honest reason to skip it is that a 4x teacher-student gap buys far less than the
reference's 320x - not cost, and not supersession.

**Open experiments:** pure engine value vs a 0.5/0.5 blend with the averaged outcome
(R6 vs R7 - 0.75/0.25 would sit too close to R6 to discriminate); human policy vs
the MultiPV soft target (R8).

---

## 9. Training

| Decision | Why |
|---|---|
| loss = policy CE + value_weight x value MSE | the reference's recipe |
| policy CE over all 4,672 logits, **unmasked at training time** | masking to legal moves removes the gradient that teaches the net which moves are plausible at all; the mask belongs at inference, where `chessml.search` already applies it. Note the soft-policy runs (R8, R9) do mask, since their target is a distribution over legal moves - that asymmetry is deliberate and worth re-testing if R8 disappoints |
| loader holds the split in RAM, not memmap | measured: `/workspace` is MooseFS over FUSE at ~15 ms per random read - validating 8M rows took 8m25s wall for 13s of CPU. 124 GB of RAM covers the 8M and 40M shards outright; 100M still needs the bit-packed path |
| **value_weight tested at 1.0 and 2.5** | the reference tuned 1.0 for single-pass play; we are search-first and the value head is what search amplifies |
| value target = engine where present, else outcome (masked) | never drop rows lacking evals; their moves still train the policy |
| AdamW 1e-3, wd 1e-4, batch 1024, BF16, clip 1.0, cosine to zero | the reference's validated settings; BF16 because FP16 silently NaNs deep towers |
| cosine horizon = epochs actually run | a longer horizon leaves the model at high LR and measurably worse |
| 8 epochs over 40M | enough for saturation at this size; the train/val gap reported per run says whether to extend |
| EMA of weights, decay 0.999 | near-free +10-30 in supervised training |
| checkpoint every epoch, raw and EMA, selection by *play* | final epoch is not automatically best under search; accuracy is a policy metric and the value head is what matters |
| val accuracy tracked **split by colour** | a collapse on one colour is a mirror bug, not a training problem |
| loader: memmap + block shuffle, else bit-pack to ~101 B/row in RAM | the reference stalled at 2k samples/s on disk I/O; at that rate 8 epochs is 44 hours |
| run matrix R0 smoke -> R1 partial labels -> R2-R5 full labels | earlier runs are insurance for later ones; a model on Sept 3 beats a better one on Sept 8 |

---

## 10. Agent runtime and reliability

Every failure mode is a full loss, so the runtime is built around not losing games
for free. In a Swiss of hobby agents this is worth an estimated +100-200 Elo
equivalent over a same-strength fragile entry.

| Decision | Why |
|---|---|
| `get_move` cannot raise; any exception returns a legal move and drops state | crash = loss |
| history reconstructed by matching the opponent's reply among legal moves; rebase from FEN if nothing matches | the contract supplies only a FEN; the repetition plane and draw logic need history |
| repetition/fifty-move tracked with a zobrist-count dict | the referee auto-claims; the agent must see it coming |
| veto moves that hand the referee a claim when winning (q > +0.3); seek them when losing (q < -0.3) | a claimed draw is a free half point in a lost position and a thrown win in a won one |
| past ply 240, material balance drives that logic | ply 300 is adjudicated on material |
| budget = clamp(left / max(14, 46 - move) + 0.4 s, <= 4 s, <= left - 1 s) | flagging is the most common self-inflicted loss |
| < 2 s: one forward pass; < 0.25 s: first legal move | degraded modes instead of a flag |
| repo gate: ruff, mypy strict, 40 tests, two clean fast games | the starter's own bar. On random weights the agent loses those games; the gate checks legality and the clock, not strength |

Verified: a full game at real time control and the gate's fast games completed with
zero crashes, illegal moves or flags on random weights; search finds mate-in-one for
both colours from the rules alone.

---

## 11. Things rejected, and why

| Rejected | Reason |
|---|---|
| bigger models (>2.4M) | section 2 |
| WDL (3-logit) value head | touches model, export, net and search for a refinement (draw awareness); engine values are the big win and work with the scalar head |
| batched-leaf MCTS | measured <=20% amortisation on CPU |
| Numba for the tree loop | overhead is ~5% of simulation cost after the fixes in section 3 |
| distillation | section 8 - on the timeline and the teacher gap available to us, not because engine labels supersede it; MultiPV supplies the policy half it would have brought |
| Numbfish as a reference | ships Stockfish's NNUE weights: disqualifying on two counts; the architecture without them needs billions of positions |
| architecture changes | the reference's ablation already settled them |
| 100M positions | section 7 |
| committing weights to git | build artifacts; every retrain would add megabytes to history permanently; `make weights` regenerates |

---

## 12. Process and timeline

| Decision | Why |
|---|---|
| skeleton built and gated before any data exists | correctness of encoding, search and runtime is testable without a trained model, and those bugs are the silent ones |
| **upload a valid entry before the ladder opens Sept 4** | the platform's validation log is the only authority on acceptance; the ladder seeds the Swiss; the ladder is the only measurement against the real field; earlier submission is a tie-break |
| final candidate checkpoint by Sept 9; the 11th is the lock, not the target | two days of integration, tuning and ladder validation; a checkpoint arriving on the 11th has zero games behind it and any bug in it is fatal |
| rented compute (RunPod-class), 64 GB+ RAM, many vCPUs, modest GPU | data and labelling are CPU/RAM bound; a 1.4M model cannot saturate a large GPU; the data never has to move - only a <30 MB checkpoint comes back |
| training-side code in `train/`, own dependency group, outside mypy | torch never enters the submission environment |
| five commits by layer, each independently green | each boundary is a place the gate passes |
| no comments in code; rationale lives in commit messages and this file | the starter's 434 lines carry 2 comment lines; the shipped `agent.py` stays plain for a judge |
| four stop-and-report points in the friend's brief | training on subtly wrong data burns GPU hours producing a model to throw away |

---

## 13. Uncertainties, and the experiment that resolves each

| Uncertainty | Resolves it |
|---|---|
| where in 116k-2.4M the optimum actually sits at our training budget | the size bracket incl. the d32 anchor, 400-game arena |
| whether 40M positions is enough for the larger models | train-vs-val gap on R2; if validation is still climbing at epoch 8, extend to 100M for the final runs |
| pure engine value vs blend with outcome | R6 vs R7 (0.5/0.5) |
| whether the MultiPV soft policy target beats human one-hot | R8; this is the distillation replacement and the least-evidenced claim in this document |
| whether Stockfish's MultiPV line order can be trusted | measured: it disagrees with the scores it reports on 18.5% of positions, because a node-capped search leaves the four lines at different depths. Sorting by stored value makes the invariant hold by construction, at the cost of making the value target the max of four noisy scores; a depth-limited search removes the cause instead, if throughput allows |
| value_weight 1.0 vs 2.5 | R2 vs R3 |
| int8 accuracy on the trained model | the export parity gate. Measured on the reference's trained 116k hero: **0.863 argmax agreement, well under the 0.99 bar**. Small models may simply not survive dynamic quantisation, in which case fp32 ships and the sims-per-second table's int8 column does not apply |
| competition core speed | the init-time probe, per game |
| whether the platform's cores are truly dedicated for pondering | simulation count during the opponent's turn, in the validation log |
| how strong the field is | the ladder from Sept 4; nothing else measures it |
| whether we are actually at least as good as the reference | head-to-head against `baselines/reference-hero` - the reference's own published 116k hero, converted to our checkpoint format by `train/import_reference_hero.py` and run through our search and runtime unchanged. Model-only (equal sims) isolates training; full-agent at the real clock is the answer to the question as asked. Its capsule was trained under the pre-fix en passant convention, so it sees a slightly different input on ~10% of rows; acceptable for a baseline |

Expected strength on the reference's Stockfish-anchored scale: ~2,200-2,500 for the
first working model, 2,600-2,800 if engine labels, tau and the search work all land.
Those are estimates with +/-300 error bars, and they do not convert to ladder Elo.
