# Search cannot explore low-prior moves in winning positions

Found 2026-09-03 while installing R2's weights into `weights/`, which un-skipped
`chessml/tests/test_search.py`. Both mate-in-one tests fail.

This is a search issue, not a weights issue. It was filed rather than fixed
because `chessml/search.py` is the shipped agent's core and search is Junseo's
to change. **Fixed 2026-09-04; see the resolution at the end.**

## Symptom

```
test_finds_mate_in_one          best=f2f4  want=a1a8   FAIL
test_finds_mate_in_one_as_black best=f7f5  want=a8a1   FAIL
```

Position `6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1`. `Ra8#` is mate in one. With 400
simulations the model plays a pawn move instead, and **the mate receives zero
visits**. Not a near miss - it is never tried once.

## Cause

`chessml/search.py:58`:

```python
q = np.divide(node.w, node.n, out=np.zeros_like(node.w), where=node.n > 0)
```

Unvisited children are scored `Q = 0`. There is no First Play Urgency. So an
unvisited move is selected only when its exploration bonus alone beats an
explored move's `Q + U`.

At this root, `c_puct = 1.5`, `N = 400`, root value `+0.796`:

| move | visits | Q | U | Q+U |
|------|--------|-------|-------|-------|
| f2f4 | 99 | 0.792 | 0.067 | **0.859** |
| a1a4 | 65 | 0.785 | 0.074 | 0.859 |
| a1a8 (mate) | 0 | **0.000** | 0.481 | 0.481 |

The measured policy priors put the mate 13th of 20 at `P = 0.016`.

Generally, an unvisited move is reachable only when its exploration bonus alone
clears the best explored move's whole `Q + U`, not its `Q` alone:

```
P > (Q_explored + U_explored) / (c_puct * sqrt(N))
```

which here is `0.859 / (1.5 * 20) ~= 0.029`. At 400 simulations every move with a
prior under ~2.9% is excluded by the arithmetic, not merely unlikely.

That exclusion is real but budget-dependent, because the threshold falls as
`1/sqrt(N)`. Measured on this position, everything else held fixed:

| simulations | mate visits | move played |
|---|---|---|
| 400 | 0 | f2f4 |
| 800 | 0 | f2f4 |
| 1,600 | 406 | a1a8 |
| 3,200 | 2,001 | a1a8 |

The formula puts the crossover near 1,300 simulations; measured, it falls between
800 and 1,600. This changes how the finding should be tested, not whether it
bites: the shipped fp32 agent gets roughly 400-500 simulations per move, well inside
the excluded regime. Anyone re-testing at a few thousand simulations will not
reproduce it, and should not conclude from that that it is not there.

The consequence is backwards from what you want: **the better the position
looks, the less the search explores.** At `Q ~= 0` the threshold is near zero
and everything gets tried; at `Q ~= 0.8`, at the budget we ship, a quarter of
the move list becomes invisible. The agent stops looking for wins exactly when
it is winning.

## Fix, and evidence it works

Give unvisited children the parent's value, minus a reduction, instead of zero:

```python
fpu = node.value - 0.25
q = np.divide(node.w, node.n, out=np.full_like(node.w, fpu), where=node.n > 0)
```

Measured on the same two positions, same weights, same 400 simulations:

```
BASELINE (unvisited Q = 0)
  best=f2f4  want=a1a8  mate_visits=  0  FAIL
  best=f7f5  want=a8a1  mate_visits=  0  FAIL

WITH FPU (unvisited Q = parent value - 0.25)
  best=a1a8  want=a1a8  mate_visits=221  PASS
  best=a8a1  want=a8a1  mate_visits=218  PASS
```

0.25 is a starting point, not a tuned value; Lc0 uses roughly this magnitude for
absolute FPU. It should be swept in the arena rather than taken on faith. The
direction matters: a reduction that is too *small* makes the search wander
(unvisited moves look as good as the parent), and one that is too *large*
makes it narrow (they look bad, so the visited moves keep the visits).

## Why this probably matters more than the tests suggest

The tests use a mate, because a mate is unambiguous. The mechanism is not
specific to mates - it applies to any good move the policy head ranks low, in
any position the value head likes. That is the conversion problem: finding the
one strong move in a won position is precisely the case where the policy is
least reliable and the value is highest.

It may also be a second contributor to the 60% threefold-repetition rate in R0's
arena, alongside the `q_best < -0.3` draw-seeking in `agent.py:200-208`. Shuffling
in a won position is what this bug looks like from the outside.

## Caveat on the run matrix

Every arena result so far - R0's 40% against the reference hero, and any bracket
comparison run before this is fixed - was measured with this in effect. It
affects all candidates equally, so relative rankings are probably still
informative, but absolute strength is understated and models whose policy is
sharper will have been flattered relative to models with a better value head.
Worth re-running the decisive comparisons afterwards.

## Resolution

Fixed in `chessml/search.py` on 2026-09-04. `MCTS` takes `fpu_reduction`
(default 0.25) and `_select` scores an unvisited child as

```python
running = (node.value + node.w.sum()) / (1 + node.total)
fpu = running - fpu_reduction
```

One change from the patch proposed above: the anchor is the node's *running*
mean, not `node.value`. The static net value is only the first term of that
mean, so the two agree on a fresh node and diverge as search learns more. Had
the anchor stayed at `node.value`, a node whose Q climbed well above the net's
guess would have reproduced the original collapse for its unvisited children.

Both mate-in-one tests pass at 400 simulations. `make test` and `make gate`
now run them; CI skips them because its export is a random-init net (see
`chessml/tests/weights_fixture.py`).

Still open: 0.25 is untuned. Sweep it in the arena before trusting it, and
re-run the decisive comparisons — every result before this date was measured
with the collapse in effect.

Same-net sanity check, 2026-09-04: R2 with this fix against R2 without it, both
sides otherwise identical (`sparring/agent-r2-new` vs `sparring/agent-r2-old`,
the latter frozen at `a66083f`), 30 s + 0.3 s, standard start, two lanes of
eight games in parallel, colours alternating (`sparring/arena_fpu_cache`).

| | games | W-D-L | score |
|---|---|---|---|
| fix as White | 8 | +3 =5 -0 | 68.8% |
| fix as Black | 8 | +4 =3 -1 | 68.8% |
| **total** | **16** | **+7 =8 -1** | **68.8%** (95% 54-83%) |

Implied +137 Elo for the fix (95% +28 to +281); decisive games 7-1, two-sided
binomial p = 0.07. Terminations: 8 checkmates, 8 threefold repetitions. Lane 2
won its first four games outright and then drew four; lane 1's only loss was
game 8 with the fix as Black.

Exploration profile, measured offline on six positions at 150 and 400 simulations
(`scratchpad/probe.py`, old vs new search, same R2 net). The old rule was not
uniformly narrow: scoring an unvisited child 0 is a reduction equal to the
parent's Q, so it was near zero in level positions and ~0.8 in won ones. The
constant 0.25 is therefore *more* cautious than before at Q ~ 0 and far less at
Q ~ 0.8. In the four level positions (start, Italian, a middlegame, a K+P ending)
the new search leaves one to three more root moves unvisited and the share of
interior nodes that ever visited only one child rises from 17-20% to 25-29%. In
the won positions it opens up: the mate-in-one goes from 0 visits to 48% of the
root's, and interior single-child nodes fall from 25% to 7%. So the arena's
seven-to-one in decisive games is the won-position half of that trade, and the
level-position half (slightly narrower search) is what the sanity arena cannot
see and the full head-to-head must: any harm will show as losses from missed
defensive resources in equal middlegames, not as failed conversions.

What it does and does not show. Same net on both sides, so every point of the
difference is search. At this clock each side gets roughly 150 simulations per
move, a third of the shipped budget, which makes the collapse *worse* than in
rated play (the threshold falls as 1/sqrt(N)) and the cache matter *less*
(short searches transpose less); the direction is right, the size is not
transferable. Seven of the eight decisive games were won by the fix, which is
the mechanism at work: finding the low-prior winning move in a position the
value head already likes is exactly the conversion problem `ARENA2` section 7
names. Half the games were still threefold draws, so the fix does not remove
the draw wall; both sides share the draw-seeking rule and the same net. Not
significant at 5%, no PGNs kept, both lanes shared the machine (contention is
symmetric under a wall-clock budget). The measurement that would settle it: the
same pairing at 120 s + 0.5 s with PGNs, and a sweep of `fpu_reduction` over
{0.15, 0.25, 0.4}, plus one arm with Lc0's scaling (reduction times the square
root of the visited policy mass), which is near zero while a node is fresh and
grows as its policy is explored, so it covers both regimes with one constant.
