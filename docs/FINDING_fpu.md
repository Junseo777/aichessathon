# Search cannot explore low-prior moves in winning positions

Found 2026-09-03 while installing R2's weights into `weights/`, which un-skipped
`chessml/tests/test_search.py`. Both mate-in-one tests fail.

This is a search issue, not a weights issue. It is filed rather than fixed
because `chessml/search.py` is the shipped agent's core and search is Junseo's
to change.

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
bites: the shipped agent gets roughly 700 evaluations per move, well inside the
excluded regime. Anyone re-testing at a few thousand simulations will not
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
absolute FPU. It should be swept in the arena rather than taken on faith, and a
reduction that is too large will make the search wander.

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
