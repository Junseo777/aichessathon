# Rated games

PGN exports from the platform's rated ladder, one file per round, kept exactly as
downloaded. The platform anonymises the headers, starts each game from a curated
opening position (the `FEN` tag, move 6 to 9), and stamps every move with the
clock after it. Time control 120 s + 0.5 s.

## Which side is ours

The headers do not say. The clocks do: our spend per move matches `_budget_s` in
`agent.py` to within 10 ms in every game, the opponent's is off by seconds.

What played is the uploaded `submission_R1_e8_ema.zip` (built Sept 4, 15:29):
its `_budget_s`, `_hands_over_draw_claim` and `_pick` thresholds are identical
to the working tree, but its search is the Sept 1 PUCT with the FPU fix and
none of the Sept 4 features (proofs, smart pruning, scaled FPU, policy
temperature, in-tree draw score). The games agree: not one of the 373 searched
moves stopped before its deadline.

| Round | Opponent | Ours | Result | Termination | Stockfish, our view, at the end or at the slip |
|---|---|---|---|---|---|
| 1 | keep-kann-and-caro-on | White | draw | threefold | +0.6, after 38. Nh4 gave back +1.8 (Ne1 kept it) |
| 2 | bishop | Black | win | mate | opponent walked its king into mate from −2.3 |
| 3 | hyperfish | White | win | mate | opponent moved instantly all game |
| 4 | tobias-carlsen | Black | win | mate | clean, opponent erred at 19. Bc6 and 28. Bg1 |
| 5 | omega3-fish | Black | draw | insufficient material | +3.9 at move 49, 0.0 by move 55 |
| 6 | ai-fellows | White | win | mate | clean, 42...Rb5 allowed Rh8# |
| 7 | asher-falcon | White | loss | mate | −0.2 at move 59, −3.0 after 60. Rg3, 80. Kg4 into mate |
| 8 | fuzzybot | Black | draw | threefold | +2.2 for us when the draw was declared |

Score 5.5/8. The four wins were all against opponents who blundered into mate.

## What the games show

**The platform's repetition rule is not the one the agent models.** The referee
declares a draw as soon as the side to move *could* complete a threefold, the
`board.outcome(claim_draw=True)` semantics of `harness/referee.py`. Both drawn
final positions prove it: round 8's had occurred twice and round 1's once, and
both were terminated as `threefold_repetition`. `_hands_over_draw_claim` only
flags a move that makes a *third* occurrence. In round 8 the decision point was
42...Kf8, a second occurrence: the current search rates it q +0.45, above the
+0.30 winning threshold, so the avoidance branch would have fired had the move
been flagged. Rd1+ was the fresh alternative with q +0.47 and is Stockfish's
choice. Two plies later fuzzybot's rook shuffle put us in a position from which
we could repeat, and the referee claimed on our behalf. By move 43 every legal
move was flagged under the real rule. Fix: when winning, avoid *second*
occurrences; when seeking a draw while losing, use the referee's can-claim
check. Round 1 is milder: 43. Rc4 handed the draw at q 0.28 to 0.29, just under
the threshold, in a position Stockfish calls +0.6.

**Time management plays every game down to two seconds.** The burn-down is the
same in all eight games, whatever the position or the opponent's clock:

| our clock after move | 20 | 30 | 40 | 50 | 80 |
|---|---|---|---|---|---|
| seconds | 78 | 47 | 23 | 12 | 2 |

After move 32 the divisor floors at 14 and spend decays geometrically. In the
two long games (rounds 5 and 7) the last 25 to 30 moves were played at 0.6 s
each. Every error above 12% win probability in the whole set came on a move made
with under a second: three in a row in round 5 (moves 50 to 52, where b6 or Rh3
kept +3.8) and four in round 7. The opponents in those games were at 2 to 5 s
from move 40 and held anyway.

**Queen versus rook is a net weakness, not only a clock problem.** At round 7
move 60 the current search still picks Rg3 with 6 s (q −0.32); Stockfish keeps
Rb3 or Rd3 near −0.2. asher-falcon converted the ending at 0.5 s per move, which
marks it as a fast classical engine and the strongest opponent here.

**The opening pre-search never fires in rated play.** Games start at move 6 to
9; `_adopt_opening` matches the start position or one ply from it. Harmless,
since it runs in the init budget, but dead weight.

**Opponents.** hyperfish moves in 0.06 s and is free points. bishop and
tobias-carlsen use fixed-time moves and collapsed tactically. omega3-fish and
asher-falcon burn their clock to nothing by move 45 and are still hard to beat
on the increment. fuzzybot keeps 35 s in hand and drew from −2.2.

## What the clocks say about the platform's core

The rules state only "1 dedicated CPU core, 2 GB RAM, identical hardware". The
clock stamps give an indirect probe: the search loop checks its deadline once
per simulation, so each move overruns the budget by a fixed harness overhead
plus the remainder of one in-flight simulation. Over the 373 searched moves the
overrun runs from 2.8 ms (5th percentile) to 12.2 ms (95th), never negative, so
a full simulation takes roughly 10 ms on the platform. Idle reference points:
the M1 at 3.6 to 6.0 ms per forward, about 7 to 10 ms per simulation; the box
at 2.3 ms per forward, about 5 to 6 ms, though its timings swing by 60% between
runs. The platform therefore looks like the Mac, if anything a little slower,
and not like the box. The direct number is in the validation log of the upload:
`load_fastest` prints `forward_ms` measured at import, and the init pre-search
prints its simulation count for 5 s.

## Method

Every position evaluated with Stockfish at 0.25 s and 4 threads for the
trajectory and per-move loss; 4 s multipv on the positions named above. The
agent's own verdicts come from replaying those positions through the working
tree's `chessml` search, which carries the Sept 4 features the upload lacks,
with the budget it actually had in the game and again with 6 s, on a laptop
that was running sparring lanes at the time, so simulation counts are neither
the platform's nor idle.
