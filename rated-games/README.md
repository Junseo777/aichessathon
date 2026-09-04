# Rated games

PGN exports from the platform's rated ladder, one file per round, kept exactly as
downloaded. The platform anonymises the headers, starts each game from a curated
opening position (the `FEN` tag, move 6 to 9), and stamps every move with the
clock after it. Time control 120 s + 0.5 s.

## Which side is ours

The headers do not say. The clocks do: our spend per move matches `_budget_s` in
`agent.py` to within 10 ms in every game, the opponent's is off by seconds. From
round 15 the pruning build spends under its budget, so there the fingerprint is
the side that never exceeds `_budget_s` and stops most of its searches early.

Which upload played which round (Junseo, Sept 4 evening): rounds 1 to 8 the
R3_e8_ema upload of 02:56; rounds 9 to 14 R1_e8_ema on the same code; round 15
onwards R1 with smart pruning. The clocks agree: not one of the 722 searched
moves in rounds 1 to 14 stopped before its deadline, and in round 15 the bot
stopped 62 of 63 searches early and spent 47% of its budget. The code before
pruning was `main` at f2a78b8: its `_budget_s`, `_hands_over_draw_claim` and
`_pick` thresholds are identical to the working tree, but its search is the
Sept 1 PUCT with the FPU fix and none of the Sept 4 features (proofs, smart
pruning, scaled FPU, policy temperature, in-tree draw score).

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
| 9 | blunder-buss | White | draw | threefold | +4.3 at move 59, +0.3 after 60. Qe7 and 61. c7 |
| 10 | meshpotato | Black | loss | mate | +1.5 after 14. Bxa7, gone after 14...Qa5; 30...Nd5 lost the ending |
| 11 | im-master | Black | win | mate | +4 by move 35; 35...Bxe5 gave a third back, the opponent's 35. fxe5 and 36. Kf2 gave it all back |
| 12 | zak | White | win | mate | +6 by move 40 and converted, with 4.1 s left at move 68 |
| 13 | alien-gambit | Black | win | mate | declined a mate in three at move 44 for a slower win; not provable within 800 simulations, so proofs would not have played it either |
| 14 | pgn | Black | draw | stalemate | +4.3 at move 44; 45...fxg2 at 1.6 s with 15.6 s left walked into a queen sacrifice; bare king by move 64, the opponent stalemated us at 111 |
| 15 | 50centraise | White | win | mate | first game of the pruning build: worst move 5.8%, eval rising throughout, 36 s left at the end |

Score 10.5/15. The wins were all against opponents who blundered, seven of the
eight into mate. Rounds 11 to 15 were downloaded at 22:18 and analysed the same
way as the first ten.

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

**The opening is not where the games go, either.** Our first five moves from
the curated start, scored against Stockfish at depth 16 in eight of the fifteen
games (rounds 1, 2 and 10 to 15): mean loss 6 cp per move, and 2 of the 40
moves lost 30 cp or more, both in round 2, which we won. A book cannot improve
on that; it can only return the same moves without the 3.5 s the budget formula
spends on each, 7 to 10 s a game if the opponent stays in book for two or three
moves. The curated pool is finite and public: our round 15 start (French
Tarrasch, ply 9) is the position SoberJackson played in its round 9, one exact
repeat among 30 sampled starts, so a pool of a few hundred positions, and every
public game page carries the FEN. The verdict and what would change it are in
`docs/DECISIONS.md` sections 11 and 13.

**Opponents.** hyperfish moves in 0.06 s and is free points. bishop and
tobias-carlsen use fixed-time moves and collapsed tactically. omega3-fish and
asher-falcon burn their clock to nothing by move 45 and are still hard to beat
on the increment. fuzzybot keeps 35 s in hand and drew from −2.2.

**Round 9 is the conversion failure again, round 10 is not.** In round 9 we
held +3.6 to +4.4 from move 45 with queen and three pawns against rook, bishop
and three pawns, then 59. e5 and 60. Qe7 at 0.9 s per move with 6 s left let
the bishop and rook trade into a dead queen-versus-rook-and-pawns draw; the
winning 60. Qe3 was not among the search's top five candidates even with 6 s.
Round 10 was lost at full budget: 14...Qa5 (94 s in hand, Stockfish wanted b6,
+1.5) and 30...Nd5 (46 s in hand, Nd7 held at 0.0) were each searched for the
formula's 3.4 s. With 6 s the working-tree search finds both b6 and Nd7. Those
are the first errors in the set that time alone would have fixed.

**Rounds 11 to 15 repeat the pattern and show the pruning build's clock.** Round
14 is the clearest throw in the set: +4.3 at move 44, then 45...fxg2 at 1.6 s with
15.6 s left, into 46. Qxf6+ and a lost rook ending. Replaying that position with
the step-1 search picks fxg2 up to 200 simulations and abandons it at 250, and
proofs make no difference at any budget, since the refutation is a material win,
not a mate. Round 15, the first ladder game with pruning, reached move 45 with
76 s against 16 s in every pre-pruning game, so the same position would have
had the full 4 s budget. Round 13 is the test case for proofs: at move 44 the bot
declined a mate in three for a slower win, and with proofs on the replay does
not prove it within 800 simulations, so proofs would not have played it either;
it won anyway.

**A signature worth acting on.** In every replayed decisive error the
most-visited move had a lower q than a less-visited rival. Scoring each pick
against that rival with Stockfish at 3 s:

| position | budget | most-visited | cp | higher-q rival | cp |
|---|---|---|---|---|---|
| R10 14...  | 3.4 s | Qa5 | −27 | b6 | +104 |
| R10 30...  | 3.5 s | Nd5 | −224 | Nd7 | −25 |
| R7 60.     | 0.9 s | Rg3 | −508 | Rf4 | 0 |
| R7 60.     | 6 s   | Rg3 | −541 | Rd3 | −21 |
| R5 50...   | 1.3 s | h5 | +295 | Rh3 | +422 |
| R5 50...   | 6 s   | c4 | +413 | Rh3 | +439 |
| R8 42...   | 2.0 s | Kf8 | +242 | Rd1+ | +269 |
| R9 60.     | 0.9 s | Qe7 | +3 | Qa7 | 0 |
| R1 43.     | 1.9 s | f3 | +64 | Rc4 | 0 (the repetition) |

Five for the rival, three ties, one for the pick. Picking by raw q is not the
answer, since low-visit q is noise, but the disagreement itself is a usable
signal: extend the search while the best-visited and best-q moves differ, or
select by a lower confidence bound on q among moves with a share of the
visits, as KataGo does. Both are cheap to arena.

## What the clocks say about the platform's core

The rules state only "1 dedicated CPU core, 2 GB RAM, identical hardware". The
clock stamps give an indirect probe: the search loop checks its deadline once
per simulation, so each move overruns the budget by a fixed harness overhead
plus the remainder of one in-flight simulation. Over the 373 searched moves of
rounds 1 to 8 the overrun runs from 2.8 ms (5th percentile) to 12.2 ms (95th),
never negative, so a full simulation takes roughly 10 ms on the platform; the
107 moves of rounds 9 and 10 give 2.8 to 12.6 ms, the same. It varies by game,
7 to 11 ms, so the hosts are not equally loaded. Idle reference points:
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
the platform's nor idle. The first-five-move losses were scored on 2026-09-04
with Stockfish at depth 16 on one thread, our side identified by the clock
fingerprint above. The lichess opening explorer refused unauthenticated
requests that day, so how deep a human-games book would follow the curated
starts is unmeasured.
