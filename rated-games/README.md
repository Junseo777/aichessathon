# Rated games

PGN exports from the platform's rated ladder, one file per round, kept exactly as
downloaded. The headers carry the team names (ours is Team1, as the public leaderboard
lists it); the platform starts each game from a curated
opening position (the `FEN` tag, move 6 to 9), and stamps every move with the
clock after it. Time control 120 s + 0.5 s.

## Which side is ours

The headers do not say. The clocks do: our spend per move matches `_budget_s` in
`agent.py` to within 10 ms in every game, the opponent's is off by seconds. From
round 15 the pruning build spends under its budget, so there the fingerprint is
the side that never exceeds `_budget_s` and stops most of its searches early.

Which upload played which round (Junseo, Sept 4 evening): rounds 1 to 8 the
R3_e8_ema upload of 02:56; rounds 9 to 14 R1_e8_ema on the same code; round 15
onwards R1 with smart pruning; rounds 16 to 23 the 03:26 upload of Sept 5
(temperature 1.359, root FPU 1.0, pruning, the old repetition test); rounds 24 to
30 `submission_fixes_R1_e8_ema_fe58a9c.zip`, built 15:31 and byte-identical to
`main` at fe58a9c (repetition fix, opening-tree adoption, draw rule on, extension
and LCB off); rounds 31 onwards `submission_R8_int8.zip`, built 06:17 UTC on Sept 6:
the R8_e8_ema net (checkpoint `912e617b…`) shipped as fp32, fused and int8 graphs
(the two head MatMuls kept fp32), the same `agent.py` and `chessml` as fe58a9c, and
a loader that times the graphs at init and plays the fastest; round 40 onwards
`submission_R8_int8_tb.zip`, built 15:51 UTC, the same plus the Syzygy 3-4 piece
tables of `aae6dd1` (the round-40 log's init reads `syzygy 70 tables up to 4
pieces`). The logs date each upload: an upload made a few minutes before a game
starts plays that game. The clocks agree: not one of the 722 searched
moves in rounds 1 to 14 stopped before its deadline, and in round 15 the bot
stopped 62 of 63 searches early and spent 47% of its budget. The moves agree
too: replaying the 210 search-decided positions of rounds 1 to 10 at 300
simulations under each net, R3 matches the bot's move more often in rounds 3
to 8 (78/67/79/74/73/69% against R1's 67/39/67/68/55/62%) and R1 in rounds 9
and 10 (73/71% against 42/54%); where the two searches choose different moves,
round 9 goes to R1 eleven to one and round 7 to R3 ten to four. The code before
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
| 16 | ms | White | win | mate | +2.6 from move 20 after the opponent's 19...Kh8, converted over 77 moves with 9.9 s left |
| 17 | mate-in-one | Black | loss | mate | −0.9 at move 12 already; every bot move from 12 to 24 within a quarter pawn of Stockfish's; outplayed, 30.7 s left |
| 18 | sobriety | White | loss | mate | three 0.6-pawn slips at 15, 20 and 22 (Rab1, Ng3, Kg1 were better), then a melee played at Stockfish's first choice from 27 on; 57.9 s left |
| 19 | gijs-smit | Black | draw | threefold | repeated queen checks from move 39 with our own q at +0.5 and 53 s left: the round-8 rule mismatch again, on the build without the fix |
| 20 | checkers | White | win | mate | q rising from move 24, mated at 72 with 16 s left; not yet through Stockfish |
| 21 | slopfish | White | draw | threefold | level throughout: no move by either side lost more than 35 cp; 51 s left |
| 22 | Lubina | White | draw | insufficient material | −1.4 at move 27, +1.2 after 27...Ne7, +3.3 by move 60 with our q at +0.5 to +0.8; 68. Rf3 at 1.6 s with 16 s left gave it back (Nc5 kept +2.2); 5.2 s left at move 98 |
| 23 | No More Ammo | Black | draw | threefold | +5.5 after 59. Bb5, where 59...Bxb5 wins the pawn race; 59...Bf7 at q +0.16 with 28 s left returned 0.0, then the bishops shuffled into the threefold |
| 24 | THE ROOOOOKKK!!!! | Black | loss | mate | +1.4 at move 41, five bishop shuffles at full budget, then 46...Rd7 at q +0.24 with 38 s left lost the exchange (Rde8 held −0.6); mated at 83 with 9 s |
| 25 | pheanup | Black | loss | mate | drift to −0.8 by move 22; 28...Rc1+ (−1.6, Bxf3 held) and 33...Qe6 after 320 simulations with 68 s left (−2.6 more, Rxb2 held); queen lost at 39, mated at 63 |
| 26 | mangodogo | White | draw | threefold | worse from 20. Bc4, −1.9 by move 36; the opponent gave it back twice; the repetition at 46 was 0.0 with our q at 0.00 |
| 27 | Something | White | loss | mate | +0.5 for 25 moves, then 26. Be2 and 28. Bb2 (−0.9 each) and four rook moves at full budget, 36. Rf1 to 40. Red6, from 0.0 to −2.0; mated at 80 with 13 s |
| 28 | Make_no_mistakes | Black | loss | mate | 20...Qa1+ at 99 s and q −0.05 lost 2.1 pawns (Bb7 held 0.0), a move both builds play at 2,400 simulations; mated at 52 |
| 29 | Amplifirm | Black | draw | threefold | +3.1 at move 23 after the opponent's two exchange sacrifices; q +0.65 to +0.79 through move 44 while six slips at 45 to 98 s (37...e5 −2.0, 39...Bh8 −1.4) took it to 0.0; perpetual check from move 60 |
| 30 | Finlay Phillips | Black | win | mate | clean: +1.9 after 25. Qg3, q tracking Stockfish from +0.36 to +0.96, mated at 70 with 13 s; the opponent got down to 1.2 s |
| 31 | LeetBeaters | White | loss | mate | first R8 int8 game, on the slowest machine of the set; 16. fxe4 and 29. bxc6 to −2.4 at full clock, the opponent gave it all back, 51. Kg2 missed Kg4 (+2.1), 54. Rc6 and 58. Be3 lost it again; 98 moves, 5 s left |
| 32 | CheckmateGPT | White | loss | mate | +0.7 at move 24 with 83 s; 25. h5, 28. Kb1, 40. Ka2 and 41. Qc1 (Qb5 held) at full clock took it to −4.3 while q went +0.41 to −0.27 |
| 33 | zachFree-zone_sponsorPhanty | Black | win | mate | +1.6 by move 21, +5.8 by 54; nine slips of 50 to 360 cp between moves 54 and 89 at 4 to 21 s changed nothing; mated at 95 with 3.7 s |
| 34 | The Good Boys | White | win | mate | +2.8 by move 26, the opponent erred back each time we did; converted at 88 with 5.4 s |
| 35 | Ryan Vincent | White | win | mate | +1 to +4 from move 31, a queen ending converted at 89 with 6.6 s |
| 36 | zak | Black | draw | insufficient material | level throughout, one 64 cp slip at 24...g3 answered by 25. Nxe4; king against king at 74 |
| 37 | Solo Man | White | draw | threefold | 489 plies, the longest game in the set; a pawn down from move 34, 16 s at move 60 and 2 to 5 s from move 90 for 145 moves; lost on the board from 62. Rh5 and 65. Kh5, mate scores through the 80s and 90s, but Solo Man at 2.2 s could not convert and gave it back a dozen times; bishop against bishop by 240 |
| 38 | mangodogo | Black | win | mate | +1.4 after the opponent's 12. Nxe5, +3.3 by 33, a pawn promoted at 54, mated at 73 with 16 s |
| 39 | adashima | Black | win | mate | +5 by move 41 and mate in 11 at move 63; 29 more moves to deliver it, queen and bishop against a bare king from move 73 at 0.5 s a move, mated at 92 with 2.6 s; the repetition rule refused a repeat eight times at q +0.9 |
| 40 | xx | White | win | mate | first game with the Syzygy tables (never reached four pieces: queen and rook against king and pawn at the mate); +1.0 after the opponent's 12. Rxf3, +6 by move 45, mated at 64 with 13.6 s; the strongest team beaten so far (2052, 21st) |

Score 14.5/25. The wins were all against opponents who blundered, eight of the
nine into mate; the two losses to leaders in rounds 17 and 18 came from
accumulated small errors, not a blunder. Rounds 21 to 25 went D D D L L, all
against teams ranked 2 to 13 on the public leaderboard (1986 to 2144 that
evening; we stood 15th of 282 at 1960 after round 24, down from 2001 after
round 20). Rounds 26 to 30 went D L L D W
against 1882, 1906, 1993, 1845 and 1849; the rating fell from 2001 after round 20
to 1848 after round 29, 31st of 295. The pruning build's tally, rounds 15 to 30,
is 8.5/16; the fixes zip's, rounds 24 to 30, 1.5/7. Rounds 31 to 40, the R8 int8
upload, went L L W W W D D W W W, 7/10 against an expectation of 4.7 from the
ratings (the losses to the 12th-ranked team and to a 1789, the wins against 1733
to 2052); the rating fell to 1765 after round 32 and stood at 2027 after round 39,
23rd of 323. Score 23.5/40. Rounds 11 to 18 were analysed the same
way as the first ten; rounds 16 to 18 were played by the Sept 5 upload
(temperature 1.359, root FPU 1.0, pruning; no repetition fix).

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

**Rounds 17 and 18 are losses to stronger play, and the clock was not used.**
The platform's own match log for round 17 (`aichessathon-round-17-mate-in-one.log`,
beside the PGN) settles two open questions. Init measured `forward_ms` 5.44 and
832 pre-search simulations in 5 s, so a fresh simulation costs about 6 ms there,
Mac-like and a little faster than the overrun estimate. And `pondered` is 6 to
10 per move, so the process really is suspended between moves. The per-move
lines show the search spending a median 480 simulations, stopping early on 42
of 48 moves, four times under 200 simulations with more than a minute in hand
(moves 28 and 37 stopped after 32 new simulations because a reused subtree of
900 to 1,300 visits made the visit lead look unbeatable), and the 4 s cap
binding on five moves. Time used was 113 s of 144 s available, and round 18
ended with 58 s unused. Whether more search would have changed the drift
moves is the replay question recorded below.

**Rounds 21 to 25: two wins thrown in one move each, and the draw rule was
never in play.** Round 21 was a real draw: the whole game inside 35 cp for both
sides, final position 0.0. The other four each turned on a single position,
all scored with Stockfish at 4 s, one thread, multipv 4:

| position | clock left | search | our q | played | Stockfish | best |
|---|---|---|---|---|---|---|
| R22 68. | 16.5 s | 320 sims, 1.62 s, stopped early | +0.69 | Rf3 +43 | +240 before | Nc5 +223 |
| R23 59... | 28.1 s | 416 sims, 1.67 s, stopped early | +0.16 | Bf7 +3 | +546 before | Bxb5 +770 |
| R24 46... | 38.5 s | 736 sims, 3.05 s, stopped early | +0.24 | Rd7 −252 | −42 before | Rde8 −58 |
| R25 28... | 75.3 s | 768 sims, 3.85 s, stopped early | −0.24 | Rc1+ −233 | −34 before | Bxf3 −32 |
| R25 33... | 68.2 s | 320 sims, 1.67 s, stopped early | −0.47 | Qe6 −737 | −315 before | Rxb2 −291 |

Round 22 is the conversion failure of rounds 5, 9 and 14 once more, and this
time the search knew: q sat at +0.5 to +0.8 from move 30 to move 67 while
Stockfish had +1.3 to +3.3 (rook and knight against rook and knight, an outside
a-pawn). The clock was the constraint. The 46-move horizon had spent 92 s of
the first 117 by move 50 in a position that was level or worse, so the winning
ending from move 50 to 98 was played on 28 s plus the increment: 22 of the 91
searches ran under 200 simulations, and 68. Rf3 with 16.5 s (q +0.69, Nc5 was
+2.2) let the knight in; the a-pawn fell at move 79 and the game ended king
against king with 5.2 s left. The earlier slips (38. Rcc3, 44. Rb3, 47. Rc7,
60. Kf4) cost 20 to 65 cp each at 4 s, not the win. This is the game the step-2
clock reshape (`docs/ARENA13_STEP2_CLOCK.md`) is for; its 44% was measured on a
thrashing Mac and still needs the rerun on Justin's list.

Round 23 is the search's blind spot rather than the clock's. After No More
Ammo's 58. Bf1 and 59. Bb5 the position was +5.5: 59...Bxb5 60. axb5 a4 and the
a-pawn queens two moves before the g-pawn. Our search, with 28 s in hand, stopped
after 416 simulations at q +0.16 and kept the bishops; Stockfish's Bf7 is 0.0,
and from there Bg6/Bf7 against Bd7/Bc6 repeated. The draw rule did not fire
because q never reached +0.3, and it was right not to: the position after Bf7
is drawn. A pawn race ten plies deep is invisible to the value head and to 400
simulations; proofs would not see it either (no mate). Round 23 also drifted
from +1.2 at move 26 to 0.0 by move 29 through 24...Qe6, 26...Be5 and 27...Qxg4
(Bxb5 kept +1.2), each 50 to 90 cp at 2 to 3.5 s.

Round 24, against the second-ranked team, was better for 30 moves: Stockfish
+0.4 to +1.4 from move 21 to 41 with q rising to +0.44, so the net and the
board agreed. Then five bishop moves in a row, Be7 Bb4 Be7 Bf8 Bb4, at 860 to
1,090 simulations and 3.1 to 3.7 s each, gave the pawn back (41...Be7 cost 102
cp) and made no plan in a closed position, and 46...Rd7 at full budget with 38 s
walked into 47. Bh4 Be7 48. Nc5, the exchange, and a −3 ending the opponent
converted cleanly. The bot's q was +0.46 the move before Stockfish's 0.0 and
−0.10 when Stockfish had −2.7: the value head lags tactics by two moves in this
kind of position, and 736 simulations did not close the gap.

Round 25 was lost at full clock. Two decisive errors with 75 s and 68 s left:
28...Rc1+ gave 1.6 pawns for a check (Bxf3 held −0.3), and 33...Qe6, after a
search that stopped at 320 simulations with a 68 s clock and a 4 s cap, walked
into 34. Qd8+ and 35. Ra8, losing the queen for rook and bishop by move 39.
That stop is the round-17 pattern: a reused subtree of 254 visits made the
visit lead look unbeatable, so pruning ended the search at 40% of its budget in
a position that was already −3 and about to be −7. Before that the drift from
0.0 to −0.8 by move 22 came in 30 to 40 cp steps (11...c5, 16...Qc8, 19...h6,
21...Qe6) with a minute and a half on the clock: outplayed, not out-clocked.

What the five games add to the list: conversion of won endings is the largest
single leak in the set, now five half-points (rounds 5, 9, 14, 22, 23), and it
splits into a clock problem (22: nothing left by move 50) and a calculation
problem (23: a pawn race no value head sees); the smart-pruning stop on a reused
subtree ended two decisive searches early (R25 33..., R23 59...) with 28 to 68 s
in hand; and the draw rule, on or off, decided nothing in any of the five.

**Rounds 26 to 30: a won game thrown at full clock, and why the search cannot
feel it.** Round 29 removes the clock from the conversion story. After Amplifirm's
two exchange sacrifices we were +3.1 at move 23 with 98 s; the search read q
+0.65 and kept reading +0.62 to +0.79 for 22 moves while Stockfish went +3.1,
+2.0, +1.8, +0.9, +0.3: 23...Rxa2 (a5 kept +2.9), 24...a5, 31...Rc8, 37...e5
(Rh8 kept +3.8), 39...Bh8 (Ra8 kept +2.0), 45...Ke6, each at 2.3 to 3.9 s and
288 to 896 simulations with 45 to 98 s in hand. By move 58 it was 0.0 and the
knight's perpetual from move 60 was the correct result. The search never saw
the advantage go, and the table below says why. Over the 536 searched positions
of rounds 21 to 30 with both a q and a Stockfish score:

| Stockfish, our view | positions | mean q | spread |
|---|---|---|---|
| −300 to −150 | 33 | −0.37 | 0.20 |
| −150 to −50 | 41 | −0.12 | 0.14 |
| −50 to +50 | 257 | +0.08 | 0.14 |
| +50 to +150 | 47 | +0.37 | 0.17 |
| +150 to +300 | 55 | +0.62 | 0.08 |
| +300 to +500 | 18 | +0.68 | 0.14 |
| above +500 | 15 | +0.88 | 0.19 |

Between −1 and +1 the value head moves 0.29 per pawn; between +1 and +4 it
moves 0.10, and +1.5 and +3.5 read the same to within the spread. So a
two-pawn slip inside a won position changes nothing the search selects on, and
the +0.3 threshold that arms the repetition avoidance is crossed at about +0.8
pawns and stays crossed. This is the same head in both builds (rounds 21 to 23
+0.62 in the +150 to +300 band, rounds 24 to 30 +0.63). It is a training
target question, not a search one: a value head that separates +1.5 from +3.5
(a WDL head, or the scaled targets of `docs/ARENA8_R2_VALUE_SCALE.md`) is what
conversion needs, and it is the first thing to measure on the R6b and R8 nets.

**The fixes zip and the losing streak.** Rounds 24 to 28 scored 0.5 against an
expectation of 2.07 from the opponents' ratings, a 5% event under no change,
after rounds 15 to 23 had scored 5.0 against 3.73. The eleven positions that
decided rounds 22 to 28 were replayed through both uploads' search code, same
net, 500 and 1,500 simulations: 19 of the 22 choices are identical, including
every losing move (R24 46...Rd7 at 1,500, R25 33...Qe6, R27's four rook moves,
R28 20...Qa1+ at 2,400 simulations, q −0.1). Of the three that differ, two favour
the fixes (R23 59...Bxb5 found at 1,500, R25 28...Rc1+ avoided at 1,500) and one
the old code (R24 46...Rc8 at 500, a 93-to-84 visit near-tie). The root rule had
a move to make on four of the 406 searched moves of rounds 24 to 30, twice in
round 24 at a cost of 9 and 0 cp, twice in round 30's mating sequence. The
in-tree rule is visible only in the simulation counts: 13,000 to 83,000 per move
in round 29's perpetual, where repeated positions end a path without a forward
pass.

**Rounds 31 to 38: the R8 int8 upload on the platform.** The first thing the logs
settle is the speed. `load_fastest` picked the int8 graph on every one of the eight
machines: 2.83 to 3.17 ms against 3.45 to 4.06 for the fused graph and 4.23 to 5.03
for fp32 on seven of them, 1.45 to 1.6 times fp32, less than the box's 1.83 but real;
round 31's machine was the slow one of the set (4.59 / 5.04 / 5.96 ms, 21 s to
ready). The pre-search made 1,184 to 1,344 simulations in 5 s against 768 to 1,070
before, and the median search per move was 460 to 800 simulations against 320 to
576 in rounds 15 to 30, so the platform is now at roughly the 800-simulation side of
ARENA #11's doubling. The games themselves: two losses, one to the 12th-ranked team
in a game R1 would have lost the same way (the replay in the previous section's
method applies: R1, R8 fp32 and R8 int8 chose the same move at 13 of 16 decisive
comparisons, and int8 matched fp32 at all 16), and one to a 1789 at full clock,
round 32, where four moves between 25 and 41 (h5, Kb1, Ka2, Qc1) took +0.7 to −4.3
while q read +0.41 to −0.27. The five wins were conversions against 1733 to 1959
teams, all played down to 3.7 to 16 s.

R8's value head has the same shape as R1's. On the 568 searched positions of rounds
31 to 38 with both a q and a Stockfish score, the band means are +0.32 (+50 to +150),
+0.57 (+150 to +300), +0.74 (+300 to +500) and +0.90 above +500, against R1's +0.37,
+0.62, +0.68 and +0.88; the slope between +1 and +4 pawns is 0.14 per pawn against
R1's 0.10. Same training target, same saturation, so the conversion risk of rounds
22, 23 and 29 is unchanged by the net; the extra search is what changed.

**Round 37 and two platform rules the code did not know.** Round 37 ran to ply 489
and ended by threefold. The platform's docs give the failure table: `ply_cap`, "the
game reached 600 plies", result draw. `agent.py` carries `_PLY_CAP = 300` and
`harness/rules.py` `PLY_CAP = 300`, and `_pick` switches to material-based winning
and losing from ply 240, so in round 37 the branch was live from move 120 to move
245 on a cap that does not exist, and the cap that does is a draw for both sides,
not an adjudication. In this game it happened to be right: a pawn down from move 34
(13. b4 was the slip), the bot reached 16 s at move 60 and then played 145 moves at
0.55 s each on a 2 to 5 s clock, lost on the board from 62. Rh5 and 65. Kh5 by
Stockfish's count with mate scores through the 80s and 90s, and was saved because
Solo Man, itself at 2.2 s, gave the win back a dozen times until bishop against
bishop at move 240. The second rule is the log: stdout and stderr are kept as the
first 4 KB plus the last 4 KB, which is why the round-37 log holds moves 9 to 62 and
237 to 245 and nothing between. Both are in `BACKLOG.md` G12 with the fix.

**Round 39 is the mate-finding problem in its purest form.** Against adashima the
bot was +5 by move 41 and Stockfish had mate in 11 at move 63, mate in 8 at 71 and
mate in 4 at 79, queen and bishop against a bare king from move 73. The bot read
q +0.92 to +1.00 throughout, which is the value head's ceiling and carries no
distance-to-mate, so it played 29 more moves after the mate-in-11, checking from
square to square at 0.5 to 0.7 s and 190 to 320 simulations a move while the clock
went from 14 s to 2.6 s. It won because a bare king has no counterplay; against a
defender with a pawn it would have flagged. The repetition rule was live the whole
way and refused a repeat eight times at q +0.9, the round-19 failure inverted.
This is the case for proofs or, since it is a four-piece ending, for the Syzygy
3-4 piece tables, which landed in `aae6dd1` the same day: exact terminals in the
tree and a distance-to-zeroing pick at the root. Replayed through them, round 39
from move 73 (the first position with four pieces on our move) is mate in five,
nine plies, against the game's 26 moves; the tables' own tests pass. Their cover
starts at four pieces, so moves 63 to 72, seven pieces down to five with the
mate-in-11 already on the board, stay with the value head, which reads +0.95
for all of them.

## What the match logs add

The platform's per-game logs sit beside the PGNs from round 1. Our stdout is
captured only from round 10 on, so rounds 1 to 9 carry the clock table and the
result but no search telemetry. What the captured games show:

- **Pondering stopped between rounds 15 and 16.** Pondered simulations per move
  were 199 to 473 in rounds 10 to 15 and 7 to 23 from round 16 (Sept 5, 07:06 UTC),
  which dates the platform's suspension rule and the harness change that mirrors it.
- **The machines are alike.** Ten machine ids over eleven games: init forward time
  4.5 to 6.2 ms, pre-search 768 to 1,070 simulations in 5 s, about 150 to 210 fresh
  simulations per second. The 10 ms per simulation inferred from clock overruns was
  the in-game figure with tree bookkeeping, not the raw forward.
- **The pruning build leaves time in short games only.** Time used and left at the
  end: round 15 115 s and 36 s, 16 145 s and 10 s, 17 113 s and 31 s, 18 79 s and
  58 s, 19 86 s and 53 s, 20 136 s and 16 s, 21 87 s and 51 s, 22 160 s and 5 s
  (98 moves), 23 125 s and 23 s, 24 148 s and 9 s (83 moves), 25 125 s and 23 s,
  26 104 s and 38 s, 27 142 s and 13 s, 28 104 s and 39 s, 29 129 s and 19 s,
  30 138 s and 13 s; the R8 int8 games 31 160 s and 5 s, 32 135 s and 14 s, 33 160 s
  and 4 s, 34 156 s and 5 s, 35 155 s and 7 s, 36 143 s and 11 s, 37 236 s and 2 s,
  38 137 s and 16 s, 39 163 s and 3 s, 40 135 s and 14 s. The faster net did not change the formula, so games are still
  played down to a few seconds.
  Median simulations per move 320 (round 22) to 480; searches under 200
  simulations 2 to 6 per game except 22 in round 22. The pre-pruning games used 120 to 169 s and ended at
  2 to 19 s. Pruning stopped 33 of 34 to 66 of 70 searches; the 4 s cap bound on
  five moves in round 17.
- **The decisions we replayed, as the search saw them.** Round 14 move 45 had
  209 simulations at q +0.73, and the replay's flip point was 250. Round 18 moves
  15, 20 and 22 had 672 to 736 simulations each at 3.0 to 3.9 s, and the replay
  finds no better move up to 2,500, so those were the net's opinion, not a budget
  shortfall. Round 10 move 14 had 670 simulations plus a reused subtree of 1,055.
- **The ponder thread crashes after a mating move.** Every win by checkmate with
  pondering (rounds 11, 12, 13, 15, 16, 20, 30, 33, 34, 35, 38) ends its log with `ValueError: no
  legal moves at search root` from `_ponder`. The game is over by then, so it costs
  nothing; a one-line guard in `_ponder` would silence it.
- **Round 19 lost a half point to the repetition rule.** From move 39 the search
  rated the position +0.46 to +0.57 and played Qg5, Qg1+, Qf1+, Qg1+, Qf1+; the
  referee declared the threefold with 53 s on our clock. The fix in
  `ship-step1`/`ship-combined` was not in the build that played.

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
the platform's nor idle. Rounds 21 to 30 were scored on 2026-09-05, and 31 to 40 on 2026-09-06, at 0.25 s and one thread
for the trajectory and 4 s multipv 4 on the named positions; the calibration
table pools every searched position of those ten games with a q in the log. The
build replay ran each upload's `agent.py` and `chessml` from its zip with
`node_budget` 500 and 1,500 on the Mac, eleven positions, no arenas (the queues
are stopped, see `docs/justins-simulations.md`);
the opponents' ranks and ratings are the public leaderboard at 17:20 BST that
day, and ours is the team page's rating history. The first-five-move losses were scored on 2026-09-04
with Stockfish at depth 16 on one thread, our side identified by the clock
fingerprint above. The lichess opening explorer refused unauthenticated
requests that day, so how deep a human-games book would follow the curated
starts is unmeasured.
