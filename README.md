# AI Chessathon starter

Fork this to build an agent for [AI Chessathon](https://aichessathon.com). It gives you a working
submission, baselines to beat, and a local harness that speaks the same protocol and enforces the
same clock as the platform, so you can see whether a change actually helped before you upload it.

```
git clone https://github.com/advitrocks9/aichessathon-starter
cd aichessathon-starter
make setup
make play
```

That plays your agent against a baseline over a full 120 s + 0.5 s game and prints the result.
When you like it, `make zip` and drop `submission.zip` on your dashboard.

## Writing an agent

`agent.py` is the whole submission. One function:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The fork ships a legal random-mover, so the loop works before you write anything. Replace the body.

```
make play                                          # one game, real time control
make arena                                         # 20 fast games, prints a score
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

Anything your agent writes to stdout or stderr shows up under the result, so `print` debugging
works. The platform discards it during rated games and shows it in your validation log.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 20 | 10 s + 0.1 s | 10.0% (+1 =2 -17) |
| greedy vs minimax | 6 | 120 s + 0.5 s | 0.0% (+0 =0 -6) |
| numba vs minimax | 6 | 10 s + 0.5 s | 66.7% (+2 =4 -0) |

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## What's here

```
agent.py             the submission: time management, history, pondering, degraded modes
chessml/             what agent.py imports - encoding, ONNX net, PUCT search
baselines/           random, greedy, minimax, numba, reference-hero; each a dir with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and adjudication rules
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
train/               model, ONNX export, supervised trainer - never imported at play time
pipeline/            data: acquire, filter, shard, Stockfish-label, validate
docs/DECISIONS.md    every material decision, with the evidence behind it
docs/PIPELINE_BRIEF.md  the data and training runbook
docs/IDEAS.md        where the strength actually comes from
docs/PROVENANCE.md   how to check the shipped weights are ours
docs/STOP<n>_*.md    pipeline stop-and-report points, numbered by PIPELINE_BRIEF §7
docs/ARENA<n>_*.md   measurement reports: games played, Elo estimated
docs/FINDING_*.md    a bug or effect worth recording, filed rather than fixed
```

**Two numbering schemes, deliberately separate.** `STOP<n>` belongs to the five
stop-and-report points defined in `docs/PIPELINE_BRIEF.md` §7 — they gate the run
matrix (R0 -> RA, then the full track) and each one waits on a confirmation before
the next stage starts. Only those five may take a STOP number. `ARENA<n>` is for
results of games actually played, numbered independently in the order they were
run. A sparring result is not a stop point, however useful it is: reusing the
STOP sequence for one makes it ambiguous whether the protocol has advanced.

`weights/` is a build artifact, not in git: `make weights` regenerates it, and
`make baseline-hero` rebuilds the reference-project opponent. Both `train/` and
`pipeline/` need the `train` dependency group and are outside the mypy strict set;
nothing in either ever enters the submission.

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
