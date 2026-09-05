from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.shard import ARRAYS, Shard

# The MultiPV lines of some rows: moves int16 (n, 4), -1 where absent; values float16 (n, 4) on the
# [-1, 1] scale, NaN where absent; sorted best first (pipeline/shard.py).
SoftLabels = tuple[np.ndarray, np.ndarray]
# x, the human move, the value target, side to move, and the MultiPV lines when the policy source
# asks for them (None under "human").
Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SoftLabels | None]

BLOCK_ROWS = 262_144
WINDOW_BLOCKS = 16
RAM_LIMIT_BYTES = 12_000_000_000

_X_DTYPE, _X_SHAPE = ARRAYS["X"]
ROW_BYTES = _X_DTYPE.itemsize * int(np.prod(_X_SHAPE, dtype=int))


def side_to_move(shard_dir: Path, n: int) -> np.ndarray:
    cache = shard_dir / "stm.bin"
    if cache.exists() and cache.stat().st_size == n:
        return np.fromfile(cache, dtype=np.int8)
    stm = np.zeros(n, dtype=np.int8)
    with open(shard_dir / "fen.txt", "rb") as fh:
        for i in range(n):
            stm[i] = 1 if b" w " in fh.readline() else 0
    stm.tofile(cache)
    return stm


VALUE_SOURCES = ("engine", "lichess", "outcome", "blend")
POLICY_SOURCES = ("human", "multipv", "mix")


@dataclass(frozen=True)
class Labels:
    policy: np.ndarray  # the human move, int64 (n,)
    value: np.ndarray  # the value target, float32 (n,)
    stm: np.ndarray  # side to move, int8 (n,)
    value_labelled: int  # rows whose value came from the chosen source, not the outcome
    soft: SoftLabels | None  # the MultiPV lines, or None under policy_source "human"
    policy_labelled: int  # rows with at least one engine line (0 when soft is None)


def _labels(
    shard: Shard,
    idx: np.ndarray,
    stm: np.ndarray,
    value_source: str = "engine",
    policy_source: str = "human",
) -> Labels:
    """Policy and value targets, side to move, and how many rows took a label from the chosen
    sources rather than the game record.

    Value:
    engine   the Stockfish label where one exists, else the outcome (R2-R5)
    lichess  the Lichess [%eval] where one exists, else the outcome (R1's target)
    outcome  the game outcome on every row
    blend    0.5 engine + 0.5 outcome where the engine label exists, else the outcome (R7)

    Policy:
    human    the move played, as a one-hot index (R0-R7)
    multipv  the Stockfish MultiPV lines where they exist, the human move elsewhere (R8)
    mix      both, weighted by the trainer's alpha where lines exist (R8b)
    The lines ship raw; the trainer turns them into a distribution (train.train.policy_target).
    """
    if value_source not in VALUE_SOURCES:
        raise ValueError(f"value_source must be one of {VALUE_SOURCES}, not {value_source!r}")
    if policy_source not in POLICY_SOURCES:
        raise ValueError(f"policy_source must be one of {POLICY_SOURCES}, not {policy_source!r}")
    policy = np.asarray(shard.Y_policy)[idx].astype(np.int64)
    outcome = np.asarray(shard.Y_value)[idx].astype(np.float32)
    soft = None
    policy_labelled = 0
    if policy_source != "human":
        moves = np.asarray(shard.Y_policy_engine4)[idx]
        values = np.asarray(shard.Y_value_engine4)[idx]
        soft = (moves, values)
        policy_labelled = int((moves[:, 0] >= 0).sum())
    if value_source == "outcome":
        return Labels(policy, outcome, stm[idx], 0, soft, policy_labelled)
    column = shard.Y_value_lichess if value_source == "lichess" else shard.Y_value_engine
    label = np.asarray(column)[idx].astype(np.float32)
    present = ~np.isnan(label)
    target = 0.5 * label + 0.5 * outcome if value_source == "blend" else label
    value = np.where(present, target, outcome).astype(np.float32)
    return Labels(policy, value, stm[idx], int(present.sum()), soft, policy_labelled)


def _pick(soft: SoftLabels | None, pos: np.ndarray) -> SoftLabels | None:
    return None if soft is None else (soft[0][pos], soft[1][pos])


def describe_target(value_source: str, labelled: int, n: int) -> str:
    if value_source == "outcome":
        return "value target: game outcome on every row"
    return (
        f"value target: {value_source} on {100 * labelled / max(n, 1):.2f}% of rows, "
        "outcome elsewhere"
    )


def describe_policy(policy_source: str, labelled: int, n: int) -> str:
    if policy_source == "human":
        return "policy target: the human move on every row"
    what = "multipv lines" if policy_source == "multipv" else "human move mixed with multipv lines"
    return (
        f"policy target: {what} on {100 * labelled / max(n, 1):.2f}% of rows, "
        "the human move elsewhere"
    )


class _BlockReader:
    """Reads X in contiguous row blocks with plain file I/O into one staging buffer. Nothing is
    memory-mapped, so a consumed block costs nothing afterwards (docs/DECISIONS.md section 9)."""

    def __init__(self, x_path: Path, n: int, block_rows: int) -> None:
        self.x_path = x_path
        self.blocks = [(s, min(s + block_rows, n)) for s in range(0, n, block_rows)]
        self.staging = np.empty((block_rows, *_X_SHAPE), dtype=_X_DTYPE)

    def rows(self, fh, block: int, mask: np.ndarray, out: np.ndarray) -> int:
        """Copy the rows of `block` that `mask` selects into `out`; returns how many."""
        lo, hi = self.blocks[block]
        m = mask[lo:hi]
        k = int(np.count_nonzero(m))
        if k == 0:
            return 0
        n_rows = hi - lo
        flat = self.staging.reshape(-1)[: n_rows * ROW_BYTES]
        fh.seek(lo * ROW_BYTES)
        got = fh.readinto(flat)
        if got != flat.nbytes:
            raise OSError(f"{self.x_path}: short read at block {block}: {got} of {flat.nbytes}")
        np.compress(m, self.staging[:n_rows], axis=0, out=out[:k])
        return k


class RamSplit:
    def __init__(
        self,
        shard: Shard,
        which: int,
        stm: np.ndarray,
        name: str,
        *,
        block_rows: int = BLOCK_ROWS,
        value_source: str = "engine",
        policy_source: str = "human",
    ) -> None:
        mask = np.asarray(shard.split) == which
        idx = np.flatnonzero(mask)
        self.n = idx.size
        print(f"  loading {name}: {self.n:,} rows (ram)", flush=True)
        t = time.time()
        self.x = np.empty((self.n, *_X_SHAPE), dtype=_X_DTYPE)
        reader = _BlockReader(shard.root / "X.bin", shard.n, block_rows)
        fill = 0
        with open(reader.x_path, "rb") as fh:
            for b in range(len(reader.blocks)):
                fill += reader.rows(fh, b, mask, self.x[fill:])
        assert fill == self.n
        labels = _labels(shard, idx, stm, value_source, policy_source)
        self.policy, self.value, self.stm, self.soft = (
            labels.policy,
            labels.value,
            labels.stm,
            labels.soft,
        )
        print(
            f"    {self.x.nbytes / 1e9:.1f} GB in {time.time() - t:.0f}s; "
            + describe_target(value_source, labels.value_labelled, self.n)
            + "; "
            + describe_policy(policy_source, labels.policy_labelled, self.n),
            flush=True,
        )

    def batches(self, batch: int, rng: np.random.Generator | None) -> Iterator[Batch]:
        order = rng.permutation(self.n) if rng is not None else np.arange(self.n)
        stop = self.n - (batch - 1) if rng is not None else self.n
        for s in range(0, stop, batch):
            sel = order[s : s + batch]
            yield (
                self.x[sel],
                self.policy[sel],
                self.value[sel],
                self.stm[sel],
                _pick(self.soft, sel),
            )


class BlockShuffledSplit:
    """X stays on disk. Each epoch reads its blocks in shuffled order, `window_blocks` at a time,
    into one preallocated window that is shuffled whole and gathered by index. Resident memory
    is one window (two with `prefetch`) plus one block of staging, and it does not grow."""

    def __init__(
        self,
        shard: Shard,
        which: int,
        stm: np.ndarray,
        name: str,
        *,
        window_blocks: int = WINDOW_BLOCKS,
        block_rows: int = BLOCK_ROWS,
        prefetch: bool = False,
        value_source: str = "engine",
        policy_source: str = "human",
    ) -> None:
        self.mask = np.asarray(shard.split) == which
        idx = np.flatnonzero(self.mask)
        self.n = idx.size
        self.row_of = idx
        labels = _labels(shard, idx, stm, value_source, policy_source)
        self.policy, self.value, self.stm, self.soft = (
            labels.policy,
            labels.value,
            labels.stm,
            labels.soft,
        )
        self.pos_of = np.full(shard.n, -1, dtype=np.int64)
        self.pos_of[idx] = np.arange(self.n)
        self.reader = _BlockReader(shard.root / "X.bin", shard.n, block_rows)
        self.window_blocks = window_blocks
        self.block_rows = block_rows
        self.prefetch = prefetch
        window_gb = window_blocks * block_rows * ROW_BYTES / 1e9
        copies = 2 if prefetch else 1
        print(
            f"  loading {name}: {self.n:,} rows (block-shuffled, "
            f"{self.n * ROW_BYTES / 1e9:.1f} GB on disk; resident "
            f"{window_gb:.1f} GB window x{copies} + "
            f"{self.reader.staging.nbytes / 1e9:.1f} GB staging)",
            flush=True,
        )
        print(
            f"    {describe_target(value_source, labels.value_labelled, self.n)}; "
            f"{describe_policy(policy_source, labels.policy_labelled, self.n)}; "
            f"{len(self.reader.blocks):,} blocks of {block_rows:,}",
            flush=True,
        )

    def batches(self, batch: int, rng: np.random.Generator | None) -> Iterator[Batch]:
        blocks = self.reader.blocks
        order = rng.permutation(len(blocks)) if rng is not None else np.arange(len(blocks))
        windows = [
            order[w : w + self.window_blocks] for w in range(0, len(order), self.window_blocks)
        ]
        capacity = self.window_blocks * self.block_rows + batch
        slots = 2 if self.prefetch else 1
        buf_x = [np.empty((capacity, *_X_SHAPE), dtype=_X_DTYPE) for _ in range(slots)]
        buf_p = [np.empty(capacity, dtype=np.int64) for _ in range(slots)]
        rows = [0] * slots
        carry_x = np.empty((0, *_X_SHAPE), dtype=_X_DTYPE)
        carry_p = np.empty(0, dtype=np.int64)

        with open(self.reader.x_path, "rb") as fh:
            failure: list[BaseException] = []

            def fill(slot: int, window: np.ndarray, cx: np.ndarray, cp: np.ndarray) -> None:
                try:
                    n = len(cx)
                    buf_x[slot][:n] = cx
                    buf_p[slot][:n] = cp
                    for b in window:
                        lo, hi = blocks[b]
                        k = self.reader.rows(fh, int(b), self.mask, buf_x[slot][n:])
                        if k:
                            buf_p[slot][n : n + k] = self.pos_of[lo:hi][self.mask[lo:hi]]
                            n += k
                    rows[slot] = n
                except BaseException as exc:  # re-raised on the consuming thread
                    failure.append(exc)

            def start(
                slot: int, window: np.ndarray, cx: np.ndarray, cp: np.ndarray
            ) -> threading.Thread:
                thread = threading.Thread(target=fill, args=(slot, window, cx, cp), daemon=True)
                thread.start()
                return thread

            pending: threading.Thread | None = None
            try:
                if windows:
                    pending = start(0, windows[0], carry_x, carry_p)
                for wi, _ in enumerate(windows):
                    slot = wi % slots
                    if pending is not None:
                        pending.join()
                        pending = None
                    if failure:
                        raise failure[0]
                    n = rows[slot]
                    if n == 0:
                        if wi + 1 < len(windows):
                            pending = start((wi + 1) % slots, windows[wi + 1], carry_x, carry_p)
                        continue
                    perm = rng.permutation(n) if rng is not None else np.arange(n)
                    full = (n // batch) * batch
                    carry_x = buf_x[slot][perm[full:]]
                    carry_p = buf_p[slot][perm[full:]]
                    next_window = windows[wi + 1] if wi + 1 < len(windows) else None
                    if self.prefetch and next_window is not None:
                        pending = start((wi + 1) % slots, next_window, carry_x, carry_p)
                    x, p = buf_x[slot], buf_p[slot]
                    for s in range(0, full, batch):
                        sel = perm[s : s + batch]
                        pos = p[sel]
                        yield (
                            x[sel],
                            self.policy[pos],
                            self.value[pos],
                            self.stm[pos],
                            _pick(self.soft, pos),
                        )
                    if not self.prefetch and next_window is not None:
                        pending = start(slot, next_window, carry_x, carry_p)
            finally:
                if pending is not None:
                    pending.join()

        if len(carry_x) and rng is None:
            for s in range(0, len(carry_x), batch):
                pos = carry_p[s : s + batch]
                yield (
                    carry_x[s : s + batch],
                    self.policy[pos],
                    self.value[pos],
                    self.stm[pos],
                    _pick(self.soft, pos),
                )


def make_split(
    shard: Shard,
    which: int,
    stm: np.ndarray,
    name: str,
    *,
    window_blocks: int = WINDOW_BLOCKS,
    block_rows: int = BLOCK_ROWS,
    prefetch: bool = False,
    value_source: str = "engine",
    policy_source: str = "human",
):
    idx_bytes = int((np.asarray(shard.split) == which).sum()) * ROW_BYTES
    if idx_bytes <= RAM_LIMIT_BYTES:
        return RamSplit(
            shard,
            which,
            stm,
            name,
            block_rows=block_rows,
            value_source=value_source,
            policy_source=policy_source,
        )
    return BlockShuffledSplit(
        shard,
        which,
        stm,
        name,
        window_blocks=window_blocks,
        block_rows=block_rows,
        prefetch=prefetch,
        value_source=value_source,
        policy_source=policy_source,
    )
