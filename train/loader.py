from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from pipeline.shard import Shard

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

BLOCK_ROWS = 262_144
WINDOW_BLOCKS = 16
RAM_LIMIT_BYTES = 12_000_000_000


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


def _labels(shard: Shard, idx: np.ndarray, stm: np.ndarray):
    policy = np.asarray(shard.Y_policy)[idx].astype(np.int64)
    outcome = np.asarray(shard.Y_value)[idx].astype(np.float32)
    engine = np.asarray(shard.Y_value_engine)[idx].astype(np.float32)
    value = np.where(np.isnan(engine), outcome, engine).astype(np.float32)
    labelled = int((~np.isnan(engine)).sum())
    return policy, value, stm[idx], labelled


class RamSplit:
    def __init__(self, shard: Shard, which: int, stm: np.ndarray, name: str) -> None:
        idx = np.flatnonzero(np.asarray(shard.split) == which)
        self.n = idx.size
        print(f"  loading {name}: {self.n:,} rows (ram)", flush=True)
        t = time.time()
        self.x = np.asarray(shard.X)[idx]
        self.policy, self.value, self.stm, labelled = _labels(shard, idx, stm)
        print(
            f"    {self.x.nbytes / 1e9:.1f} GB in {time.time() - t:.0f}s, "
            f"engine-labelled {100 * labelled / max(self.n, 1):.2f}%",
            flush=True,
        )

    def batches(self, batch: int, rng: np.random.Generator | None) -> Iterator[Batch]:
        order = rng.permutation(self.n) if rng is not None else np.arange(self.n)
        stop = self.n - (batch - 1) if rng is not None else self.n
        for s in range(0, stop, batch):
            sel = order[s : s + batch]
            yield self.x[sel], self.policy[sel], self.value[sel], self.stm[sel]


class BlockShuffledSplit:
    def __init__(self, shard: Shard, which: int, stm: np.ndarray, name: str) -> None:
        self.mask = np.asarray(shard.split) == which
        idx = np.flatnonzero(self.mask)
        self.n = idx.size
        self.x_mm = np.asarray(shard.X)
        self.row_of = idx
        self.policy, self.value, self.stm, labelled = _labels(shard, idx, stm)
        self.pos_of = np.full(shard.n, -1, dtype=np.int64)
        self.pos_of[idx] = np.arange(self.n)
        row_bytes = int(np.prod(self.x_mm.shape[1:]))
        nbytes = self.n * row_bytes
        buf_gb = WINDOW_BLOCKS * BLOCK_ROWS * row_bytes / 1e9
        self.blocks = [(s, min(s + BLOCK_ROWS, shard.n)) for s in range(0, shard.n, BLOCK_ROWS)]
        print(
            f"  loading {name}: {self.n:,} rows (block-shuffled memmap, "
            f"{nbytes / 1e9:.1f} GB on disk, "
            f"{buf_gb:.1f} GB buffer)",
            flush=True,
        )
        print(
            f"    engine-labelled {100 * labelled / max(self.n, 1):.2f}%, "
            f"{len(self.blocks):,} blocks of {BLOCK_ROWS:,}",
            flush=True,
        )

    def batches(self, batch: int, rng: np.random.Generator | None) -> Iterator[Batch]:
        order = (
            rng.permutation(len(self.blocks)) if rng is not None else np.arange(len(self.blocks))
        )
        carry_x: list[np.ndarray] = []
        carry_p: list[np.ndarray] = []
        for w in range(0, len(order), WINDOW_BLOCKS):
            xs: list[np.ndarray] = list(carry_x)
            ps: list[np.ndarray] = list(carry_p)
            carry_x, carry_p = [], []
            for b in order[w : w + WINDOW_BLOCKS]:
                lo, hi = self.blocks[b]
                m = self.mask[lo:hi]
                if not m.any():
                    continue
                xs.append(self.x_mm[lo:hi][m])
                ps.append(self.pos_of[lo:hi][m])
            if not xs:
                continue
            buf_x = np.concatenate(xs)
            buf_p = np.concatenate(ps)
            del xs, ps
            if rng is not None:
                perm = rng.permutation(buf_x.shape[0])
                buf_x = buf_x[perm]
                buf_p = buf_p[perm]
            full = (buf_x.shape[0] // batch) * batch
            for s in range(0, full, batch):
                p = buf_p[s : s + batch]
                yield buf_x[s : s + batch], self.policy[p], self.value[p], self.stm[p]
            if full < buf_x.shape[0]:
                carry_x = [buf_x[full:]]
                carry_p = [buf_p[full:]]
        if carry_x and rng is None:
            buf_x = np.concatenate(carry_x)
            buf_p = np.concatenate(carry_p)
            for s in range(0, buf_x.shape[0], batch):
                p = buf_p[s : s + batch]
                yield buf_x[s : s + batch], self.policy[p], self.value[p], self.stm[p]


def make_split(shard: Shard, which: int, stm: np.ndarray, name: str):
    idx_bytes = int((np.asarray(shard.split) == which).sum()) * int(
        np.prod(np.asarray(shard.X).shape[1:])
    )
    if idx_bytes <= RAM_LIMIT_BYTES:
        return RamSplit(shard, which, stm, name)
    return BlockShuffledSplit(shard, which, stm, name)
