from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.shard import SPLIT_TRAIN, SPLIT_VAL, Shard, ShardWriter
from train.loader import BlockShuffledSplit, RamSplit, side_to_move

N = 1000
BLOCK = 37
WINDOW = 3
BATCH = 16


def row_id(x: np.ndarray) -> int:
    return int(np.ascontiguousarray(x.reshape(-1)[:4]).view(np.int32)[0])


@pytest.fixture(scope="module")
def shard(tmp_path_factory: pytest.TempPathFactory) -> Shard:
    root: Path = tmp_path_factory.mktemp("shard")
    with ShardWriter(root) as writer:
        for i in range(N):
            x = np.zeros((21, 8, 8), dtype=np.int8)
            x.reshape(-1)[:4] = np.array([i], dtype=np.int32).view(np.int8)
            writer.append(
                x=x,
                y_policy=i,
                y_value=i % 5 - 2,
                z=i,
                fen=f"8/8/8/8/8/8/8/8 {'w' if i % 2 else 'b'} - - 0 1",
                split=SPLIT_VAL if i % 7 == 0 else SPLIT_TRAIN,
                y_value_engine=float("nan") if i % 2 == 0 else float(i % 3 - 1),
            )
        writer.write_meta({})
    return Shard(root)


def expected_labels(shard: Shard, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    engine = np.asarray(shard.Y_value_engine)[rows].astype(np.float32)
    outcome = np.asarray(shard.Y_value)[rows].astype(np.float32)
    return np.where(np.isnan(engine), outcome, engine), (rows % 2).astype(np.int8)


def drain(split: BlockShuffledSplit | RamSplit, rng: np.random.Generator | None):
    ids: list[int] = []
    policy: list[int] = []
    value: list[float] = []
    stm: list[int] = []
    sizes: list[int] = []
    for x, p, v, s in split.batches(BATCH, rng):
        ids.extend(row_id(row) for row in x)
        policy.extend(p.tolist())
        value.extend(v.tolist())
        stm.extend(s.tolist())
        sizes.append(len(p))
    return np.array(ids), np.array(policy), np.array(value), np.array(stm), sizes


def test_ordered_epoch_is_the_split_in_row_order(shard: Shard) -> None:
    stm = side_to_move(shard.root, shard.n)
    split = BlockShuffledSplit(shard, SPLIT_TRAIN, stm, "t", window_blocks=WINDOW, block_rows=BLOCK)
    ids, policy, value, side, sizes = drain(split, None)
    np.testing.assert_array_equal(ids, split.row_of)
    np.testing.assert_array_equal(policy, split.row_of)
    want_value, want_stm = expected_labels(shard, split.row_of)
    np.testing.assert_array_equal(value, want_value)
    np.testing.assert_array_equal(side, want_stm)
    assert sizes[:-1] == [BATCH] * (len(sizes) - 1) and 0 < sizes[-1] <= BATCH


def test_shuffled_epoch_visits_every_row_once_and_keeps_labels_aligned(shard: Shard) -> None:
    stm = side_to_move(shard.root, shard.n)
    split = BlockShuffledSplit(shard, SPLIT_TRAIN, stm, "t", window_blocks=WINDOW, block_rows=BLOCK)
    ids, policy, value, side, sizes = drain(split, np.random.default_rng(0))
    assert sizes == [BATCH] * (split.n // BATCH)
    assert len(set(ids.tolist())) == len(ids) == split.n - split.n % BATCH
    assert set(ids.tolist()) <= set(split.row_of.tolist())
    np.testing.assert_array_equal(policy, ids)
    want_value, want_stm = expected_labels(shard, ids)
    np.testing.assert_array_equal(value, want_value)
    np.testing.assert_array_equal(side, want_stm)
    assert not np.array_equal(ids, np.sort(ids))


def test_shuffle_is_seeded_and_differs_between_epochs(shard: Shard) -> None:
    stm = side_to_move(shard.root, shard.n)
    split = BlockShuffledSplit(shard, SPLIT_TRAIN, stm, "t", window_blocks=WINDOW, block_rows=BLOCK)
    rng = np.random.default_rng(5)
    first = drain(split, rng)[0]
    second = drain(split, rng)[0]
    again = drain(split, np.random.default_rng(5))[0]
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, again)


def test_prefetch_yields_the_same_batches(shard: Shard) -> None:
    stm = side_to_move(shard.root, shard.n)
    plain = BlockShuffledSplit(shard, SPLIT_TRAIN, stm, "t", window_blocks=WINDOW, block_rows=BLOCK)
    ahead = BlockShuffledSplit(
        shard, SPLIT_TRAIN, stm, "t", window_blocks=WINDOW, block_rows=BLOCK, prefetch=True
    )
    for a, b in zip(
        plain.batches(BATCH, np.random.default_rng(1)),
        ahead.batches(BATCH, np.random.default_rng(1)),
        strict=True,
    ):
        for got, want in zip(a, b, strict=True):
            np.testing.assert_array_equal(got, want)


def test_early_exit_leaves_no_thread_behind(shard: Shard) -> None:
    import threading

    stm = side_to_move(shard.root, shard.n)
    split = BlockShuffledSplit(
        shard, SPLIT_TRAIN, stm, "t", window_blocks=WINDOW, block_rows=BLOCK, prefetch=True
    )
    before = threading.active_count()
    for _ in split.batches(BATCH, np.random.default_rng(2)):
        break
    assert threading.active_count() == before


def test_ram_split_reads_the_rows_of_its_split(shard: Shard) -> None:
    stm = side_to_move(shard.root, shard.n)
    split = RamSplit(shard, SPLIT_VAL, stm, "v", block_rows=BLOCK)
    rows = np.flatnonzero(np.asarray(shard.split) == SPLIT_VAL)
    assert split.n == len(rows)
    assert [row_id(x) for x in split.x] == rows.tolist()
    ids, policy, value, side, _ = drain(split, None)
    np.testing.assert_array_equal(ids, rows)
    np.testing.assert_array_equal(policy, rows)
    want_value, want_stm = expected_labels(shard, rows)
    np.testing.assert_array_equal(value, want_value)
    np.testing.assert_array_equal(side, want_stm)
