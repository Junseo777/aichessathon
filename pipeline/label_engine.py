from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from chessml.encoding import encode_move, mirror_move
from pipeline.labels import score_to_value
from pipeline.shard import MULTIPV, SOURCE_STOCKFISH, Shard

SORT_LINES = True

DEFAULT_ENGINE = "/workspace/tools/stockfish/stockfish-ubuntu-x86-64-avx2"
CHUNK = 2000

_engine: chess.engine.SimpleEngine | None = None
_nodes: int | None = 25000
_depth: int | None = None
DEPTH_NODE_CEILING = 150000


def _init_worker(engine_path: str, nodes: int | None, depth: int | None = None) -> None:
    global _engine, _nodes, _depth
    _nodes = nodes
    _depth = depth
    _engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    _engine.configure({"Threads": 1, "Hash": 16})


def _shutdown_worker() -> None:
    global _engine
    if _engine is not None:
        _engine.quit()
        _engine = None


def _limit(nodes: int | None, depth: int | None) -> chess.engine.Limit:
    if depth is not None:
        return chess.engine.Limit(depth=depth, nodes=DEPTH_NODE_CEILING)
    return chess.engine.Limit(nodes=nodes)


def analyse_raw(
    board: chess.Board, nodes: int | None = None, depth: int | None = None
) -> tuple[list[int], list[float], list[int], int]:
    assert _engine is not None
    moves = [-1] * MULTIPV
    values = [float("nan")] * MULTIPV
    depths = [-1] * MULTIPV
    total_nodes = 0
    infos = _engine.analyse(board, _limit(nodes, depth), multipv=MULTIPV)
    if isinstance(infos, dict):
        infos = [infos]
    for k, info in enumerate(infos[:MULTIPV]):
        total_nodes = max(total_nodes, int(info.get("nodes", 0) or 0))
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue
        values[k] = score_to_value(score.pov(board.turn))
        depths[k] = int(info.get("depth", -1) or -1)
        mv = pv[0]
        label = mirror_move(mv) if board.turn == chess.BLACK else mv
        idx = encode_move(label)
        moves[k] = idx if idx >= 0 else -1
    return moves, values, depths, total_nodes


def sort_lines(
    moves: list[int], values: list[float], depths: list[int]
) -> tuple[list[int], list[float], list[int]]:
    order = sorted(
        range(MULTIPV),
        key=lambda k: (moves[k] < 0, -(values[k] if values[k] == values[k] else -1e9)),
    )
    return [moves[k] for k in order], [values[k] for k in order], [depths[k] for k in order]


def analyse_one(board: chess.Board) -> tuple[list[int], list[float], list[int]]:
    moves, values, depths, _ = analyse_raw(board, nodes=_nodes, depth=_depth)
    if SORT_LINES:
        moves, values, depths = sort_lines(moves, values, depths)
    return moves, values, depths


def _work_chunk(
    job: tuple[int, list[str]],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    start, fens = job
    moves = np.full((len(fens), MULTIPV), -1, dtype=np.int16)
    values = np.full((len(fens), MULTIPV), np.nan, dtype=np.float16)
    depths = np.full((len(fens), MULTIPV), -1, dtype=np.int8)
    for i, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
            if not any(board.legal_moves):
                continue
            m, v, d = analyse_one(board)
            moves[i] = m
            values[i] = v
            depths[i] = d
        except Exception:
            continue
    return start, moves, values, depths


def _work_chunk_sweep(
    job: tuple[int, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, fens = job
    n = len(fens)
    moves = np.full((n, MULTIPV), -1, dtype=np.int16)
    values = np.full((n, MULTIPV), np.nan, dtype=np.float32)
    depths = np.full((n, MULTIPV), -1, dtype=np.int16)
    nodes = np.zeros(n, dtype=np.int64)
    for i, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
            if not any(board.legal_moves):
                continue
            m, v, d, nd = analyse_raw(board, nodes=_nodes, depth=_depth)
            moves[i], values[i], depths[i], nodes[i] = m, v, d, nd
        except Exception:
            continue
    return moves, values, depths, nodes


def phase_unique(shard_dir: Path, work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    sh = Shard(shard_dir)
    z = np.asarray(sh.Z)
    print(f"rows {sh.n:,}", flush=True)
    uniq, first = np.unique(z, return_index=True)
    print(f"unique positions {uniq.size:,}  ({100 * uniq.size / sh.n:.1f}% of rows)", flush=True)

    uniq.tofile(work / "unique_z.bin")
    first.astype(np.int64).tofile(work / "unique_row.bin")

    offs = np.asarray(sh.fen_offset)[first]
    order = np.argsort(offs)
    fens: list[bytes] = [b""] * uniq.size
    with open(shard_dir / "fen.txt", "rb") as fh:
        for j in order:
            fh.seek(int(offs[j]))
            fens[j] = fh.readline().strip()
    with open(work / "unique_fen.txt", "wb") as out:
        for f in fens:
            out.write(f + b"\n")

    for name, shape, dtype in (
        ("labels_moves.bin", (uniq.size, MULTIPV), np.int16),
        ("labels_values.bin", (uniq.size, MULTIPV), np.float16),
        ("labels_depth.bin", (uniq.size, MULTIPV), np.int8),
    ):
        p = work / name
        if not p.exists() or p.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            arr = np.memmap(p, dtype=dtype, mode="w+", shape=shape)
            arr[:] = np.nan if np.dtype(dtype).kind == "f" else -1
            arr.flush()
            del arr
    n_chunks = (uniq.size + CHUNK - 1) // CHUNK
    done = work / "done.bin"
    if not done.exists() or done.stat().st_size != n_chunks:
        np.zeros(n_chunks, dtype=np.uint8).tofile(done)
    (work / "unique_meta.json").write_text(
        json.dumps({"n_unique": int(uniq.size), "n_rows": int(sh.n), "chunk": CHUNK}, indent=2)
    )
    return int(uniq.size)


def phase_label(
    work: Path, engine_path: str, nodes: int | None, workers: int, depth: int | None = None
) -> None:
    meta = json.loads((work / "unique_meta.json").read_text())
    n = int(meta["n_unique"])
    fens = (work / "unique_fen.txt").read_bytes().decode("ascii").splitlines()
    assert len(fens) == n, f"fen count {len(fens)} != {n}"

    moves = np.memmap(work / "labels_moves.bin", dtype=np.int16, mode="r+", shape=(n, MULTIPV))
    values = np.memmap(work / "labels_values.bin", dtype=np.float16, mode="r+", shape=(n, MULTIPV))
    depths = np.memmap(work / "labels_depth.bin", dtype=np.int8, mode="r+", shape=(n, MULTIPV))
    done = np.memmap(work / "done.bin", dtype=np.uint8, mode="r+")

    todo = [c for c in range(len(done)) if not done[c]]
    limit = f"depth={depth} (ceiling {DEPTH_NODE_CEILING} nodes)" if depth else f"nodes={nodes}"
    print(
        f"chunks {len(done):,}  remaining {len(todo):,}  workers {workers}  {limit}  "
        f"multipv={MULTIPV}  sort_lines={SORT_LINES}"
    )
    if not todo:
        print("nothing to do")
        return

    jobs = ((c * CHUNK, fens[c * CHUNK : (c + 1) * CHUNK]) for c in todo)
    started = time.time()
    labelled = 0
    first_rate_reported = False

    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(engine_path, nodes, depth)) as pool:
        for start, m, v, d in pool.imap_unordered(_work_chunk, jobs, chunksize=1):
            moves[start : start + len(m)] = m
            values[start : start + len(v)] = v
            depths[start : start + len(d)] = d
            done[start // CHUNK] = 1
            labelled += len(m)
            elapsed = time.time() - started
            if not first_rate_reported and labelled >= 100_000:
                first_rate_reported = True
                print(
                    f"  RATE after {labelled:,}: {labelled / elapsed:,.0f} pos/sec "
                    f"aggregate on {workers} cores",
                    flush=True,
                )
            if labelled % (CHUNK * 50) == 0:
                rate = labelled / max(elapsed, 1e-9)
                left = (n - labelled) / max(rate, 1e-9) / 3600
                print(
                    f"  {labelled:,}/{n:,}  {rate:,.0f}/s  eta {left:.1f}h",
                    flush=True,
                )
    moves.flush()
    values.flush()
    depths.flush()
    done.flush()
    print(f"labelled {labelled:,} in {(time.time() - started) / 60:.1f} min")


def phase_join(shard_dir: Path, work: Path) -> dict:
    meta = json.loads((work / "unique_meta.json").read_text())
    n_unique = int(meta["n_unique"])
    uniq = np.fromfile(work / "unique_z.bin", dtype=np.uint64)
    moves = np.memmap(
        work / "labels_moves.bin", dtype=np.int16, mode="r", shape=(n_unique, MULTIPV)
    )
    values = np.memmap(
        work / "labels_values.bin", dtype=np.float16, mode="r", shape=(n_unique, MULTIPV)
    )
    depths = np.memmap(
        work / "labels_depth.bin", dtype=np.int8, mode="r", shape=(n_unique, MULTIPV)
    )

    sh = Shard(shard_dir, mode="r+")
    added = sh.ensure_columns()
    if added:
        print(f"created missing columns: {added}")
    z = np.asarray(sh.Z)
    pos = np.searchsorted(uniq, z)
    pos = np.clip(pos, 0, n_unique - 1)
    ok = uniq[pos] == z
    labelled = moves[pos, 0] >= 0
    fill = ok & labelled

    sh.Y_policy_engine4[:] = moves[pos]
    sh.Y_value_engine4[:] = values[pos]
    sh.Y_depth_engine4[:] = depths[pos]
    sh.Y_policy_engine[:] = np.where(fill, moves[pos, 0], -1)
    sh.Y_value_engine[:] = np.where(fill, values[pos, 0], np.float16("nan"))
    src = np.asarray(sh.value_source)
    src[fill] = SOURCE_STOCKFISH
    sh.value_source[:] = src
    for name in (
        "Y_policy_engine4",
        "Y_value_engine4",
        "Y_depth_engine4",
        "Y_policy_engine",
        "Y_value_engine",
        "value_source",
    ):
        getattr(sh, name).flush()

    return {
        "rows": int(sh.n),
        "unique": n_unique,
        "rows_labelled": int(fill.sum()),
        "coverage_pct": round(100 * float(fill.mean()), 3),
        "unique_labelled": int((moves[:, 0] >= 0).sum()),
        "unique_coverage_pct": round(100 * float((moves[:, 0] >= 0).mean()), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=Path)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--nodes", type=int, default=25000)
    ap.add_argument("--depth", type=int, default=None, help="use a depth limit instead of nodes")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--phase", choices=["unique", "label", "join", "all"], default="all")
    args = ap.parse_args()

    work = args.work or (args.shard.parent / f"{args.shard.name}_labels")

    if args.phase in ("unique", "all"):
        print("=== phase: unique ===", flush=True)
        phase_unique(args.shard, work)
    if args.phase in ("label", "all"):
        print("=== phase: label ===", flush=True)
        phase_label(work, args.engine, None if args.depth else args.nodes, args.workers, args.depth)
    if args.phase in ("join", "all"):
        print("=== phase: join ===", flush=True)
        report = phase_join(args.shard, work)
        print(json.dumps(report, indent=2))
        (work / "join_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
