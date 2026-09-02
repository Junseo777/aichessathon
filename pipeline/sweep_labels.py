from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import label_engine
from pipeline.shard import Shard

CONFIGS = [
    ("nodes=25000", 25000, None),
    ("nodes=10000", 10000, None),
    ("depth=10", None, 10),
    ("depth=12", None, 12),
]


def load_fens(shard_dir: Path, n: int, seed: int = 11) -> list[str]:
    sh = Shard(shard_dir)
    idx = np.sort(np.random.default_rng(seed).choice(sh.n, n, replace=False))
    offs = np.asarray(sh.fen_offset)[idx]
    out: list[str] = []
    with open(shard_dir / "fen.txt", "rb") as fh:
        for o in offs:
            fh.seek(int(o))
            out.append(fh.readline().decode().strip())
    return out


def run(fens: list[str], workers: int, nodes: int | None, depth: int | None, engine: str) -> dict:
    chunks = [(j * 500, fens[j * 500 : (j + 1) * 500]) for j in range((len(fens) + 499) // 500)]
    ctx = mp.get_context("fork")
    t = time.time()
    acc_moves: list[np.ndarray] = []
    acc_values: list[np.ndarray] = []
    acc_depths: list[np.ndarray] = []
    acc_nodes: list[np.ndarray] = []
    init_args = (engine, nodes, depth)
    with ctx.Pool(workers, initializer=label_engine._init_worker, initargs=init_args) as pool:
        for m, v, d, nd in pool.imap_unordered(label_engine._work_chunk_sweep, chunks, chunksize=1):
            acc_moves.append(m)
            acc_values.append(v)
            acc_depths.append(d)
            acc_nodes.append(nd)
    dt = time.time() - t
    moves = np.concatenate(acc_moves)
    values = np.concatenate(acc_values)
    depths = np.concatenate(acc_depths)
    ndarr = np.concatenate(acc_nodes)

    present = moves >= 0
    have = present[:, 0]
    mixed = 0
    disagree = 0
    checked = 0
    for i in range(len(moves)):
        ks = [k for k in range(moves.shape[1]) if present[i, k] and not np.isnan(values[i, k])]
        if len(ks) < 2:
            continue
        checked += 1
        ds = {int(depths[i, k]) for k in ks if depths[i, k] >= 0}
        if len(ds) > 1:
            mixed += 1
        vs = [float(values[i, k]) for k in ks]
        if any(vs[j + 1] > vs[j] + 1e-6 for j in range(len(vs) - 1)):
            disagree += 1
    return {
        "pos_per_sec": round(len(moves) / dt, 1),
        "seconds": round(dt, 1),
        "median_nodes": int(np.median(ndarr[ndarr > 0])) if (ndarr > 0).any() else 0,
        "p90_nodes": int(np.percentile(ndarr[ndarr > 0], 90)) if (ndarr > 0).any() else 0,
        "median_depth_line1": int(np.median(depths[have, 0])) if have.any() else -1,
        "mixed_depth_frac": round(mixed / max(checked, 1), 4),
        "order_disagree_frac": round(disagree / max(checked, 1), 4),
        "mean_lines": round(float(present.sum(1).mean()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, default=Path("/workspace/data/shard_8m"))
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--engine", default="/root/sf")
    ap.add_argument("--out", type=Path, default=Path("/workspace/logs/sweep.json"))
    args = ap.parse_args()

    print(f"loading {args.n:,} positions from {args.shard}", flush=True)
    fens = load_fens(args.shard, args.n)
    print(f"loaded {len(fens):,}\n", flush=True)

    rows = []
    hdr = (
        f"{'config':14s} {'wk':>3s} {'pos/s':>8s} {'medNodes':>9s} {'p90Nodes':>9s} "
        f"{'medDepth':>8s} {'mixed':>7s} {'disagree':>9s} {'lines':>6s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for workers in (16, 32):
        for name, nodes, depth in CONFIGS:
            r = run(fens, workers, nodes, depth, args.engine)
            r.update(config=name, workers=workers)
            rows.append(r)
            print(
                f"{name:14s} {workers:3d} {r['pos_per_sec']:8,.0f} {r['median_nodes']:9,d} "
                f"{r['p90_nodes']:9,d} {r['median_depth_line1']:8d} "
                f"{r['mixed_depth_frac']:7.3f} {r['order_disagree_frac']:9.3f} "
                f"{r['mean_lines']:6.2f}",
                flush=True,
            )
            args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
