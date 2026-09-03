from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.model import ChessNet, Config, count_params

TOL = 5e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_structure(blob: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    cfg = Config(**blob["config"])
    reference = ChessNet(cfg)
    ref_sd = reference.state_dict()
    got_sd = blob["model"]

    missing = sorted(set(ref_sd) - set(got_sd))
    extra = sorted(set(got_sd) - set(ref_sd))
    mismatched = [
        k
        for k in sorted(set(ref_sd) & set(got_sd))
        if tuple(ref_sd[k].shape) != tuple(got_sd[k].shape)
    ]

    notes.append(f"config              {blob['config']}")
    notes.append(f"tensors in model.py {len(ref_sd)}")
    notes.append(f"tensors in file     {len(got_sd)}")
    notes.append(f"parameters          {count_params(reference):,}")
    if missing:
        notes.append(f"MISSING tensors     {missing[:5]}")
    if extra:
        notes.append(f"UNEXPECTED tensors  {extra[:5]}")
    if mismatched:
        notes.append(f"SHAPE MISMATCH      {mismatched[:5]}")

    marker = "blocks.0.attn.rel_bias"
    if marker in got_sd:
        notes.append(f"geometry bias       {marker} {tuple(got_sd[marker].shape)}")
    else:
        notes.append(f"geometry bias       {marker} ABSENT")

    ok = not missing and not extra and not mismatched and marker in got_sd
    return ok, notes


def check_metrics(blob: dict, shard: Path, batch: int) -> tuple[bool, list[str]]:
    from pipeline.shard import SPLIT_VAL, Shard
    from train.loader import make_split, side_to_move
    from train.train import evaluate

    prov = blob.get("provenance") or {}
    recorded = prov.get("metrics")
    if recorded is None:
        return True, ["no metrics in provenance; skipping reproduction check"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(Config(**blob["config"])).to(dev)
    model.load_state_dict(blob["model"])
    sh = Shard(shard)
    stm = side_to_move(shard, sh.n)
    val = make_split(sh, SPLIT_VAL, stm, "val")
    got = evaluate(model, val, dev, batch)

    notes = []
    ok = True
    for key in ("val_policy_loss", "val_acc", "val_value_mse", "val_acc_white", "val_acc_black"):
        if key not in recorded:
            continue
        delta = abs(got[key] - recorded[key])
        flag = "ok" if delta <= TOL else "MISMATCH"
        if delta > TOL:
            ok = False
        notes.append(f"  {key:18s} recorded {recorded[key]:.6f}  recomputed {got[key]:.6f}  {flag}")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a checkpoint was produced by this repository.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--shard", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print(f"file                {args.checkpoint}")
    print(f"sha256              {sha256(args.checkpoint)}")
    print(f"top-level keys      {sorted(blob)}")
    print()

    print("STRUCTURE  does this file fit train/model.py exactly?")
    ok_struct, notes = check_structure(blob)
    for line in notes:
        print(f"  {line}")
    print(f"  -> {'PASS' if ok_struct else 'FAIL'}")
    print()

    prov = blob.get("provenance")
    print("PROVENANCE  what does the file say produced it?")
    if prov:
        for k in (
            "run",
            "epoch",
            "epochs_planned",
            "seed",
            "value_weight",
            "batch",
            "lr",
            "git_commit",
            "created_utc",
            "torch",
            "host",
        ):
            if k in prov:
                print(f"  {k:18s} {prov[k]}")
        if "shard" in prov:
            for k, v in prov["shard"].items():
                print(f"  shard.{k:12s} {v}")
    else:
        print("  none - checkpoint predates the provenance block")
    print()

    ok_metrics = True
    if args.shard is not None:
        print("REPRODUCTION  does it still produce its recorded metrics?")
        ok_metrics, notes = check_metrics(blob, args.shard, args.batch)
        for line in notes:
            print(line)
        print(f"  -> {'PASS' if ok_metrics else 'FAIL'}")
        print()
    else:
        print("REPRODUCTION  skipped (pass --shard to run it)")
        print()

    verdict = ok_struct and ok_metrics
    print(f"VERDICT  {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
