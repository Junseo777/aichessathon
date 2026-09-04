import argparse
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chessml.encoding import SCALE, encode_move, featurize_int8
from train.export_onnx import export, verify_parity
from train.model import ChessNet, Config

RENAMES = {"embed.weight": "stem.embed.weight"}


def load_capsule(capsule_dir: Path) -> ChessNet:
    cap: dict[str, Any] = json.loads((capsule_dir / "capsule.json").read_text())
    raw = np.fromfile(capsule_dir / cap["weights_file"], dtype=np.float32)
    assert raw.nbytes == cap["weights_bytes"]
    tensors = {
        t["name"]: torch.from_numpy(
            raw[t["offset"] : t["offset"] + t["length"]].reshape(t["shape"]).copy()
        )
        for t in cap["tensors"]
    }
    c = cap["config"]
    assert c["stem_blocks"] == 0 and c["stem_kernel"] == 1 and c["geometry_bias"]
    assert cap["folded_bn"]
    model = ChessNet(
        Config(
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_blocks=c["n_blocks"],
            ffn_mult=c["ffn_mult"],
            value_hidden=c["value_hidden"],
        )
    )
    state: dict[str, torch.Tensor] = {}
    unused = set(tensors)
    for name, param in model.state_dict().items():
        if name.startswith("embed_bn."):
            leaf = name.rsplit(".", 1)[1]
            if leaf == "bias":
                state[name] = tensors["stem.embed.bias"]
                unused.discard("stem.embed.bias")
            elif leaf in ("weight", "running_var"):
                state[name] = torch.ones_like(param)
            elif leaf == "running_mean":
                state[name] = torch.zeros_like(param)
            else:
                state[name] = param
            continue
        source = RENAMES.get(name, name)
        assert tuple(tensors[source].shape) == tuple(param.shape), name
        state[name] = tensors[source]
        unused.discard(source)
    assert not unused, unused
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def sanity(model: ChessNet) -> None:
    for fen in (chess.STARTING_FEN, "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"):
        board = chess.Board(fen)
        x = torch.from_numpy(featurize_int8(board).astype(np.float32) / SCALE)[None]
        with torch.no_grad():
            policy, value = model(x)
        legal = {encode_move(m): m for m in board.legal_moves}
        top = sorted(legal, key=lambda i: -float(policy[0, i]))[:3]
        print(f"{fen}: value {float(value[0]):+.3f}, top {[legal[i].uci() for i in top]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--capsule-dir",
        type=Path,
        default=Path("../neural-chess/viz/public/weights/v3.1-nano"),
    )
    ap.add_argument("--out", type=Path, default=Path("baselines/reference-hero/weights"))
    args = ap.parse_args()
    model = load_capsule(args.capsule_dir)
    sanity(model)
    cap = json.loads((args.capsule_dir / "capsule.json").read_text())
    print(export(model, args.out, args.capsule_dir / cap["weights_file"]))
    verify_parity(model, args.out, strict_int8=False)
    (args.out / "model.int8.onnx").unlink()


if __name__ == "__main__":
    main()
