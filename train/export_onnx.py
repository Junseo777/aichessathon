import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train.model import ChessNet, Config, count_params


class Int8GateFailure(AssertionError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(model: ChessNet, out_dir: Path, checkpoint: Path | None = None) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    fp32_path = out_dir / "model.onnx"
    example = torch.zeros(1, 21, 8, 8)
    torch.onnx.export(
        model,
        (example,),
        str(fp32_path),
        input_names=["x"],
        output_names=["policy", "value"],
        dynamic_axes={"x": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    int8_path = out_dir / "model.int8.onnx"
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # The checkpoint entry is what tells two exports of the same architecture apart
    # in the agent's init line, and what lets the tests know a random-init export
    # (checkpoint: null) from a trained one.
    manifest: dict[str, object] = {
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "n_blocks": model.cfg.n_blocks,
        "params": count_params(model),
        "checkpoint": None
        if checkpoint is None
        else {"file": checkpoint.name, "sha256": sha256(checkpoint)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def verify_parity(
    model: ChessNet, out_dir: Path, n: int = 256, seed: int = 0, strict_int8: bool = True
) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.choice([0.0, 0.5, 1.0], size=(n, 21, 8, 8), p=[0.85, 0.05, 0.10])
    x = x.astype(np.float32)

    with torch.no_grad():
        t_policy, t_value = model(torch.from_numpy(x))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(str(out_dir / "model.onnx"), so)
    o_policy, o_value = sess.run(None, {"x": x})
    assert (t_policy.argmax(1).numpy() == o_policy.argmax(1)).all(), "fp32 argmax mismatch"
    verr = float(np.abs(t_value.numpy() - o_value).max())
    assert verr < 1e-4, f"fp32 value error {verr}"

    sess8 = ort.InferenceSession(str(out_dir / "model.int8.onnx"), so)
    q_policy, q_value = sess8.run(None, {"x": x})
    agree = float((t_policy.argmax(1).numpy() == q_policy.argmax(1)).mean())
    q_verr = float(np.abs(t_value.numpy() - q_value).max())
    print(f"parity: fp32 exact, int8 argmax agreement {agree:.3f}, int8 value err {q_verr:.4f}")
    if strict_int8 and agree < 0.99:
        raise Int8GateFailure(f"int8 argmax agreement too low: {agree}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("weights"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.checkpoint is not None:
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config(**blob["config"])
        model = ChessNet(cfg)
        model.load_state_dict(blob["model"])
    else:
        cfg = Config(d_model=args.d_model, n_heads=args.n_heads, n_blocks=args.n_blocks)
        model = ChessNet(cfg)
        print("no checkpoint given: exporting RANDOM weights (runtime testing only)")

    manifest = export(model, args.out, args.checkpoint)
    try:
        verify_parity(model, args.out, strict_int8=args.checkpoint is not None)
    except Int8GateFailure as exc:
        (args.out / "model.int8.onnx").unlink(missing_ok=True)
        print(f"int8 below the gate: {exc}. kept fp32, removed model.int8.onnx")
    except AssertionError:
        for name in ("model.onnx", "model.int8.onnx", "manifest.json"):
            (args.out / name).unlink(missing_ok=True)
        raise
    print(f"exported {manifest['params']:,} params to {args.out}/")


if __name__ == "__main__":
    main()
