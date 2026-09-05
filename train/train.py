from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chessml.encoding import SCALE
from pipeline.shard import SPLIT_TRAIN, SPLIT_VAL, Shard
from train.loader import (
    POLICY_SOURCES,
    VALUE_SOURCES,
    WINDOW_BLOCKS,
    SoftLabels,
    make_split,
    side_to_move,
)
from train.model import NUM_MOVES, ChessNet, Config, count_params

SoftTensors = tuple[torch.Tensor, torch.Tensor]  # MultiPV moves (B, 4) long, values (B, 4) float


def to_gpu(x: np.ndarray, p: np.ndarray, v: np.ndarray, soft: SoftLabels | None, dev: torch.device):
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True).float().div_(SCALE)
    pt = torch.from_numpy(p).to(dev, non_blocking=True)
    vt = torch.from_numpy(v).to(dev, non_blocking=True)
    st: SoftTensors | None = None
    if soft is not None:
        st = (
            torch.from_numpy(soft[0]).to(dev, non_blocking=True).long(),
            torch.from_numpy(soft[1]).to(dev, non_blocking=True).float(),
        )
    return xt, pt, vt, st


def policy_target(
    human: torch.Tensor, soft: SoftTensors | None, alpha: float, temperature: float
) -> torch.Tensor:
    """The policy distribution one batch is trained towards, (B, NUM_MOVES), rows summing to 1.

    human        the move played, (B,)
    soft         the MultiPV lines: moves (B, 4) with -1 where absent, values (B, 4) on the
                 [-1, 1] scale with NaN where absent; or None
    alpha        the human move's share on rows that have engine lines; the lines share the rest,
                 softmaxed over value / temperature. Rows without a line, and every row when soft
                 is None, are one-hot on the human move.
    """
    n = human.size(0)
    rows = torch.arange(n, device=human.device)
    target = torch.zeros(n, NUM_MOVES, device=human.device)
    if soft is None:
        target[rows, human] = 1.0
        return target
    moves, values = soft
    present = moves >= 0
    has = present.any(dim=1)
    scores = torch.where(present, values, torch.full_like(values, float("-inf"))) / temperature
    weight = torch.softmax(scores, dim=1)  # NaN on rows with no line, 0 on absent entries
    weight = torch.where(present, weight, torch.zeros_like(weight))
    share = torch.where(
        has, torch.full((n,), alpha, device=human.device), torch.ones(n, device=human.device)
    )
    target.scatter_add_(1, moves.clamp(min=0), (1.0 - share).unsqueeze(1) * weight)
    target[rows, human] += share
    return target


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, val, dev: torch.device, batch: int, alpha: float, temperature: float
) -> dict:
    """val_policy_loss and val_acc are always against the human move, so runs stay comparable;
    under a soft policy source val_target_loss is against the training target and
    val_acc_engine against the engine's best line."""
    model.eval()
    tot = correct = 0
    correct_w = n_w = correct_b = n_b = 0
    pol_loss = val_mse = target_loss = 0.0
    engine_hit = engine_n = 0
    steps = 0
    soft_seen = False
    for x, p, v, stm, soft in val.batches(batch, None):
        xt, pt, vt, st = to_gpu(x, p, v, soft, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, value = model(xt)
        logits = logits.float()
        value = value.float().squeeze(-1)
        pol_loss += F.cross_entropy(logits, pt).item()
        val_mse += F.mse_loss(value, vt).item()
        top = logits.argmax(dim=1)
        if st is not None:
            soft_seen = True
            target_loss += F.cross_entropy(logits, policy_target(pt, st, alpha, temperature)).item()
            best = st[0][:, 0]
            labelled = best >= 0
            engine_hit += int((top[labelled] == best[labelled]).sum())
            engine_n += int(labelled.sum())
        hit = (top == pt).cpu().numpy()
        correct += int(hit.sum())
        tot += hit.size
        w = stm == 1
        correct_w += int(hit[w].sum())
        n_w += int(w.sum())
        correct_b += int(hit[~w].sum())
        n_b += int((~w).sum())
        steps += 1
    model.train()
    out = {
        "val_policy_loss": pol_loss / max(steps, 1),
        "val_value_mse": val_mse / max(steps, 1),
        "val_acc": correct / max(tot, 1),
        "val_acc_white": correct_w / max(n_w, 1),
        "val_acc_black": correct_b / max(n_b, 1),
    }
    if soft_seen:
        out["val_target_loss"] = target_loss / max(steps, 1)
        out["val_acc_engine"] = engine_hit / max(engine_n, 1)
    return out


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            s.copy_(b)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def shard_fingerprint(shard_dir: Path) -> dict:
    meta = shard_dir / "meta.json"
    raw = meta.read_bytes() if meta.exists() else b""
    return {
        "path": str(shard_dir),
        "meta_sha256": hashlib.sha256(raw).hexdigest() if raw else "missing",
        "rows": json.loads(raw)["n_samples"] if raw else -1,
    }


def save(model: torch.nn.Module, cfg: Config, path: Path, provenance: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"config": asdict(cfg), "model": model.state_dict()}
    if provenance is not None:
        blob["provenance"] = provenance
    torch.save(blob, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", default="run")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument(
        "--value-source",
        choices=VALUE_SOURCES,
        default="engine",
        help="value target: engine = Stockfish label where present else outcome (R2-R5); "
        "lichess = the [%%eval] where present else outcome (R1's target); outcome = the "
        "game result only; blend = 0.5 engine + 0.5 outcome where the engine label exists",
    )
    ap.add_argument(
        "--policy-source",
        choices=POLICY_SOURCES,
        default="human",
        help="policy target: human = the move played, one-hot (R0-R7); multipv = the Stockfish "
        "MultiPV lines softmaxed over value / temperature where they exist, the human move "
        "elsewhere (R8); mix = alpha x human + (1 - alpha) x multipv where lines exist (R8b)",
    )
    ap.add_argument(
        "--policy-alpha",
        type=float,
        default=0.5,
        help="the human move's share under --policy-source mix (multipv is alpha 0)",
    )
    ap.add_argument(
        "--policy-temperature",
        type=float,
        default=0.05,
        help="softmax temperature over the MultiPV values on the [-1, 1] scale; at 0.05 the second "
        "line carries 0.6x the first's mass at the shard's median gap of 0.024 and 0.03x at the "
        "90th-percentile gap of 0.18",
    )
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--window-blocks",
        type=int,
        default=WINDOW_BLOCKS,
        help="blocks of 262,144 rows shuffled together per window; 16 is a 5.6 GB window "
        "and the resident footprint of the loader (plus 0.4 GB staging)",
    )
    ap.add_argument(
        "--prefetch",
        action="store_true",
        help="read the next window in a background thread while the GPU trains on this "
        "one; costs a second window of RAM",
    )
    args = ap.parse_args()
    if not 0.0 <= args.policy_alpha <= 1.0:
        ap.error("--policy-alpha must lie in [0, 1]")
    if args.policy_temperature <= 0.0:
        ap.error("--policy-temperature must be positive")
    alpha = {"human": 1.0, "multipv": 0.0, "mix": args.policy_alpha}[args.policy_source]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    cfg = Config(d_model=args.d_model, n_heads=args.n_heads, n_blocks=args.n_blocks)
    model = ChessNet(cfg).to(dev)
    print(f"model d{cfg.d_model} x b{cfg.n_blocks}: {count_params(model):,} params", flush=True)

    sh = Shard(args.shard)
    stm = side_to_move(args.shard, sh.n)
    print(f"shard {args.shard} n={sh.n:,}", flush=True)
    train = make_split(
        sh,
        SPLIT_TRAIN,
        stm,
        "train",
        window_blocks=args.window_blocks,
        prefetch=args.prefetch,
        value_source=args.value_source,
        policy_source=args.policy_source,
    )
    val = make_split(
        sh, SPLIT_VAL, stm, "val", value_source=args.value_source, policy_source=args.policy_source
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = train.n // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=0.0)
    ema = EMA(model, args.ema_decay)

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    rng = np.random.default_rng(args.seed)
    commit = git_commit()
    fingerprint = shard_fingerprint(args.shard)
    policy_desc = args.policy_source
    if args.policy_source != "human":
        policy_desc += f" (alpha {alpha}, temperature {args.policy_temperature})"
    print(
        f"git {commit[:12]}  shard meta sha256 {fingerprint['meta_sha256'][:16]}  "
        f"value source {args.value_source}  policy source {policy_desc}",
        flush=True,
    )
    print(f"steps/epoch {steps_per_epoch:,}  total {total_steps:,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        seen = 0
        run_loss = run_pol = run_val = 0.0
        steps = 0
        for x, p, v, _, soft in train.batches(args.batch, rng):
            xt, pt, vt, st = to_gpu(x, p, v, soft, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value = model(xt)
            logits = logits.float()
            value = value.float().squeeze(-1)
            if st is None:
                pol = F.cross_entropy(logits, pt)
            else:
                pol = F.cross_entropy(logits, policy_target(pt, st, alpha, args.policy_temperature))
            vl = F.mse_loss(value, vt)
            loss = pol + args.value_weight * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            run_loss += loss.item()
            run_pol += pol.item()
            run_val += vl.item()
            steps += 1
            seen += len(p)
            if steps == 50 or steps % 2000 == 0:
                print(
                    f"  e{epoch} step {steps:,}/{steps_per_epoch:,} "
                    f"loss {run_loss / steps:.4f} pol {run_pol / steps:.4f} "
                    f"val {run_val / steps:.4f} {seen / (time.time() - t0):,.0f} samples/s",
                    flush=True,
                )

        secs = time.time() - t0
        m_raw = evaluate(model, val, dev, args.batch, alpha, args.policy_temperature)
        m_ema = evaluate(ema.shadow, val, dev, args.batch, alpha, args.policy_temperature)
        row = {
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train_loss": run_loss / max(steps, 1),
            "train_policy_loss": run_pol / max(steps, 1),
            "train_value_mse": run_val / max(steps, 1),
            "samples_per_sec": round(seen / secs),
            "seconds": round(secs),
            **m_raw,
            **{f"ema_{k}": v for k, v in m_ema.items()},
            "train_val_gap": run_pol / max(steps, 1)
            - m_raw.get("val_target_loss", m_raw["val_policy_loss"]),
        }
        history.append(row)
        print(
            f"[{args.run}] epoch {epoch}: "
            f"train_pol {row['train_policy_loss']:.4f} val_pol {row['val_policy_loss']:.4f} "
            f"gap {row['train_val_gap']:+.4f} | acc {row['val_acc']:.4f} "
            f"(W {row['val_acc_white']:.4f} / B {row['val_acc_black']:.4f}) | "
            f"val_mse {row['val_value_mse']:.4f} | ema_acc {row['ema_val_acc']:.4f} | "
            f"{row['samples_per_sec']:,}/s",
            flush=True,
        )
        prov = {
            "run": args.run,
            "epoch": epoch,
            "epochs_planned": args.epochs,
            "seed": args.seed,
            "value_weight": args.value_weight,
            "value_source": args.value_source,
            "policy_source": args.policy_source,
            "policy_alpha": alpha,
            "policy_temperature": args.policy_temperature,
            "batch": args.batch,
            "lr": args.lr,
            "git_commit": commit,
            "shard": fingerprint,
            "metrics": row,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "host": platform.node(),
        }
        save(model, cfg, args.out / f"{args.run}_e{epoch}.pt", prov)
        save(ema.shadow, cfg, args.out / f"{args.run}_e{epoch}_ema.pt", {**prov, "ema": True})
        (args.out / f"{args.run}_history.json").write_text(json.dumps(history, indent=2))

    print(json.dumps(history[-1], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
