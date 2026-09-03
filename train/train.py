from __future__ import annotations

import argparse
import copy
import json
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
from train.loader import make_split, side_to_move
from train.model import ChessNet, Config, count_params


def to_gpu(x: np.ndarray, p: np.ndarray, v: np.ndarray, dev: torch.device):
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True).float().div_(SCALE)
    pt = torch.from_numpy(p).to(dev, non_blocking=True)
    vt = torch.from_numpy(v).to(dev, non_blocking=True)
    return xt, pt, vt


@torch.no_grad()
def evaluate(model: torch.nn.Module, val, dev: torch.device, batch: int) -> dict:
    model.eval()
    tot = correct = 0
    correct_w = n_w = correct_b = n_b = 0
    pol_loss = val_mse = 0.0
    steps = 0
    for x, p, v, stm in val.batches(batch, None):
        xt, pt, vt = to_gpu(x, p, v, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, value = model(xt)
        logits = logits.float()
        value = value.float().squeeze(-1)
        pol_loss += F.cross_entropy(logits, pt).item()
        val_mse += F.mse_loss(value, vt).item()
        hit = (logits.argmax(dim=1) == pt).cpu().numpy()
        correct += int(hit.sum())
        tot += hit.size
        w = stm == 1
        correct_w += int(hit[w].sum())
        n_w += int(w.sum())
        correct_b += int(hit[~w].sum())
        n_b += int((~w).sum())
        steps += 1
    model.train()
    return {
        "val_policy_loss": pol_loss / max(steps, 1),
        "val_value_mse": val_mse / max(steps, 1),
        "val_acc": correct / max(tot, 1),
        "val_acc_white": correct_w / max(n_w, 1),
        "val_acc_black": correct_b / max(n_b, 1),
    }


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


def save(model: torch.nn.Module, cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "model": model.state_dict()}, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", default="run")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
    train = make_split(sh, SPLIT_TRAIN, stm, "train")
    val = make_split(sh, SPLIT_VAL, stm, "val")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = train.n // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=0.0)
    ema = EMA(model, args.ema_decay)

    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    rng = np.random.default_rng(args.seed)
    print(f"steps/epoch {steps_per_epoch:,}  total {total_steps:,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        seen = 0
        run_loss = run_pol = run_val = 0.0
        steps = 0
        for x, p, v, _ in train.batches(args.batch, rng):
            xt, pt, vt = to_gpu(x, p, v, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value = model(xt)
            logits = logits.float()
            value = value.float().squeeze(-1)
            pol = F.cross_entropy(logits, pt)
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
        m_raw = evaluate(model, val, dev, args.batch)
        m_ema = evaluate(ema.shadow, val, dev, args.batch)
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
            "train_val_gap": run_pol / max(steps, 1) - m_raw["val_policy_loss"],
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
        save(model, cfg, args.out / f"{args.run}_e{epoch}.pt")
        save(ema.shadow, cfg, args.out / f"{args.run}_e{epoch}_ema.pt")
        (args.out / f"{args.run}_history.json").write_text(json.dumps(history, indent=2))

    print(json.dumps(history[-1], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
