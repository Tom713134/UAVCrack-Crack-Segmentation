"""方法②：EMA（指数滑动平均）权重 + Lovász 训练入口。

对每个 run：
  1. 用 lovasz_plus_cal 或 l0 训练
  2. 同时维护 EMA 权重
  3. 训练结束保存两份：last.pt（原始最后 epoch）与 ema.pt
  可后续比较两者，也可把 ema.pt 用于集成。

用法
----
    python train_ema.py --seed 101 --out <dir> --loss lovasz
    python train_ema.py --seed 101 --out <dir> --loss cal
"""

from __future__ import annotations
import os

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from clean_manifest import EXCLUDE  # noqa: E402
from data import TRAIN_EXPAND_FACTOR, CrackSet  # noqa: E402
from engine import evaluate, pad_bottom  # noqa: E402
from losses import l0_cal_dice  # noqa: E402
from lovasz import lovasz_plus_cal  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402

CLEAN = Path(os.environ.get("UAV_CLEAN_DIR", "../数据/UAV-CrackX-clean"))
REF = Path(os.environ.get("UAV_REF_DIR", "../数据/UAV-CrackX"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def loader(ds, train: bool, batch: int, workers: int) -> DataLoader:
    kw = {}
    if workers > 0:
        kw = dict(
            num_workers=workers,
            persistent_workers=True,
            prefetch_factor=4,
            multiprocessing_context="fork",
        )
    return DataLoader(ds, batch_size=batch, shuffle=train, pin_memory=True, **kw)


class EMA:
    """指数滑动平均。β=0.999 时等效平均最近约 1000 步。"""

    def __init__(self, model: torch.nn.Module, beta: float = 0.999) -> None:
        self.beta = float(beta)
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.beta).add_(v.detach().float(), alpha=1.0 - self.beta)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--loss", default="lovasz", choices=["lovasz", "cal"])
    ap.add_argument("--ema-beta", type=float, default=0.999)
    ap.add_argument("--lovasz-weight", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    seed_all(int(args.seed))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    split = build_split(CLEAN, reference_root=REF, exclude=EXCLUDE)
    train_ds = CrackSet(CLEAN, split, "train", train=True)
    select_ds = CrackSet(CLEAN, split, "select", train=False)
    confirm_ds = CrackSet(CLEAN, split, "confirm", train=False)

    tl = loader(train_ds, True, int(args.batch), int(args.workers))
    sl = loader(select_ds, False, int(args.batch), int(args.workers))
    cl = loader(confirm_ds, False, int(args.batch), int(args.workers))

    print("=" * 70, flush=True)
    print(
        f"seed={args.seed} loss={args.loss} ema_beta={args.ema_beta} "
        f"lovasz_w={args.lovasz_weight} epochs={args.epochs}",
        flush=True,
    )
    print(f"train={len(train_ds)} select={len(select_ds)} confirm={len(confirm_ds)}", flush=True)

    model = ResNet34UNet(3, False).to(device)
    ema = EMA(model, beta=float(args.ema_beta))

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    hist = []
    for ep in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        ls, n = 0.0, 0
        for bt in tl:
            img = pad_bottom(bt["image"].to(device, non_blocking=True))
            mask = bt["mask"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logit = model(img)[:, :, : mask.shape[-2]]
                if args.loss == "lovasz":
                    loss = lovasz_plus_cal(
                        logit, mask, lovasz_weight=float(args.lovasz_weight)
                    )
                else:
                    loss = l0_cal_dice(logit, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            ema.update(model)
            ls += float(loss.detach()) * img.shape[0]
            n += img.shape[0]
        sched.step()

        m = evaluate(model, sl, device, 0.5)
        iou = float(m["all"]["iou"])
        hist.append(
            {
                "epoch": ep,
                "loss": ls / max(n, 1),
                "select_iou_t0.5": iou,
                "sec": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{args.loss} s{args.seed}] ep {ep:02d}/{args.epochs}  "
            f"loss {hist[-1]['loss']:.4f}  IoU@0.5 {iou:.4f}  {hist[-1]['sec']}s",
            flush=True,
        )

    # 保存原始与 EMA 两份
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / f"{args.loss}_seed{args.seed}_last.pt")
    torch.save(ema.state_dict(), out / f"{args.loss}_seed{args.seed}_ema.pt")

    # 评测两份
    results = {}
    for tag, state in (("last", model.state_dict()), ("ema", ema.state_dict())):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        sel = evaluate(model, sl, device, 0.2)
        con = evaluate(model, cl, device, 0.2)
        results[tag] = {
            "select": {k: v for k, v in sel.items() if not k.startswith("_")},
            "confirm": {k: v for k, v in con.items() if not k.startswith("_")},
            "confirm_summary": con["_summary"],
        }
        print(f"\n[{tag}] confirm {results[tag]['confirm']['all']}", flush=True)

    dump(
        out / f"{args.loss}_seed{args.seed}_result.json",
        {
            "seed": args.seed,
            "loss": args.loss,
            "ema_beta": args.ema_beta,
            "lovasz_weight": args.lovasz_weight,
            "history": hist,
            "results": results,
        },
    )
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
