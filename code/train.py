"""训练脚本：可复现终版中的任意单个模型。

用法
----
    # 训练 1 个模型（seed=101）
    python train.py --data /path/to/UAV-CrackX --out ./exp_seed101 --seed 101

    # 训练全部 5 个 seed（串行；并行请自行分流到多卡/多次调用）
    for s in 101 202 303 404 505; do
        python train.py --data /path/to/UAV-CrackX --out ./exp_seed$s --seed $s
    done

终版 5 个模型除随机种子外配置完全相同。
"""


from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import TRAIN_EXPAND_FACTOR, CrackSet
from engine import evaluate, pad_bottom
from losses import l0_cal_dice
from model import ResNet34UNet
from split_data import build_split

# ---------------- 终版超参（不要随意改动，改动即偏离复现基线） ----------------
EPOCHS = 30
BATCH = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
CAL_DELTA = 0.5
CAL_GAMMA = 0.2
CAL_LAMBDA = 0.5
DICE_WEIGHT = 0.5
GRAD_CLIP = 1.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_loader(ds, *, train: bool, batch: int, workers: int) -> DataLoader:
    kw = {}
    if workers > 0:
        kw = dict(
            num_workers=workers,
            persistent_workers=True,
            prefetch_factor=4,
            multiprocessing_context="fork",
        )
    return DataLoader(
        ds, batch_size=batch, shuffle=train, pin_memory=True, drop_last=False, **kw
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="UAV-CrackX 根目录")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if TRAIN_EXPAND_FACTOR != 5:
        raise RuntimeError(f"需要几何×5，当前 {TRAIN_EXPAND_FACTOR}")
    if not torch.cuda.is_available():
        print("警告：无 CUDA，将用 CPU 训练（极慢）", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_all(int(args.seed))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    split = build_split(args.data)
    train_ds = CrackSet(args.data, split, "train", train=True)
    select_ds = CrackSet(args.data, split, "select", train=False)
    confirm_ds = CrackSet(args.data, split, "confirm", train=False)

    train_loader = make_loader(train_ds, train=True, batch=int(args.batch), workers=int(args.workers))
    select_loader = make_loader(select_ds, train=False, batch=int(args.batch), workers=int(args.workers))
    confirm_loader = make_loader(confirm_ds, train=False, batch=int(args.batch), workers=int(args.workers))

    print("=" * 70)
    print(f"seed={args.seed}  device={device}")
    print(f"split={ {z: split['zooms'][z]['n'] for z in split['zooms']} }")
    print(
        f"train(expanded)={len(train_ds)}  select={len(select_ds)}  confirm={len(confirm_ds)}"
    )
    print(
        f"loss=CAL(δ={CAL_DELTA},γ={CAL_GAMMA},λ={CAL_LAMBDA})+{DICE_WEIGHT}·Dice  "
        f"epochs={args.epochs} batch={args.batch} lr={args.lr}"
    )
    print("=" * 70, flush=True)

    # ---------- 模型与优化器 ----------
    model = ResNet34UNet(3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(args.epochs), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------- 训练 ----------
    history: list[dict] = []
    best_iou, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0

        for batch in train_loader:
            image = pad_bottom(batch["image"].to(device, non_blocking=True))
            mask = batch["mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(image)[:, :, : mask.shape[-2]]
                loss = l0_cal_dice(
                    logits,
                    mask,
                    dice_weight=DICE_WEIGHT,
                    cal_lam=CAL_LAMBDA,
                    cal_delta=CAL_DELTA,
                    cal_gamma=CAL_GAMMA,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()

            loss_sum += float(loss.detach()) * image.shape[0]
            n_seen += image.shape[0]

        sched.step()

        # 早停判据：固定 τ=0.5 的 select IoU（仅用于跨 epoch 选权重）
        m = evaluate(model, select_loader, device, threshold=0.5)
        iou = m["all"]["Crack IoU"] / 100.0
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(n_seen, 1),
            "select_iou_t0.5": iou,
            "sec": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"[seed{args.seed}] epoch {epoch:02d}/{args.epochs}  "
            f"loss {row['loss']:.4f}  select IoU@0.5 {iou:.4f}  {row['sec']}s",
            flush=True,
        )

        if iou > best_iou:
            best_iou, best_epoch = iou, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt = out / f"seed{args.seed}.pt"
            torch.save(best_state, ckpt)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- 最终评测（固定 τ=0.2） ----------
    sel = evaluate(model, select_loader, device, threshold=0.2)
    conf = evaluate(model, confirm_loader, device, threshold=0.2)

    payload = {
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "best_select_iou_t0.5": best_iou,
        "history": history,
        "config": {
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "lr": float(args.lr),
            "weight_decay": WEIGHT_DECAY,
            "cal_delta": CAL_DELTA,
            "cal_gamma": CAL_GAMMA,
            "cal_lambda": CAL_LAMBDA,
            "dice_weight": DICE_WEIGHT,
            "expand_factor": TRAIN_EXPAND_FACTOR,
            "tau_eval": 0.2,
        },
        "split_counts": {z: split["zooms"][z]["n"] for z in split["zooms"]},
        "select": {k: v for k, v in sel.items() if not k.startswith("_")},
        "confirm": {k: v for k, v in conf.items() if not k.startswith("_")},
        "ckpt": str(out / f"seed{args.seed}.pt"),
    }
    dump(out / f"seed{args.seed}_result.json", payload)

    def show(name: str, m: dict) -> None:
        a = m["all"]
        print(
            f"\n{name} @ τ=0.2 (seed {args.seed})\n"
            f"  mIoU {100*a['miou']:.2f}  F1 {100*a['f1']:.2f}  "
            f"IoU {100*a['iou']:.2f}  P {100*a['precision']:.2f}  R {100*a['recall']:.2f}  "
            f"aAcc {100*a['aacc']:.2f}"
        )

    show("select", sel)
    show("confirm", conf)
    print(f"\nbest_epoch={best_epoch}  ckpt={out / f'seed{args.seed}.pt'}")


if __name__ == "__main__":
    main()
