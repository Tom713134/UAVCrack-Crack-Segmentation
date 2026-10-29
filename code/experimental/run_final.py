"""统一训练/评测入口（终版战役专用）。

特点：
- 数据：干净集 UAV-CrackX-clean + 冻结 confirm（reference_root 保证可比）
- 评测：分变焦 τ_z 为默认口径（可在 select 上重扫，同时报 confirm）
- 加载：多进程 num_workers（远端 208 核，GPU 利用率是瓶颈）
- 方法：--method 选择损失/增广组合，便于一轮三思路并行

用法示例：
  python run_final.py --method base --out /root/.../exp_r0_base
"""

from __future__ import annotations
import os

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
sys.path.insert(0, str(CODE))

from clean_manifest import EXCLUDE  # noqa: E402
from data import TRAIN_EXPAND_FACTOR, CrackSet  # noqa: E402
from engine import (  # noqa: E402
    board_row,
    evaluate,
    pad_bottom,
    sweep_tau_fine,
)
from losses import l0_cal_dice  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402
from tau_z import DEFAULT_TAU_Z  # noqa: E402

EPOCHS = 30
LR = 3e-4
BATCH = 8
SEED = 42
CAL_DELTA = 0.5

CLEAN_ROOT = Path(os.environ.get("UAV_CLEAN_DIR", "../数据/UAV-CrackX-clean"))
REF_ROOT = Path(os.environ.get("UAV_REF_DIR", "../数据/UAV-CrackX"))

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def pct(x: float) -> float:
    return round(100.0 * float(x), 4)


def make_loader(ds, train: bool, batch: int, workers: int) -> DataLoader:
    kw = {}
    if workers > 0:
        kw = dict(
            num_workers=workers,
            persistent_workers=True,
            prefetch_factor=4,
            multiprocessing_context="fork",
        )
    return DataLoader(
        ds, batch_size=int(batch), shuffle=train, pin_memory=True, drop_last=False, **kw
    )


def build_datasets(
    clean: bool,
    workers: int,
    batch: int,
    curriculum: bool,
    mode: str,
    *,
    data_clean_root: Path | None = None,
    data_ref_root: Path | None = None,
):
    ref = Path(data_ref_root or REF_ROOT)
    data_root = Path(data_clean_root or CLEAN_ROOT) if clean else ref
    split = build_split(
        data_root,
        reference_root=ref if clean else None,
        exclude=EXCLUDE if clean else None,
    )
    common = dict(
        need_skel=False,
        need_hard_masks=bool(curriculum),
        curriculum=bool(curriculum),
        curriculum_mode=mode,
    )
    train_ds = CrackSet(data_root, split, "train", train=True, **common)
    select_ds = CrackSet(data_root, split, "select", train=False)
    confirm_ds = CrackSet(data_root, split, "confirm", train=False)
    return (
        data_root,
        split,
        train_ds,
        make_loader(train_ds, True, batch, workers),
        make_loader(select_ds, False, batch, workers),
        make_loader(confirm_ds, False, batch, workers),
    )


def train_loop(
    model,
    loader,
    select_loader,
    device,
    epochs: int,
    lr: float,
    cal_delta: float,
    *,
    loss_fn=None,
    tag: str = "",
    out_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """通用训练循环；loss_fn(logits, mask, batch) 可选，默认 L0。

    与定稿协议一致：每 epoch 在 select 上评 τ=0.5 IoU，**保存最优权重**。
    """
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_iou, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for batch in loader:
            image = pad_bottom(batch["image"].to(device, non_blocking=True))
            mask = batch["mask"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(image)[:, :, : mask.shape[-2]]
                if loss_fn is None:
                    loss = l0_cal_dice(logits, mask, cal_delta=float(cal_delta))
                else:
                    loss = loss_fn(logits, mask, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            loss_sum += float(loss.detach()) * image.shape[0]
            n_seen += image.shape[0]
        sched.step()

        metrics = evaluate(model, select_loader, device, 0.5)
        iou = float(metrics["all"]["iou"])
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(n_seen, 1),
                "select_iou_t0.5": iou,
                "sec": round(time.time() - t0, 1),
            }
        )
        print(
            f"[{tag}] epoch {epoch:02d}/{epochs}  loss {history[-1]['loss']:.4f}  "
            f"select IoU@0.5 {iou:.4f}  {history[-1]['sec']}s",
            flush=True,
        )
        if iou > best_iou:
            best_iou, best_epoch = iou, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out_dir is not None:
                ck = out_dir / "checkpoints" / "best.pt"
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, ck)

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, {"best_epoch": best_epoch, "best_select_iou_t0.5": best_iou}


def finalize(model, select_loader, confirm_loader, device, out_dir: Path) -> dict:
    """select 上全网格扫 τ_z → confirm 多口径评测。"""
    sweep = sweep_tau_fine(model, select_loader, device)
    tau = dict(DEFAULT_TAU_Z)
    tau.update({k: float(v) for k, v in sweep["tau_z"].items()})
    confirm_z = evaluate(model, confirm_loader, device, tau)
    confirm_g = evaluate(model, confirm_loader, device, 0.2)
    confirm_def = evaluate(model, confirm_loader, device, DEFAULT_TAU_Z)
    select_z = evaluate(model, select_loader, device, tau)
    return {
        "tau_z": tau,
        "tau_z_scanned": sweep["tau_z"],
        "confirm_tau_z": {k: pct(v) for k, v in board_row(confirm_z["all"]).items()},
        "confirm_tau_z_default": {k: pct(v) for k, v in board_row(confirm_def["all"]).items()},
        "confirm_tau02": {k: pct(v) for k, v in board_row(confirm_g["all"]).items()},
        "select_tau_z": {k: pct(v) for k, v in board_row(select_z["all"]).items()},
        "confirm_by_zoom": {
            z: {
                "Crack IoU": pct(confirm_z[z]["crack_iou"]),
                "P": pct(confirm_z[z]["crack_precision"]),
                "R": pct(confirm_z[z]["crack_recall"]),
            }
            for z in ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")
        },
        "sweep": {"tau_z": sweep["tau_z"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-clean", action="store_true", default=True)
    ap.add_argument("--no-clean", dest="data_clean", action="store_false")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cal-delta", type=float, default=CAL_DELTA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--init-ckpt", type=Path, default=None)
    ap.add_argument("--curriculum-mode", default="none")
    ap.add_argument("--data-dir", type=Path, default=None, help="干净数据根")
    ap.add_argument("--ref-dir", type=Path, default=None, help="原始数据根（冻结划分用）")
    args = ap.parse_args()

    if TRAIN_EXPAND_FACTOR != 5:
        raise RuntimeError(f"需要几何×5，当前={TRAIN_EXPAND_FACTOR}")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    seed_all(int(args.seed))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    curriculum = args.curriculum_mode not in ("none", "")
    mode = "shadow" if args.curriculum_mode == "shadow" else "full"

    data_root, split, train_ds, train_loader, select_loader, confirm_loader = build_datasets(
        bool(args.data_clean),
        int(args.workers),
        int(args.batch),
        curriculum,
        mode,
        data_clean_root=args.data_dir,
        data_ref_root=args.ref_dir,
    )
    print("=" * 70, flush=True)
    print(f"method={args.method}  data={data_root}", flush=True)
    print(f"split={ {z: split['zooms'][z]['n'] for z in split['zooms']} }", flush=True)
    print(
        f"epochs={args.epochs} batch={args.batch} lr={args.lr} workers={args.workers} "
        f"expand=×{TRAIN_EXPAND_FACTOR}",
        flush=True,
    )

    model = ResNet34UNet(3, False).to(device)
    if args.init_ckpt is not None and Path(args.init_ckpt).is_file():
        model.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"))
        print("init from", args.init_ckpt, flush=True)

    spec = {"method": args.method, "curriculum_mode": args.curriculum_mode}
    history, best = train_loop(
        model,
        train_loader,
        select_loader,
        device,
        int(args.epochs),
        float(args.lr),
        float(args.cal_delta),
        tag=args.method,
        out_dir=out_dir,
    )

    ckpt = out_dir / "checkpoints" / "final.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)

    report = finalize(model, select_loader, confirm_loader, device, out_dir)
    payload = {
        "experiment": f"final_{args.method}",
        "spec": spec,
        "data_root": str(data_root),
        "data_clean": bool(args.data_clean),
        "split_counts": {z: split["zooms"][z]["n"] for z in split["zooms"]},
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "lr": float(args.lr),
        "workers": int(args.workers),
        "cal_delta": float(args.cal_delta),
        "best": best,
        "history": history,
        "ckpt": str(ckpt),
        **report,
    }
    dump(out_dir / "result.json", payload)

    print("\n" + "=" * 70, flush=True)
    print(f"[{args.method}] τ_z={report['tau_z']}", flush=True)
    print(f"[{args.method}] confirm τ_z  {report['confirm_tau_z']}", flush=True)
    print(f"[{args.method}] confirm τ0.2 {report['confirm_tau02']}", flush=True)
    print(f"[{args.method}] by zoom    {report['confirm_by_zoom']}", flush=True)


if __name__ == "__main__":
    main()
