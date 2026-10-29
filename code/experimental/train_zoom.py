"""分变焦专用模型：只用单个变焦的数据训练。

动机（有硬证据）
----------------
实测三个变焦的 P/R 偏置方向相反：
    X4  P 70.34 / R 86.24  ⇒ 过检（画粗）
    X16 P 80.38 / R 78.73  ⇒ 漏检（画细）

一个模型必须在"画粗"和"画细"之间折中，无法同时最优。
若给每个变焦训独立模型，模型只需学一种宽度分布。

X4 数据量仅 1/3（390 张 → 277 train），训练很快（约 10 min/seed）。

用法
----
    python train_zoom.py --zoom UAV-CrackX4 --seed 101 --out <dir>
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
from torch.utils.data import DataLoader, Dataset

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from clean_manifest import EXCLUDE  # noqa: E402
from data import TRAIN_EXPAND_FACTOR, ZOOM_ID, _augment, _norm_chw, imread  # noqa: E402
from engine import pad_bottom  # noqa: E402
from losses import l0_cal_dice  # noqa: E402
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


class ZoomSet(Dataset):
    """单个变焦的数据集。"""

    def __init__(self, root: Path, zoom: str, names: list[str], *, train: bool) -> None:
        self.root = Path(root)
        self.zoom = zoom
        self.names = list(names)
        self.train = bool(train)
        self.expand = TRAIN_EXPAND_FACTOR if train else 1

    def __len__(self) -> int:
        return len(self.names) * self.expand

    def __getitem__(self, i: int) -> dict:
        n = len(self.names)
        base = int(i) % n
        view = int(i) // n
        name = self.names[base]
        bgr = imread(self.root / "leftImg8bit" / "train" / self.zoom / name, True)
        gt = imread(
            self.root / "gtFine" / "train" / self.zoom / (Path(name).stem + ".png"), False
        )
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.train:
            rgb, gt = _augment(rgb, gt, view_id=view)
        mask = (gt > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(_norm_chw(rgb))),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])),
            "zoom": self.zoom,
            "name": name,
            "zoom_id": ZOOM_ID.get(self.zoom, 1),
        }


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


def eval_counts(model, ldr, device, tau=0.2):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for bt in ldr:
            img = pad_bottom(bt["image"].to(device, non_blocking=True))
            h = bt["image"].shape[-2]
            mask = bt["mask"].to(device, non_blocking=True)
            p = torch.sigmoid(model(img)[:, :, :h])
            for i in range(bt["image"].shape[0]):
                pr = p[i, 0] >= tau
                g = mask[i, 0] > 0.5
                tp += int((pr & g).sum())
                fp += int((pr & ~g).sum())
                fn += int((~pr & g).sum())
                tn += int((~pr & ~g).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "IoU": 100 * iou,
        "P": 100 * prec,
        "R": 100 * rec,
        "F1": 100 * 2 * prec * rec / max(prec + rec, 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", required=True, choices=["UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--full-data", action="store_true", help="用该变焦全部图（不留 holdout）")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = False
    seed_all(int(args.seed))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    split = build_split(CLEAN, reference_root=REF, exclude=EXCLUDE)
    z = args.zoom
    if args.full_data:
        all_names = []
        d = CLEAN / "leftImg8bit" / "train" / z
        for p in sorted(d.glob("*.jpg")):
            if p.name not in EXCLUDE:
                all_names.append(p.name)
        tr_names = all_names
        va_names = []
    else:
        tr_names = split["zooms"][z]["files"]["train"]
        va_names = split["zooms"][z]["files"]["confirm"]

    tr_ds = ZoomSet(CLEAN, z, tr_names, train=True)
    tl = loader(tr_ds, True, int(args.batch), int(args.workers))
    vl = loader(ZoomSet(CLEAN, z, va_names, train=False), False, int(args.batch), int(args.workers)) if va_names else None

    print("=" * 68, flush=True)
    print(
        f"zoom={z} seed={args.seed} full_data={args.full_data} "
        f"train_img={len(tr_names)} val_img={len(va_names)} "
        f"expanded={len(tr_ds)}",
        flush=True,
    )

    model = ResNet34UNet(3, False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    hist = []
    best = (-1.0, 0, None)
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
                loss = l0_cal_dice(logit, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            ls += float(loss.detach()) * img.shape[0]
            n += img.shape[0]
        sched.step()
        row = {"epoch": ep, "loss": ls / max(n, 1), "sec": round(time.time() - t0, 1)}

        if vl is not None:
            m = eval_counts(model, vl, device, 0.5)
            row["val_iou_t0.5"] = m["IoU"]
            print(
                f"[{z[-3:]} s{args.seed}] ep {ep:02d}/{args.epochs}  loss {row['loss']:.4f}  "
                f"val IoU@0.5 {m['IoU']:.4f}  {row['sec']}s",
                flush=True,
            )
            if m["IoU"] > best[0]:
                best = (m["IoU"], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
            print(
                f"[{z[-3:]} s{args.seed}] ep {ep:02d}/{args.epochs}  loss {row['loss']:.4f}  {row['sec']}s",
                flush=True,
            )
        hist.append(row)

    if best[2] is not None:
        model.load_state_dict(best[2])
    ckpt = out / f"zoom_{z}_seed{args.seed}.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)

    payload = {
        "zoom": z,
        "seed": args.seed,
        "full_data": bool(args.full_data),
        "train_img": len(tr_names),
        "val_img": len(va_names),
        "best_epoch": best[1],
        "best_val_iou_t0.5": best[0],
        "history": hist,
        "ckpt": str(ckpt),
    }
    if vl is not None:
        payload["confirm_tau02"] = eval_counts(model, vl, device, 0.2)
        payload["confirm_tau05"] = eval_counts(model, vl, device, 0.5)
        print(f"\n  confirm@0.2 {payload['confirm_tau02']}", flush=True)
    dump(out / f"zoom_{z}_seed{args.seed}_result.json", payload)
    print(f"  saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
