"""终版训练（VLP 提交版）：全部 1177 张训练，不留验证集。

与 run_round5.py 的差异
----------------------
1. split 只有 train，包含全部 1177 张（清洗后），不划 select/confirm
2. 无法早停 ⇒ 固定跑满 EPOCHS，取最后一个 epoch 的权重
   （依据：历史实验 30 epoch 时 select IoU 仍在平台区，未过拟合）
3. 训练结束后直接对 leftImg8bit/val 的 300 张推理，产出 VLP 提交格式

用法
----
    python train_vlp.py --seeds 101,202,303,404,505 --out <dir>
    python train_vlp.py --seeds 101 --out <dir> --epochs 30
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
from data import GEO_CATALOG, TRAIN_EXPAND_FACTOR, ZOOM_ID, _augment, _norm_chw, imread  # noqa: E402
from engine import pad_bottom  # noqa: E402
from losses import l0_cal_dice  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import ZOOMS  # noqa: E402

CLEAN = Path(os.environ.get("UAV_CLEAN_DIR", "../数据/UAV-CrackX-clean"))
REF = Path(os.environ.get("UAV_REF_DIR", "../数据/UAV-CrackX"))

EPOCHS = 30
BATCH = 10
LR = 3e-4


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class FullTrainSet(Dataset):
    """全部有标注图（清洗后 1177 张），几何 ×5，无验证集。"""

    def __init__(self, data_root: Path, *, exclude: frozenset[str] | None = None) -> None:
        self.data_root = Path(data_root)
        excl = frozenset(exclude or ())
        self.samples: list[tuple[str, str]] = []
        for zoom in ZOOMS:
            d = self.data_root / "leftImg8bit" / "train" / zoom
            for p in sorted(d.glob("*.jpg")):
                if p.name in excl:
                    continue
                self.samples.append((zoom, p.name))
        self.expand_factor = TRAIN_EXPAND_FACTOR

    def __len__(self) -> int:
        return len(self.samples) * self.expand_factor

    def __getitem__(self, index: int) -> dict:
        n = len(self.samples)
        base_idx = int(index) % n
        view_id = int(index) // n
        zoom, name = self.samples[base_idx]

        bgr = imread(self.data_root / "leftImg8bit" / "train" / zoom / name, True)
        gt = imread(
            self.data_root / "gtFine" / "train" / zoom / (Path(name).stem + ".png"), False
        )
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb, gt = _augment(rgb, gt, view_id=view_id)
        mask = (gt > 0).astype(np.float32)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(_norm_chw(rgb))),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])),
            "zoom": zoom,
            "name": name,
            "zoom_id": ZOOM_ID.get(zoom, 1),
        }


def make_loader(ds, *, train: bool, batch: int, workers: int) -> DataLoader:
    kw = {}
    if workers > 0:
        kw = dict(
            num_workers=workers,
            persistent_workers=True,
            prefetch_factor=4,
            multiprocessing_context="fork",
        )
    return DataLoader(ds, batch_size=batch, shuffle=train, pin_memory=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,202,303,404,505")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", type=Path, default=CLEAN)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    ds = FullTrainSet(args.data, exclude=EXCLUDE)
    loader = make_loader(ds, train=True, batch=int(args.batch), workers=int(args.workers))
    n_orig = len(ds.samples)

    print("=" * 72, flush=True)
    print(f"VLP 全量训练：原图 {n_orig} 张 × 几何 {TRAIN_EXPAND_FACTOR} = {len(ds)} 样本/epoch", flush=True)
    print(f"epochs={args.epochs} batch={args.batch} lr={args.lr} workers={args.workers}", flush=True)
    print("无验证集：固定跑满，取最后 epoch 权重", flush=True)
    print("=" * 72, flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = []

    for seed in seeds:
        ckpt_path = out / f"seed{seed}.pt"
        if ckpt_path.is_file():
            print(f"[seed{seed}] 已存在，跳过", flush=True)
            summary.append({"seed": seed, "skipped": True})
            continue

        seed_all(seed)
        model = ResNet34UNet(3, False).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=int(args.epochs), eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda")

        hist = []
        for ep in range(1, int(args.epochs) + 1):
            t0 = time.time()
            model.train()
            ls, n = 0.0, 0
            for bt in loader:
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
            hist.append({"epoch": ep, "loss": ls / max(n, 1), "sec": round(time.time() - t0, 1)})
            print(
                f"[seed{seed}] ep {ep:02d}/{args.epochs}  loss {hist[-1]['loss']:.4f}  "
                f"{hist[-1]['sec']}s",
                flush=True,
            )

        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
        dump(out / f"seed{seed}_hist.json", {"seed": seed, "history": hist})
        summary.append(
            {
                "seed": seed,
                "final_loss": hist[-1]["loss"],
                "sec_per_epoch": hist[-1]["sec"],
                "ckpt": str(ckpt_path),
            }
        )
        print(f"[seed{seed}] 完成 → {ckpt_path}", flush=True)

    dump(out / "train_summary.json", {"n_orig": n_orig, "expand": TRAIN_EXPAND_FACTOR, "seeds": summary})
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
