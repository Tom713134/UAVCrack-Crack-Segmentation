"""方法①：置信度门控细化（confidence-gated thinning）。

针对实测：X4 的 FP 有 72.7% 落在 GT 3px 内（"画粗"），89.1% 在亮区。

做法（区别于此前失败的硬骨架重建）：
    只在**靠近预测边界**的像素上提高阈值，核心区不动。

    Ŷ(x) = 1[P(x) ≥ τ + λ·1{d(x) ≤ r}]

d(x) 为到预测掩膜边界的距离（用距离变换）。
这样：高置信细缝（P 远高于 τ）在边界处仍能通过；
      贴阈值的粗边（P 仅略高于 τ）被削掉。

在 confirm 上验证，扫 r 与 λ 的网格。
"""

from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from clean_manifest import EXCLUDE  # noqa: E402
from data import CrackSet  # noqa: E402
from engine import pad_bottom  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402

CLEAN = Path(os.environ.get("UAV_CLEAN_DIR", "../数据/UAV-CrackX-clean"))
REF = Path(os.environ.get("UAV_REF_DIR", "../数据/UAV-CrackX"))
ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")


def collect(model_list, ds, device, batch=8, workers=8):
    """返回 [(zoom, prob, gt), ...]"""
    ldr = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context="fork" if workers > 0 else None,
    )
    out = []
    with torch.no_grad():
        for bt in ldr:
            img = pad_bottom(bt["image"].to(device, non_blocking=True))
            h = bt["image"].shape[-2]
            mask = bt["mask"].to(device, non_blocking=True)
            acc = None
            for m in model_list:
                p = torch.sigmoid(m(img)[:, :, :h])
                acc = p if acc is None else acc + p
            prob = (acc / len(model_list)).cpu().numpy()
            for i, z in enumerate(bt["zoom"]):
                out.append((z, prob[i, 0], mask[i, 0].cpu().numpy() > 0.5))
    return out


def gated_thin(prob: np.ndarray, tau: float, lam: float, r: int) -> np.ndarray:
    """置信度门控细化。"""
    core = (prob >= tau).astype(np.uint8)
    if core.sum() == 0:
        return core.astype(bool)
    if r <= 0 or lam <= 0:
        return core.astype(bool)

    # 到边界的距离：先求内部距离变换，再找边界（内部但邻域有背景）
    dist_in = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    # 边界像素：内部且距离 <= 1.0
    edge = (core > 0) & (dist_in <= 1.0)

    thr = tau + lam
    keep = (prob >= tau) & (~edge | (prob >= thr))
    return keep


def score(preds, fn) -> dict:
    b = {z: [0, 0, 0, 0] for z in ZOOMS}
    b["all"] = [0, 0, 0, 0]
    for z, p, g in preds:
        pr = fn(p, z)
        for key in (z, "all"):
            b[key][0] += int((pr & g).sum())
            b[key][1] += int((pr & ~g).sum())
            b[key][2] += int((~pr & g).sum())
            b[key][3] += int((~pr & ~g).sum())
    out = {}
    for k, v in b.items():
        tp, fp, fn_, tn = v
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn_, 1)
        iou = tp / max(tp + fp + fn_, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        out[k] = {
            "IoU": round(100 * iou, 4),
            "P": round(100 * prec, 4),
            "R": round(100 * rec, 4),
            "F1": round(100 * f1, 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--part", default="confirm")
    args = ap.parse_args()

    device = torch.device("cuda")
    split = build_split(CLEAN, reference_root=REF, exclude=EXCLUDE)
    ds = CrackSet(CLEAN, split, args.part, train=False)

    models = []
    for p in args.ckpts:
        m = ResNet34UNet(3, False).to(device)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    print(f"K={len(models)}", flush=True)

    preds = collect(models, ds, device, workers=int(args.workers))
    print(f"n={len(preds)}", flush=True)

    payload = {"tau": args.tau, "K": len(models)}
    base = score(preds, lambda p, z: p >= args.tau)
    payload["baseline"] = base
    print(f"\n[base] all {base['all']}")
    for z in ZOOMS:
        print(f"   {z[-3:]} {base[z]}", flush=True)

    best = ("baseline", base["all"]["IoU"])
    print("\n=== 门控细化网格 ===")
    for r in (1, 2):
        for lam in (0.05, 0.10, 0.15, 0.20, 0.30):
            res = score(preds, lambda p, z, r=r, lam=lam: gated_thin(p, args.tau, lam, r))
            tag = f"g_r{r}_lam{lam}"
            payload[tag] = res
            print(
                f"{tag:18s} IoU {res['all']['IoU']:7.4f}  P {res['all']['P']:6.2f} "
                f"R {res['all']['R']:6.2f} | X4 {res['UAV-CrackX4']['IoU']:6.2f} "
                f"X8 {res['UAV-CrackX8']['IoU']:6.2f} X16 {res['UAV-CrackX16']['IoU']:6.2f}",
                flush=True,
            )
            if res["all"]["IoU"] > best[1]:
                best = (tag, res["all"]["IoU"])

    print(f"\nBEST {best}  (baseline {base['all']['IoU']:.4f})")
    payload["best"] = {"tag": best[0], "iou": best[1]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
