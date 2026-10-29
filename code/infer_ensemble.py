"""终版推理：5-seed 集成（交付用）。

用法：
  python infer_ensemble.py --ckpts a.pt b.pt ... --data <root> --part confirm
  python infer_ensemble.py --ckpts ... --image <path>   # 单图
"""


from __future__ import annotations
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
from data import CrackSet, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from engine import pad_bottom  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402

ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")
TAU = 0.2


class Ensemble:
    """K 个 ResNet34UNet 的概率平均集成。"""

    def __init__(self, ckpts: list[Path], device: torch.device) -> None:
        self.device = device
        self.models = []
        for p in ckpts:
            m = ResNet34UNet(3, False).to(device)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            self.models.append(m)
        if not self.models:
            raise RuntimeError("没有加载到任何模型")

    @torch.no_grad()
    def prob(self, image_chw: torch.Tensor) -> torch.Tensor:
        """image_chw: (B,3,H,W)，已归一化。返回 (B,1,H,W) 平均概率。"""
        h = image_chw.shape[-2]
        x = pad_bottom(image_chw.to(self.device))
        acc = None
        for m in self.models:
            p = torch.sigmoid(m(x)[:, :, :h])
            acc = p if acc is None else acc + p
        return acc / len(self.models)

    @property
    def n(self) -> int:
        return len(self.models)


def norm(rgb_u8: np.ndarray) -> torch.Tensor:
    x = rgb_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1))[None].copy()).float()


def predict_image(ens: Ensemble, rgb_u8: np.ndarray, tau: float = TAU) -> np.ndarray:
    p = ens.prob(norm(rgb_u8))[0, 0].cpu().numpy()
    return (p >= tau).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--part", default="confirm")
    ap.add_argument("--image", type=Path, default=None, help="单图推理，输出 PNG")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = Ensemble([Path(p) for p in args.ckpts], device)
    print(f"ensemble K={ens.n} device={device}", flush=True)

    if args.image is not None:
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(args.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = predict_image(ens, rgb, args.tau)
        out = args.out or args.image.with_suffix(".pred.png")
        cv2.imwrite(str(out), (mask * 255).astype(np.uint8))
        print(f"saved {out}  crack_px={int(mask.sum())}", flush=True)
        return

    if args.data is None:
        raise SystemExit("需要 --data 或 --image")

    split = build_split(args.data, reference_root=args.ref, exclude=EXCLUDE if args.ref else None)
    ds = CrackSet(args.data, split, args.part, train=False)
    ldr = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
        persistent_workers=int(args.workers) > 0,
        multiprocessing_context="fork" if int(args.workers) > 0 else None,
    )

    buckets = {z: [0, 0, 0, 0] for z in ZOOMS}
    buckets["all"] = [0, 0, 0, 0]
    per_img = []

    for bt in ldr:
        prob = ens.prob(bt["image"])
        mask = bt["mask"]
        for i, zoom in enumerate(bt["zoom"]):
            p = prob[i, 0].cpu().numpy() >= args.tau
            g = (mask[i, 0].numpy() > 0.5)
            tp = int((p & g).sum())
            fp = int((p & ~g).sum())
            fn = int((~p & g).sum())
            tn = int((~p & ~g).sum())
            for key in (zoom, "all"):
                buckets[key][0] += tp
                buckets[key][1] += fp
                buckets[key][2] += fn
                buckets[key][3] += tn
            per_img.append(
                {
                    "zoom": zoom,
                    "name": bt["name"][i],
                    "iou": round(tp / max(tp + fp + fn, 1), 4),
                    "p": round(tp / max(tp + fp, 1), 4),
                    "r": round(tp / max(tp + fn, 1), 4),
                    "gt_px": tp + fn,
                }
            )

    out = {}
    for k, v in buckets.items():
        tp, fp, fn, tn = v
        out[k] = {
            "Crack IoU": round(100 * tp / max(tp + fp + fn, 1), 4),
            "P": round(100 * tp / max(tp + fp, 1), 4),
            "R": round(100 * tp / max(tp + fn, 1), 4),
            "mIoU": round(50 * (tp / max(tp + fp + fn, 1) + tn / max(tn + fp + fn, 1)), 4),
        }

    ious = np.array([r["iou"] for r in per_img])
    summary = {
        "ensemble_K": ens.n,
        "tau": args.tau,
        "part": args.part,
        "n_images": len(per_img),
        "micro": out["all"],
        "per_image_mean_iou": round(float(ious.mean()) * 100, 4),
        "per_image_median_iou": round(float(np.median(ious)) * 100, 4),
        "per_image_min_iou": round(float(ious.min()) * 100, 4),
        "n_fail_lt50": int((ious < 0.5).sum()),
        "worst10_mean": round(float(np.sort(ious)[:10].mean()) * 100, 4),
        "by_zoom": {z: out[z] for z in ZOOMS},
        "per_image": per_img,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, ensure_ascii=False, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
