"""详细评测：K 模型集成 + 两级后处理，输出六项指标与逐图分布。

用法
----
    python eval_detailed.py \
        --ckpts a.pt b.pt ... \
        --data /path/to/UAV-CrackX \
        --part confirm \
        --tau 0.2 --min-area 50 --thin-lambda 0.05 \
        --out eval_result.json
"""


from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
from clean_manifest import EXCLUDE  # noqa: E402
from data import CrackSet  # noqa: E402
from engine import pad_bottom  # noqa: E402
from make_submission import area_filter, gated_thin  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402

ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")


def load_models(paths: list[Path], device: torch.device) -> list:
    ms = []
    for p in paths:
        m = ResNet34UNet(3, False).to(device)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        ms.append(m)
    return ms


@torch.no_grad()
def collect(models, loader, device, tau, min_area, lam, r) -> list[dict]:
    rows = []
    for bt in loader:
        img = pad_bottom(bt["image"].to(device, non_blocking=True))
        h = bt["image"].shape[-2]
        mask = bt["mask"].to(device, non_blocking=True)

        acc = None
        for m in models:
            p = torch.sigmoid(m(img)[:, :, :h])
            acc = p if acc is None else acc + p
        prob = (acc / len(models)).cpu().numpy()

        for i, z in enumerate(bt["zoom"]):
            pr = prob[i, 0]
            core = (pr >= tau).astype(np.uint8)
            core = area_filter(core, min_area)
            final = gated_thin(pr, core, tau, lam, r)
            g = mask[i, 0].cpu().numpy() > 0.5
            tp = int((final & g).sum())
            fp = int((final & ~g).sum())
            fn = int((~final.astype(bool) & g).sum())
            tn = int((~final.astype(bool) & ~g).sum())
            rows.append({"zoom": z, "name": bt["name"][i], "tp": tp, "fp": fp,
                         "fn": fn, "tn": tn, "gt_px": tp + fn})
    return rows


def agg(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    bg = tn / max(tn + fp + fn, 1)
    return {
        "mIoU": round(100 * 0.5 * (iou + bg), 4),
        "Crack F1": round(100 * f1, 4),
        "Crack IoU": round(100 * iou, 4),
        "Crack Precision": round(100 * prec, 4),
        "Crack Recall": round(100 * rec, 4),
        "aAcc": round(100 * (tp + tn) / max(tp + tn + fp + fn, 1), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None, help="原始根（冻结 confirm 用，可选）")
    ap.add_argument("--part", default="confirm", choices=["train", "select", "confirm"])
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--thin-lambda", type=float, default=0.05)
    ap.add_argument("--thin-r", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models([Path(c) for c in args.ckpts], device)
    print(f"K={len(models)}  part={args.part}  τ={args.tau}  "
          f"min_area={args.min_area}  λ={args.thin_lambda}")

    use_clean = args.ref is not None
    split = build_split(
        args.data,
        reference_root=args.ref if use_clean else None,
        exclude=EXCLUDE if use_clean else None,
    )
    ds = CrackSet(args.data, split, args.part, train=False)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        multiprocessing_context="fork" if args.workers > 0 else None,
    )
    print(f"{args.part} 共 {len(ds)} 张\n")

    rows = collect(models, loader, device, args.tau, args.min_area,
                   args.thin_lambda, args.thin_r)

    out = {"summary_tau02": agg(rows)}
    out["by_zoom_tau02"] = {z: agg([r for r in rows if r["zoom"] == z]) for z in ZOOMS}

    ious = np.array([r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1) for r in rows])
    out["per_image"] = {
        "n": len(rows),
        "mean": round(100 * float(ious.mean()), 4),
        "median": round(100 * float(np.median(ious)), 4),
        "min": round(100 * float(ious.min()), 4),
        "n_fail_lt50": int((ious < 0.5).sum()),
        "worst10_mean": round(100 * float(np.sort(ious)[:10].mean()), 4),
    }

    hdr = f"{'scope':>6} {'mIoU':>8} {'CrackF1':>8} {'CrackIoU':>9} {'Prec':>8} {'Recall':>8} {'aAcc':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k, label in [("all", "all")] + [(z, z[-3:]) for z in ZOOMS]:
        m = out["summary_tau02"] if k == "all" else out["by_zoom_tau02"][k]
        print(f"{label:>6} {m['mIoU']:>8.4f} {m['Crack F1']:>8.4f} {m['Crack IoU']:>9.4f} "
              f"{m['Crack Precision']:>8.4f} {m['Crack Recall']:>8.4f} {m['aAcc']:>8.4f}")

    p = out["per_image"]
    print(f"\n逐图 IoU：均值 {p['mean']:.2f} / 中位 {p['median']:.2f} / "
          f"最差 {p['min']:.2f} / IoU<50 共 {p['n_fail_lt50']} 张")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**out, "per_image_detail": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {args.out}")


if __name__ == "__main__":
    main()
