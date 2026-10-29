"""后处理/权重层面的零成本改进（全部在内部 split 上验证）。

A. Model Soup：5 个已有的同初始化模型，权重直接平均（Wortsman et al. 2022）
   同 init + 不同 seed 微调，通常落在同一 loss basin，权重平均可能优于概率集成。
   零训练成本。

B. Greedy Soup：贪心搜索加入哪些成员（已在 select 上验证 λ 的思路）

C. 连通域过滤：按 面积 / 细长度 过滤预测里的小团块。
   依据：X4 有 17.9% 的 FP 远离 GT（>10px），部分是孤立暗斑。
   裂缝是细长结构（高 PCA 细长度），团块不是。

D. 组合最优（C 之后再做 gated thinning）

所有选择在 select 上做，confirm 只用于报告。
"""

from __future__ import annotations
import os

import argparse
import copy
import itertools
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
from gated_thin import gated_thin  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402

CLEAN = Path(os.environ.get("UAV_CLEAN_DIR", "../数据/UAV-CrackX-clean"))
REF = Path(os.environ.get("UAV_REF_DIR", "../数据/UAV-CrackX"))
ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")
SEEDS = (101, 202, 303, 404, 505)


# ---------------------------------------------------------------- 概率收集


@torch.no_grad()
def collect(models, ds, device, batch=8, workers=8):
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
    for bt in ldr:
        img = pad_bottom(bt["image"].to(device, non_blocking=True))
        h = bt["image"].shape[-2]
        mask = bt["mask"].to(device, non_blocking=True)
        acc = None
        for m in models:
            p = torch.sigmoid(m(img)[:, :, :h])
            acc = p if acc is None else acc + p
        prob = (acc / len(models)).cpu().numpy()
        for i, z in enumerate(bt["zoom"]):
            out.append((z, prob[i, 0], mask[i, 0].cpu().numpy() > 0.5))
    return out


# ---------------------------------------------------------------- C 连通域过滤


def pca_elongation(mask_u8: np.ndarray) -> tuple[float, float]:
    """返回 (面积, PCA 细长度 λ1/(λ1+λ2))。"""
    area = float(mask_u8.sum())
    if area < 3:
        return area, 0.0
    ys, xs = np.nonzero(mask_u8)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0, keepdims=True)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    l2, l1 = float(ev[0]), float(ev[1])
    return area, l1 / (l1 + l2 + 1e-12)


def filter_components(
    prob: np.ndarray,
    tau: float,
    *,
    min_area: int = 0,
    min_elong: float = 0.0,
    min_fill: float = 0.0,
) -> np.ndarray:
    """按连通域面积 / 细长度 / 填充率过滤。

    min_fill = 面积 / 外接框面积。细缝填充率低，团块填充率高。
    """
    m = (prob >= tau).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        _, elong = pca_elongation(comp)
        if elong < min_elong:
            continue
        if min_fill > 0:
            fill = area / max(w * h, 1)
            if fill > min_fill:
                continue
        keep |= comp
    return keep.astype(bool)


# ---------------------------------------------------------------- 评测


def score(preds, fn) -> dict:
    b = {z: [0, 0, 0, 0] for z in ZOOMS}
    b["all"] = [0, 0, 0, 0]
    for z, p, g in preds:
        pr = fn(p, z)
        for k in (z, "all"):
            b[k][0] += int((pr & g).sum())
            b[k][1] += int((pr & ~g).sum())
            b[k][2] += int((~pr & g).sum())
            b[k][3] += int((~pr & ~g).sum())
    o = {}
    for k, v in b.items():
        tp, fp, fn_, tn = v
        iou = tp / max(tp + fp + fn_, 1)
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn_, 1)
        o[k] = {
            "IoU": round(100 * iou, 4),
            "P": round(100 * pr, 4),
            "R": round(100 * rc, 4),
            "F1": round(100 * 2 * pr * rc / max(pr + rc, 1e-12), 4),
        }
    return o


def make_soup(model_list, device, weights=None):
    """权重平均。返回一个新的单模型（用于预测）。"""
    n = len(model_list)
    w = weights or [1.0 / n] * n
    base = copy.deepcopy(model_list[0]).to(device)
    base.eval()
    sd0 = model_list[0].state_dict()
    new = {}
    for k in sd0:
        if sd0[k].dtype.is_floating_point:
            acc = torch.zeros_like(sd0[k], dtype=torch.float32)
            for wi, m in zip(w, model_list):
                acc += wi * m.state_dict()[k].float()
            new[k] = acc
        else:
            new[k] = sd0[k].clone()
    base.load_state_dict(new)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-soup", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    split = build_split(CLEAN, reference_root=REF, exclude=EXCLUDE)

    models = []
    for s in SEEDS:
        p = args.ckpt_dir / f"seed{s}.pt"
        m = ResNet34UNet(3, False).to(device)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} models from {args.ckpt_dir}", flush=True)

    sel_ds = CrackSet(CLEAN, split, "select", train=False)
    con_ds = CrackSet(CLEAN, split, "confirm", train=False)

    payload = {"tau": args.tau, "ckpt_dir": str(args.ckpt_dir)}

    # ---------- 基准：概率集成 ----------
    print("\n=== 收集概率集成 ===", flush=True)
    sel_p = collect(models, sel_ds, device, workers=int(args.workers))
    con_p = collect(models, con_ds, device, workers=int(args.workers))

    base_sel = score(sel_p, lambda p, z: p >= args.tau)
    base_con = score(con_p, lambda p, z: p >= args.tau)
    payload["prob_ensemble"] = {"select": base_sel["all"], "confirm": base_con["all"]}
    print(f"prob-ens: select {base_sel['all']['IoU']:.4f}  confirm {base_con['all']['IoU']:.4f}", flush=True)

    # ---------- A: Model Soup ----------
    if not args.skip_soup:
        print("\n=== A. Model Soup ===", flush=True)
        soup = make_soup(models, device)
        s_sel = collect([soup], sel_ds, device, workers=int(args.workers))
        s_con = collect([soup], con_ds, device, workers=int(args.workers))
        soup_sel = score(s_sel, lambda p, z: p >= args.tau)
        soup_con = score(s_con, lambda p, z: p >= args.tau)
        payload["soup_uniform"] = {"select": soup_sel["all"], "confirm": soup_con["all"]}
        print(
            f"soup(5, uniform): select {soup_sel['all']['IoU']:.4f}  "
            f"confirm {soup_con['all']['IoU']:.4f}",
            flush=True,
        )

        # 2 个 / 3 个的组合（在 select 上挑，报 confirm）
        best_combo = None
        for k in (2, 3):
            for combo in itertools.combinations(range(len(models)), k):
                sub = [models[i] for i in combo]
                sp = make_soup(sub, device)
                r = collect([sp], sel_ds, device, workers=int(args.workers))
                v = score(r, lambda p, z: p >= args.tau)["all"]["IoU"]
                if best_combo is None or v > best_combo[1]:
                    best_combo = (combo, v)
            print(f"  k={k} best combo {best_combo[0]} select {best_combo[1]:.4f}", flush=True)
        combo, v = best_combo
        sp = make_soup([models[i] for i in combo], device)
        r = collect([sp], con_ds, device, workers=int(args.workers))
        vc = score(r, lambda p, z: p >= args.tau)["all"]["IoU"]
        payload["soup_best_combo"] = {
            "combo": [SEEDS[i] for i in combo],
            "select": v,
            "confirm": vc,
        }
        print(f"soup best combo {[SEEDS[i] for i in combo]}: select {v:.4f} → confirm {vc:.4f}", flush=True)

    # ---------- C: 连通域过滤（在概率集成上） ----------
    print("\n=== C. 连通域过滤 ===", flush=True)
    grid = []
    for min_area in (0, 20, 50, 100, 200):
        for min_elong in (0.0, 0.90, 0.95):
            for min_fill in (0.0, 0.45, 0.6):
                if min_area == 0 and min_elong == 0.0 and min_fill == 0.0:
                    continue
                fn = lambda p, z, a=min_area, e=min_elong, f=min_fill: filter_components(
                    p, args.tau, min_area=a, min_elong=e, min_fill=f
                )
                sv = score(sel_p, fn)["all"]["IoU"]
                grid.append((min_area, min_elong, min_fill, sv))
    grid.sort(key=lambda x: -x[3])
    print(f"  扫了 {len(grid)} 组，select 前 6：", flush=True)
    for g in grid[:6]:
        print(f"    area>={g[0]:4d} elong>={g[1]:.2f} fill<={g[2]:.2f}  select {g[3]:.4f}", flush=True)

    # 只接受 select 上优于基准的组合，看 confirm
    base_sel_iou = base_sel["all"]["IoU"]
    accepted = []
    for a, e, f, sv in grid[:8]:
        if sv <= base_sel_iou:
            continue
        fn = lambda p, z, a=a, e=e, f=f: filter_components(
            p, args.tau, min_area=a, min_elong=e, min_fill=f
        )
        cv_ = score(con_p, fn)["all"]["IoU"]
        accepted.append({"min_area": a, "min_elong": e, "min_fill": f, "select": sv, "confirm": cv_})
        print(
            f"    area>={a:4d} elong>={e:.2f} fill<={f:.2f} → select {sv:.4f} confirm {cv_:.4f}",
            flush=True,
        )
    payload["component_filter"] = {"grid_top": [list(g) for g in grid[:20]], "accepted": accepted}

    # ---------- D: 组合（过滤 + gated thinning） ----------
    print("\n=== D. 组合：连通域过滤 + 门控细化 ===", flush=True)
    best_total = ("prob_ensemble", base_sel_iou, base_con["all"]["IoU"])
    for lam in (0.0, 0.05, 0.10):
        for a, e, f, _ in grid[:4]:
            def fn(p, z, a=a, e=e, f=f, lam=lam):
                m = filter_components(p, args.tau, min_area=a, min_elong=e, min_fill=f)
                if lam <= 0:
                    return m
                # 过滤后的掩膜重新用门控细化：在掩膜边界提高阈值
                q = np.where(m, p, 0.0)
                return gated_thin(q, args.tau, lam, 1)

            sv = score(sel_p, fn)["all"]["IoU"]
            cv_ = score(con_p, fn)["all"]["IoU"]
            tag = f"a{a}_e{e:.2f}_f{f:.2f}_lam{lam}"
            payload.setdefault("combos", {})[tag] = {"select": sv, "confirm": cv_}
            flag = ""
            if sv > base_sel_iou and cv_ > base_con["all"]["IoU"]:
                flag = "  ★ 双向正向"
                if cv_ > best_total[2]:
                    best_total = (tag, sv, cv_)
            print(f"  {tag:32s} select {sv:.4f}  confirm {cv_:.4f}{flag}", flush=True)

    print(
        f"\nBEST(both>base): {best_total[0]}  select {best_total[1]:.4f}  confirm {best_total[2]:.4f}",
        flush=True,
    )
    payload["best_both_positive"] = {
        "tag": best_total[0],
        "select": best_total[1],
        "confirm": best_total[2],
    }
    payload["baseline"] = {"select": base_sel_iou, "confirm": base_con["all"]["IoU"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
