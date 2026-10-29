"""Round 5：多 seed 集成。

理论依据：
  设 K 个独立训练的模型预测概率 p_1..p_K，各自误差 ε_k = p_k − p*，
  且 E[ε_k]=0（无偏）、Var(ε_k)=σ²、彼此相关性 ρ。
  集成平均 p̄ = (1/K)Σp_k 的方差为

      Var(p̄) = σ²·[ρ + (1−ρ)/K]

  当 ρ<1 时随 K 增大单调下降 ⇒ IoU 单调提升（在决策边界附近）。
  这是**唯一不依赖"猜对机制"**的改进方式。

同时注意 Round 3 的教训：集成是**平均**，会继承偏置。若各模型都有
"X4 过检"偏置，集成无法消除它（偏置不随平均消失）。所以：
  - 集成预期提升主要来自方差降低（边界像素判对率）；
  - 不应期望它修好 X4。

实现：
  1. 用 clean 数据 + 不同 seed 训 K 个模型（与 r0_cleanbase 同配置）
  2. 推理时对同一张图的 K 个概率图取均值
  3. 报告单模型 vs 集成的对比
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
sys.path.insert(0, str(CODE))

from clean_manifest import EXCLUDE  # noqa: E402
from data import TRAIN_EXPAND_FACTOR, CrackSet  # noqa: E402
from engine import board_row, evaluate, pad_bottom, sweep_tau_cv  # noqa: E402
from losses import l0_cal_dice  # noqa: E402
from model import ResNet34UNet  # noqa: E402
from split_data import build_split  # noqa: E402
from tau_z import DEFAULT_TAU_Z  # noqa: E402

CLEAN = Path(os.environ.get("UAV_CLEAN_DIR", "../数据/UAV-CrackX-clean"))
REF = Path(os.environ.get("UAV_REF_DIR", "../数据/UAV-CrackX"))
ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")


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


@torch.no_grad()
def ensemble_eval(models, ds, device, tau, batch=8, workers=8) -> dict:
    """K 个模型概率平均后评测。"""
    ldr = loader(ds, False, batch, workers)
    buckets = {z: [0, 0, 0, 0] for z in ZOOMS}
    buckets["all"] = [0, 0, 0, 0]
    for m in models:
        m.eval()
    for bt in ldr:
        img = pad_bottom(bt["image"].to(device, non_blocking=True))
        h = bt["image"].shape[-2]
        mask = bt["mask"].to(device, non_blocking=True)
        acc = None
        for m in models:
            p = torch.sigmoid(m(img)[:, :, :h])
            acc = p if acc is None else acc + p
        prob = acc / len(models)
        for i, zoom in enumerate(bt["zoom"]):
            t = float(tau.get(zoom, 0.2)) if isinstance(tau, dict) else float(tau)
            p = prob[i, 0] >= t
            g = mask[i, 0] > 0.5
            for key in (zoom, "all"):
                buckets[key][0] += int((p & g).sum())
                buckets[key][1] += int((p & ~g).sum())
                buckets[key][2] += int((~p & g).sum())
                buckets[key][3] += int((~p & ~g).sum())
    out = {}
    for k, v in buckets.items():
        tp, fp, fn, tn = v
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        bg = tn / max(tn + fp + fn, 1)
        out[k] = {
            "Crack IoU": pct(iou),
            "P": pct(prec),
            "R": pct(rec),
            "mIoU": pct(0.5 * (iou + bg)),
        }
    return out


def train_one(seed: int, out_dir: Path, args, device) -> Path:
    """训练单个 seed，返回 ckpt 路径。"""
    seed_all(seed)
    split = build_split(CLEAN, reference_root=REF, exclude=EXCLUDE)
    train_ds = CrackSet(CLEAN, split, "train", train=True)
    select_ds = CrackSet(CLEAN, split, "select", train=False)
    train_loader = loader(train_ds, True, int(args.batch), int(args.workers))
    select_loader = loader(select_ds, False, int(args.batch), int(args.workers))

    model = ResNet34UNet(3, False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_iou, best_ep, best_state = -1.0, 0, None
    hist = []
    for ep in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        ls, n = 0.0, 0
        for bt in train_loader:
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
        m = evaluate(model, select_loader, device, 0.5)
        iou = float(m["all"]["iou"])
        hist.append({"epoch": ep, "loss": ls / max(n, 1), "iou": iou, "sec": round(time.time() - t0, 1)})
        print(f"[S{seed}] ep {ep:02d}/{args.epochs} loss {hist[-1]['loss']:.4f} IoU {iou:.4f} {hist[-1]['sec']}s", flush=True)
        if iou > best_iou:
            best_iou, best_ep = iou, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ck = out_dir / f"seed{seed}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ck)

    ck = out_dir / f"seed{seed}.pt"
    if not ck.is_file():
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ck)
    dump(out_dir / f"seed{seed}_hist.json", {"seed": seed, "best_epoch": best_ep, "best_iou": best_iou, "history": hist})
    print(f"[S{seed}] DONE best_ep={best_ep} best_iou={best_iou:.4f}", flush=True)
    return ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "eval"])
    ap.add_argument("--seeds", default="101,202,303,404,505")
    ap.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/uav_upload_20261028/exp_r5_ens"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.mode == "train":
        for s in seeds:
            ck = args.out / f"seed{s}.pt"
            if ck.is_file():
                print(f"[S{s}] exists, skip", flush=True)
                continue
            train_one(s, args.out, args, device)
        print("ALL TRAINED", flush=True)
        return

    # -------- eval：单模型 vs 集成（K=1..N 累积） --------
    split = build_split(CLEAN, reference_root=REF, exclude=EXCLUDE)
    select_ds = CrackSet(CLEAN, split, "select", train=False)
    confirm_ds = CrackSet(CLEAN, split, "confirm", train=False)

    models = []
    for s in seeds:
        ck = args.out / f"seed{s}.pt"
        if not ck.is_file():
            print(f"missing {ck}", flush=True)
            continue
        m = ResNet34UNet(3, False).to(device)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} models", flush=True)

    payload = {"seeds": seeds, "n_models": len(models)}

    # 单模型
    singles = []
    for i, m in enumerate(models):
        r = ensemble_eval([m], confirm_ds, device, 0.2, batch=8, workers=int(args.workers))
        singles.append(r["all"])
        print(f"[single {seeds[i]}] {r['all']}", flush=True)
    payload["singles"] = singles
    if singles:
        ious = [s["Crack IoU"] for s in singles]
        payload["single_mean"] = {
            "Crack IoU": round(float(np.mean(ious)), 4),
            "Crack IoU_std": round(float(np.std(ious)), 4),
        }
        print(f"[single mean] {payload['single_mean']}", flush=True)

    # 累积集成 K=2..N
    for k in range(2, len(models) + 1):
        r = ensemble_eval(models[:k], confirm_ds, device, 0.2, batch=8, workers=int(args.workers))
        payload[f"ens_K{k}"] = r
        print(
            f"[ens K={k}] all {r['all']['Crack IoU']:.4f}  "
            f"X4 {r['UAV-CrackX4']['Crack IoU']:.4f}  "
            f"X8 {r['UAV-CrackX8']['Crack IoU']:.4f}  "
            f"X16 {r['UAV-CrackX16']['Crack IoU']:.4f}",
            flush=True,
        )
        dump(args.out / "ens_result.json", payload)

    # CV-τ 口径的集成
    if len(models) >= 2:
        from engine import collect_probs, sweep_tau_cv

        class Ens(torch.nn.Module):
            def __init__(self, ms):
                super().__init__()
                self.ms = torch.nn.ModuleList(ms)

            def forward(self, x):
                return sum(m(x) for m in self.ms) / len(self.ms)

        em = Ens(models).to(device).eval()
        sl = loader(select_ds, False, 8, int(args.workers))
        cv = sweep_tau_cv(em, sl, device)
        tau_cv = dict(DEFAULT_TAU_Z)
        tau_cv.update({k: float(v) for k, v in cv["tau_z"].items()})
        r = ensemble_eval(models, confirm_ds, device, tau_cv, batch=8, workers=int(args.workers))
        payload["ens_full_tau_cv"] = {"tau_z": tau_cv, "metrics": r}
        print(f"[ens full CV-τ] {r['all']}  tau={tau_cv}", flush=True)

    dump(args.out / "ens_result.json", payload)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
