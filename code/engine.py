"""评估与分变焦阈值扫描（失败分析 §3.3 P0）。"""


from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader

ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")
ZOOM_SHORT = {
    "UAV-CrackX4": "X4",
    "UAV-CrackX8": "X8",
    "UAV-CrackX16": "X16",
}


def pad_bottom(batch: torch.Tensor, height: int = 384) -> torch.Tensor:
    extra = height - batch.shape[-2]
    if extra <= 0:
        return batch
    return torch.nn.functional.pad(batch, (0, 0, 0, extra), mode="reflect")


def _empty() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _rates(bucket: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
    tn = int(bucket.get("tn", 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bg_iou = tn / (tn + fp + fn) if (tn + fp + fn) else 0.0
    miou = 0.5 * (iou + bg_iou)
    total = tp + tn + fp + fn
    aacc = (tp + tn) / total if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "crack_iou": iou,
        "crack_f1": f1,
        "crack_precision": precision,
        "crack_recall": recall,
        "bg_iou": bg_iou,
        "miou": miou,
        "aacc": aacc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _resolve_thr(threshold: float | dict[str, float], zoom: str) -> float:
    if isinstance(threshold, dict):
        if zoom in threshold:
            return float(threshold[zoom])
        short = ZOOM_SHORT.get(zoom, zoom)
        if short in threshold:
            return float(threshold[short])
        return float(threshold.get("all", 0.2))
    return float(threshold)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    threshold: float | dict[str, float],
) -> dict:
    model.eval()
    buckets = {z: _empty() for z in ZOOMS}
    buckets["all"] = _empty()
    for batch in loader:
        image = pad_bottom(batch["image"].to(device, non_blocking=True))
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(image)[:, :, : mask.shape[-2]]
        prob = torch.sigmoid(logits)
        gt = mask > 0.5
        zooms = batch["zoom"]
        for i, zoom in enumerate(zooms):
            thr = _resolve_thr(threshold, zoom)
            p = prob[i, 0] >= thr
            g = gt[i, 0]
            tp = int((p & g).sum())
            fp = int((p & ~g).sum())
            fn = int((~p & g).sum())
            tn = int((~p & ~g).sum())
            for key in (zoom, "all"):
                buckets[key]["tp"] += tp
                buckets[key]["fp"] += fp
                buckets[key]["fn"] += fn
                buckets[key]["tn"] += tn
    return {key: _rates(val) for key, val in buckets.items()}


def board_row(metrics_all: dict) -> dict[str, float]:
    return {
        "mIoU": float(metrics_all["miou"]),
        "Crack F1": float(metrics_all["crack_f1"]),
        "Crack IoU": float(metrics_all["crack_iou"]),
        "Crack Precision": float(metrics_all["crack_precision"]),
        "Crack Recall": float(metrics_all["crack_recall"]),
        "aAcc": float(metrics_all["aacc"]),
    }


@torch.no_grad()
def collect_probs(model, loader: DataLoader, device: torch.device):
    model.eval()
    stored: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        image = pad_bottom(batch["image"].to(device, non_blocking=True))
        mask = batch["mask"]
        prob = torch.sigmoid(model(image)[:, :, : mask.shape[-2]]).cpu()
        for i, zoom in enumerate(batch["zoom"]):
            stored.append((zoom, prob[i, 0], mask[i, 0] > 0.5))
    return stored


def _metrics_at(stored, threshold: float | dict[str, float]) -> dict:
    buckets = {z: _empty() for z in ZOOMS}
    buckets["all"] = _empty()
    for zoom, prob, gt in stored:
        thr = _resolve_thr(threshold, zoom)
        pred = prob >= thr
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        tn = int((~pred & ~gt).sum())
        for key in (zoom, "all"):
            buckets[key]["tp"] += tp
            buckets[key]["fp"] += fp
            buckets[key]["fn"] += fn
            buckets[key]["tn"] += tn
    return {key: _rates(val) for key, val in buckets.items()}


@torch.no_grad()
def sweep_threshold(model, loader: DataLoader, device: torch.device) -> dict:
    stored = collect_probs(model, loader, device)
    best = None
    grid = []
    for step in range(13):
        threshold = 0.2 + 0.05 * step
        metrics = _metrics_at(stored, threshold)
        row = {"threshold": round(threshold, 2), "iou": metrics["all"]["iou"], "metrics": metrics}
        grid.append(row)
        if best is None or row["iou"] > best["iou"]:
            best = row
    return {"best": best, "grid_iou": [{"threshold": r["threshold"], "iou": r["iou"]} for r in grid]}


@torch.no_grad()
def sweep_threshold_per_zoom(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    grids: dict[str, list[float]] | None = None,
) -> dict:
    """对每个 zoom 独立扫 IoU 最优 τ_z，再合并评 all。

    默认搜索方向（相对 CAL 全局 thr≈0.2）：
      X4: 0.18–0.32（抬 thr 压 P 低）
      X8: 0.15–0.30
      X16: 0.10–0.22（降 thr 补 R）
    """
    stored = collect_probs(model, loader, device)
    if grids is None:
        grids = {
            "UAV-CrackX4": [round(0.18 + 0.02 * i, 2) for i in range(8)],
            "UAV-CrackX8": [round(0.15 + 0.025 * i, 2) for i in range(7)],
            "UAV-CrackX16": [round(0.10 + 0.02 * i, 2) for i in range(7)],
        }

    tau_z: dict[str, float] = {}
    per_zoom = {}
    for zoom in ZOOMS:
        best_t, best_iou = None, -1.0
        zoom_grid = []
        for t in grids[zoom]:
            # 仅该 zoom 的样本用 t，其它 zoom 暂用 0.2 占位（不影响本 zoom IoU）
            thr_map = {z: 0.2 for z in ZOOMS}
            thr_map[zoom] = float(t)
            # 只统计本 zoom 桶
            buckets = _empty()
            for z, prob, gt in stored:
                if z != zoom:
                    continue
                pred = prob >= float(t)
                buckets["tp"] += int((pred & gt).sum())
                buckets["fp"] += int((pred & ~gt).sum())
                buckets["fn"] += int((~pred & gt).sum())
                buckets["tn"] += int((~pred & ~gt).sum())
            m = _rates(buckets)
            zoom_grid.append({"threshold": float(t), "iou": m["iou"], "precision": m["precision"], "recall": m["recall"]})
            if m["iou"] > best_iou:
                best_iou, best_t = m["iou"], float(t)
        tau_z[zoom] = float(best_t if best_t is not None else 0.2)
        per_zoom[zoom] = {"best_threshold": tau_z[zoom], "best_iou": best_iou, "grid": zoom_grid}

    merged = _metrics_at(stored, tau_z)
    g_best = None
    g_grid = []
    for step in range(13):
        t = 0.2 + 0.05 * step
        metrics = _metrics_at(stored, t)
        row = {"threshold": round(t, 2), "iou": metrics["all"]["iou"]}
        g_grid.append(row)
        if g_best is None or row["iou"] > g_best["iou"]:
            g_best = row

    return {
        "tau_z": tau_z,
        "per_zoom": per_zoom,
        "merged_metrics": merged,
        "global_best": g_best,
        "global_grid": g_grid,
    }


@torch.no_grad()
def sweep_tau_fine(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    lo: float = 0.03,
    hi: float = 0.70,
    step: float = 0.01,
) -> dict:
    """宽网格逐变焦扫 τ（0.03–0.70，步长 0.01）。

    旧 sweep_threshold_per_zoom 的网格是围绕定稿模型手调的窄范围；
    若新方法改变了概率校准，最优 τ 可能落在网格外，造成不公平比较。
    本函数对每个变焦独立全网格扫描，只在 select 上选，confirm 上评。
    """
    stored = collect_probs(model, loader, device)
    grid = []
    t = float(lo)
    while t <= float(hi) + 1e-9:
        grid.append(round(t, 2))
        t += float(step)

    tau_z: dict[str, float] = {}
    per_zoom = {}
    for zoom in ZOOMS:
        rows = [(p, g) for z, p, g in stored if z == zoom]
        best_t, best_iou = 0.2, -1.0
        curve = []
        for thr in grid:
            tp = fp = fn = tn = 0
            for prob, gt in rows:
                pred = prob >= thr
                tp += int((pred & gt).sum())
                fp += int((pred & ~gt).sum())
                fn += int((~pred & gt).sum())
                tn += int((~pred & ~gt).sum())
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
            curve.append({"threshold": thr, "iou": iou})
            if iou > best_iou:
                best_iou, best_t = iou, thr
        tau_z[zoom] = float(best_t)
        per_zoom[zoom] = {"best_threshold": best_t, "best_iou": best_iou, "curve": curve}

    merged = _metrics_at(stored, tau_z)
    return {"tau_z": tau_z, "per_zoom": per_zoom, "merged_metrics": merged}


def _zoom_curve(rows, grid) -> list[tuple[float, float]]:
    """(threshold, iou) 曲线。"""
    out = []
    for thr in grid:
        tp = fp = fn = 0
        for prob, gt in rows:
            pred = prob >= thr
            tp += int((pred & gt).sum())
            fp += int((pred & ~gt).sum())
            fn += int((~pred & gt).sum())
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        out.append((thr, iou))
    return out


def _smooth(vals: list[float], k: int = 3) -> list[float]:
    if k <= 1:
        return vals
    n = len(vals)
    out = []
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        out.append(float(np.mean(vals[lo:hi])))
    return out


@torch.no_grad()
def sweep_tau_cv(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    lo: float = 0.05,
    hi: float = 0.50,
    step: float = 0.01,
    n_folds: int = 5,
    smooth_k: int = 2,
    seed: int = 0,
) -> dict:
    """抗过拟合的 τ 标定：k-fold 交叉 + 曲线平滑。

    round1 教训：select 仅 51 张/变焦，在 68 个阈值点上取 argmax
    等于挑噪声，扫出的 τ 反而比默认值差 0.28。

    做法：
      1. 把每个变焦的 select 样本分成 n_folds 折；
      2. 在 n-1 折上算 IoU 曲线，**平滑后**取 argmax 作为该折的 τ；
      3. 用该 τ 在留出折上验证，记录 out-of-fold IoU；
      4. 最终 τ = 各折 τ 的中位数（稳健统计）。

    这样 τ 没见过验证的那部分样本，杜绝"在测试集上调参"。
    """
    stored = collect_probs(model, loader, device)
    grid = []
    t = float(lo)
    while t <= float(hi) + 1e-9:
        grid.append(round(t, 2))
        t += float(step)

    rng = np.random.RandomState(int(seed))
    tau_z: dict[str, float] = {}
    detail = {}

    for zoom in ZOOMS:
        rows = [(p.numpy(), g.numpy()) for z, p, g in stored if z == zoom]
        n = len(rows)
        if n == 0:
            tau_z[zoom] = 0.2
            continue
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, min(int(n_folds), n))

        fold_taus, oof = [], []
        for f in range(len(folds)):
            hold = set(folds[f].tolist())
            tr_rows = [rows[i] for i in range(n) if i not in hold]
            va_rows = [rows[i] for i in folds[f].tolist()]
            if not tr_rows or not va_rows:
                continue
            curve = _zoom_curve(tr_rows, grid)
            vals = _smooth([v for _, v in curve], k=int(smooth_k))
            best_i = int(np.argmax(vals))
            best_t = curve[best_i][0]
            fold_taus.append(best_t)
            # out-of-fold 验证
            tp = fp = fn = 0
            for prob, gt in va_rows:
                pred = prob >= best_t
                tp += int((pred & gt).sum())
                fp += int((pred & ~gt).sum())
                fn += int((~pred & gt).sum())
            oof.append((tp, fp, fn))

        if not fold_taus:
            tau_z[zoom] = 0.2
            continue
        tau_med = float(np.median(fold_taus))
        tp = sum(x[0] for x in oof)
        fp = sum(x[1] for x in oof)
        fn = sum(x[2] for x in oof)
        oof_iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        tau_z[zoom] = round(tau_med, 3)
        detail[zoom] = {
            "fold_taus": fold_taus,
            "tau_median": tau_med,
            "oof_iou": oof_iou,
            "n": n,
        }

    merged = _metrics_at(stored, tau_z)
    return {"tau_z": tau_z, "detail": detail, "merged_metrics": merged, "grid": grid}
