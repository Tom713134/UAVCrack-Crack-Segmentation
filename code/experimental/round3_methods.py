"""Round 3：由 Round 2 证据推出的三条路线。

G: 多尺度 + 翻转 TTA（推理时，针对 X4 感受野不足）
H: 边缘歧义排除（CIRL, CVPR2024 思想；监督只在高置信区）
I: 温度缩放 + 分变焦 logit 偏置（纯后处理校准，解决 τ 贴地）

G/I 都是推理/后处理，可叠加；H 是训练。
"""


from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from losses import cal_loss, dice_loss


# ================================================================ H
# 边缘歧义排除


def confident_region(
    y01: np.ndarray,
    *,
    gt_dilate: int = 2,
    ambig_band: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """把图分成「确信前景 / 确信背景 / 歧义带」。

    依据 CIRL (CVPR 2024)：裂缝数据集在**边缘非裂缝区**存本质歧义
    （低对比 + 异质纹理），强行用 0/1 监督会损害特征判别性。

    定义：
      W_pos = dilate(Y, gt_dilate)              确信前景（含标注容差）
      W_amb = dilate(Y, ambig_band) \\ W_pos      歧义带（不监督）
      W_neg = 1 − W_amb                          确信背景

    返回 (w_map, amb_mask)，w_map ∈ {0,1} 为监督权重。
    """
    y = (y01 > 0.5).astype(np.uint8)
    if int(y.sum()) == 0:
        ones = np.ones_like(y, dtype=np.float32)
        return ones, np.zeros_like(y, dtype=np.float32)

    k1 = 2 * int(gt_dilate) + 1
    k2 = 2 * int(ambig_band) + 1
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    d_pos = cv2.dilate(y, ker1).astype(bool)
    d_amb = cv2.dilate(y, ker2).astype(bool)
    amb = d_amb & (~d_pos)

    w = np.ones_like(y, dtype=np.float32)
    w[amb] = 0.0
    return w, amb.astype(np.float32)


def masked_cal_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    w_map: torch.Tensor,
    *,
    delta: float = 0.5,
    gamma: float = 0.2,
    lam: float = 0.5,
    dice_w: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """只在确信区计算 CAL + Dice（歧义带权重 0）。"""
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    w = w_map
    dims = (1, 2, 3)

    # 正类 Tversky（加权）
    tp = (p * target * w).sum(dim=dims)
    fn = ((1.0 - p) * target * w).sum(dim=dims)
    fp = (p * (1.0 - target) * w).sum(dim=dims)
    tversky = (tp + eps) / (tp + float(delta) * fn + (1.0 - float(delta)) * fp + eps)
    loss_pos = torch.pow(1.0 - tversky, 1.0 - float(gamma)).mean()

    # 负类（加权，分母按权重归一）
    p_bg = (1.0 - p).clamp(eps, 1.0 - eps)
    neg_map = w * (1.0 - target) * torch.pow(p, float(gamma)) * torch.log(p_bg)
    denom = (w * (1.0 - target)).sum(dim=dims).clamp_min(eps)
    loss_neg = -(neg_map.sum(dim=dims) / denom).mean()

    cal = float(lam) * loss_pos + (1.0 - float(lam)) * loss_neg

    # Dice（加权）
    inter = (p * target * w).sum(dim=dims)
    den = ((p + target) * w).sum(dim=dims)
    dice = (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()

    return cal + float(dice_w) * dice


# ================================================================ G
# 多尺度 + 翻转 TTA（推理）


@torch.no_grad()
def tta_predict(
    model,
    image: torch.Tensor,
    *,
    scales: tuple[float, ...] = (1.0, 1.15, 0.85),
    use_flip: bool = True,
    mode: str = "mean",
) -> torch.Tensor:
    """多尺度 + 翻转 TTA，返回概率图（B,1,H,W）。

    scales: 相对缩放。裂缝是小目标，放大有助于细缝；缩小有助于上下文。
    mode: mean / max / geo
    """
    b, c, h, w = image.shape
    probs = []

    for s in scales:
        if abs(s - 1.0) < 1e-6:
            img = image
        else:
            nh, nw = int(round(h * s)), int(round(w * s))
            img = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)

        variants = [(img, False)]
        if use_flip:
            variants.append((torch.flip(img, dims=[3]), True))

        for inp, flipped in variants:
            out = model(inp)
            out = out[:, :, : inp.shape[-2]]
            if flipped:
                out = torch.flip(out, dims=[3])
            if out.shape[-2:] != (h, w):
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            probs.append(torch.sigmoid(out))

    stack = torch.stack(probs, dim=0)
    if mode == "max":
        return stack.max(dim=0).values
    if mode == "geo":
        return stack.clamp_min(1e-6).pow(1.0 / len(probs)).prod(dim=0)
    return stack.mean(dim=0)


# ================================================================ I
# 温度缩放 + 分变焦 logit 偏置


def fit_temperature(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    grid: tuple[float, ...] = tuple(np.round(np.arange(0.5, 3.01, 0.05), 2)),
) -> float:
    """在给定数据上搜最优温度 T（最小化 BCE）。"""
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        loss = float(F.binary_cross_entropy_with_logits(logits / float(t), target))
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def apply_calibration(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    """P = sigmoid(logits / T + b)。b 用于分变焦平移决策边界。"""
    return torch.sigmoid(logits / float(temperature) + float(bias))


def fit_zoom_bias(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    grid: tuple[float, ...] = tuple(np.round(np.arange(-1.5, 1.51, 0.05), 2)),
) -> float:
    """搜最优 logit 偏置 b（等价于平移阈值）。"""
    best_b, best_loss = 0.0, float("inf")
    for b in grid:
        loss = float(F.binary_cross_entropy_with_logits(logits + float(b), target))
        if loss < best_loss:
            best_loss, best_b = loss, float(b)
    return best_b
