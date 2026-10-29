"""Round 2 三思路：针对 round1 证据修正。

D: X4 显式难负挖掘（暗纹理 → 负类，特征层面干预）
E: 骨架召回 + X4 精度约束配对（B 的修正）
F: 半监督修正版（前景平衡伪标签，解决 98% 背景吞没）
"""


from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from losses import cal_loss, dice_loss

ZOOMS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")


# ================================================================ D


def dark_texture_negative(
    rgb_u8: np.ndarray,
    y01: np.ndarray,
    *,
    dark_q: float = 0.20,
    lowc_q: float = 0.30,
    r_s: int = 15,
    dilate_gt: int = 3,
) -> np.ndarray:
    """X4 难负掩膜：暗 AND 低对比 AND 远离 GT 的像素。

    动机（round1 A 的教训）：X4 过检的根源是"看到暗就画"，
    即把车辙 / 修补带 / 树影当作缝。这是**特征混淆**，
    调损失权重无效，必须在像素级告诉模型"这些暗但不是缝"。

    M_hn = M_dark ∧ M_lowC ∧ ¬dilate(Y)
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    thr = float(np.quantile(gray, float(dark_q)))
    m_dark = gray <= thr

    k = int(r_s) * 2 + 1
    mean = cv2.blur(gray, (k, k))
    c = np.abs(gray - mean)
    c_thr = float(np.quantile(c, float(lowc_q)))
    m_lowc = c <= c_thr

    gt = (y01 > 0.5).astype(np.uint8)
    if int(gt.sum()) > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(dilate_gt) + 1, 2 * int(dilate_gt) + 1)
        )
        near_gt = cv2.dilate(gt, ker).astype(bool)
    else:
        near_gt = np.zeros_like(gt, dtype=bool)

    m = (m_dark & m_lowc & (~near_gt)).astype(np.float32)
    return m


def hard_negative_loss(
    logits: torch.Tensor,
    m_hn: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L_hn：只罚难负掩膜上的概率（压制暗纹理假阳）。"""
    p = torch.sigmoid(logits)
    numer = (p * m_hn).sum()
    denom = m_hn.sum().clamp_min(eps)
    return numer / denom


def x4_negative_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    m_hn: torch.Tensor,
    zoom_ids: torch.Tensor,
    *,
    beta_x4: float = 0.35,
    beta_other: float = 0.05,
    dice_w: float = 0.5,
    cal_delta: float = 0.5,
    cal_gamma: float = 0.2,
    cal_lam: float = 0.5,
) -> torch.Tensor:
    """L = L0 + β_z · L_hn，X4 大权重。"""
    base = cal_loss(
        logits, target, lam=cal_lam, delta=cal_delta, gamma=cal_gamma
    ) + float(dice_w) * dice_loss(logits, target)
    zid = zoom_ids.reshape(-1).long()
    b_tab = torch.tensor([beta_x4, beta_other, beta_other], device=logits.device).to(base.dtype)
    w = b_tab[zid].mean()
    return base + w * hard_negative_loss(logits, m_hn)


# ================================================================ E


def skel_recall_balanced(
    logits: torch.Tensor,
    y_skel: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """骨架召回（只奖励覆盖，不管多画）。"""
    p = torch.sigmoid(logits)
    numer = (p * y_skel).sum()
    denom = y_skel.sum().clamp_min(eps)
    return 1.0 - numer / denom


def precision_guard(
    logits: torch.Tensor,
    target: torch.Tensor,
    m_hn: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """精度护栏：非 GT 区域的高概率惩罚（背景精度）。"""
    p = torch.sigmoid(logits)
    fp_map = (1.0 - target) * p
    numer = (fp_map * m_hn).sum()
    denom = m_hn.sum().clamp_min(eps)
    return numer / denom


def skel_plus_guard_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    y_skel: torch.Tensor,
    m_hn: torch.Tensor,
    zoom_ids: torch.Tensor,
    *,
    alpha: float = 0.20,
    beta: float = 0.20,
    dice_w: float = 0.5,
    cal_delta: float = 0.5,
) -> torch.Tensor:
    """L = L0 + α·L_skel + β·L_guard。

    B 的失败原因：只有召回项，无精度约束 → X4 过检加剧。
    这里配对加上 guard，让"多画"也受罚。
    """
    base = cal_loss(logits, target, lam=0.5, delta=cal_delta) + float(dice_w) * dice_loss(
        logits, target
    )
    sk = skel_recall_balanced(logits, y_skel)
    gd = precision_guard(logits, target, m_hn)
    return base + float(alpha) * sk + float(beta) * gd


# ================================================================ F


def balanced_pseudo_loss(
    logits_u_w: torch.Tensor,
    logits_u_s: torch.Tensor,
    *,
    tau_p: float = 0.70,
    tau_n: float = 0.05,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """前景/背景**分别归一化**的伪标签损失。

    round1 C 失败原因：全图平均 ⇒ 98% 背景吞没前景 ⇒ 退化为背景压制。

    修正：按伪标签类别分别算均值再加权。
      L = 0.5·L_fg + 0.5·L_bg
    其中 L_fg 只在前景伪标签像素上平均，L_bg 只在背景上平均。
    这样即使前景只占 1%，其梯度也不会被稀释。
    """
    with torch.no_grad():
        pw = torch.sigmoid(logits_u_w)
        fg = pw >= float(tau_p)
        bg = pw <= float(tau_n)
        n_fg = fg.sum()
        n_bg = bg.sum()
        # 前景太少时只保留背景项（但降权），避免噪声伪标签
        use_fg = float(n_fg) >= 64.0

    bce = F.binary_cross_entropy_with_logits(logits_u_s, fg.to(logits_u_s.dtype), reduction="none")
    l_bg_map = bce.masked_fill(~bg, 0.0)
    l_bg = l_bg_map.sum() / n_bg.clamp_min(1.0)

    if use_fg:
        l_fg_map = bce.masked_fill(~fg, 0.0)
        l_fg = l_fg_map.sum() / n_fg.clamp_min(1.0)
        loss = 0.5 * l_fg + 0.5 * l_bg
    else:
        loss = l_bg

    return loss, {
        "fg_ratio": float(n_fg) / float(fg.numel()),
        "bg_ratio": float(n_bg) / float(bg.numel()),
        "used_fg": bool(use_fg),
    }


def semi_balanced_loss(
    logits_l,
    target_l,
    y_skel_l,
    logits_u_w,
    logits_u_s,
    *,
    lambda_u: float = 0.4,
    tau_p: float = 0.70,
    tau_n: float = 0.05,
    alpha_skel: float = 0.15,
    dice_w: float = 0.5,
    cal_delta: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """L = L0 + α·L_skel(labeled) + λ_u·L_pseudo_balanced。"""
    sup = cal_loss(logits_l, target_l, lam=0.5, delta=cal_delta) + float(dice_w) * dice_loss(
        logits_l, target_l
    )
    sk = skel_recall_balanced(logits_l, y_skel_l)
    unsup, info = balanced_pseudo_loss(
        logits_u_w, logits_u_s, tau_p=tau_p, tau_n=tau_n
    )
    total = sup + float(alpha_skel) * sk + float(lambda_u) * unsup
    info.update({"sup": float(sup.detach()), "unsup": float(unsup.detach())})
    return total, info
