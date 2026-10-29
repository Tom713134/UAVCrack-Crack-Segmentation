"""定稿 L0 = CAL(δ=0.5)+Dice，以及失败分析思路二的 L^(2)。

注意：不要用旧 total_loss 的 clDice 分支叠 α；连通项一律用 skel_recall_loss。
"""


from __future__ import annotations
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + float(dice_weight) * dice_loss(logits, target)


def cal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    lam: float = 0.5,
    delta: float = 0.5,
    gamma: float = 0.2,
    eps: float = 1e-7,
    bg_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CAL；可选难背景权 W_hn 改写 L_-（§7.1）。"""
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    dims = (1, 2, 3)
    tp = (p * target).sum(dim=dims)
    fn = ((1.0 - p) * target).sum(dim=dims)
    fp = (p * (1.0 - target)).sum(dim=dims)
    tversky = (tp + eps) / (tp + float(delta) * fn + (1.0 - float(delta)) * fp + eps)
    loss_pos = torch.pow(1.0 - tversky, 1.0 - float(gamma)).mean()

    p_bg = (1.0 - p).clamp(eps, 1.0 - eps)
    neg_map = (1.0 - target) * torch.pow(p, float(gamma)) * torch.log(p_bg)
    if bg_weight is None:
        loss_neg = -neg_map.mean()
    else:
        w = bg_weight
        denom = ((1.0 - target) * w).sum().clamp_min(eps)
        loss_neg = -(w * neg_map).sum() / denom
    return float(lam) * loss_pos + (1.0 - float(lam)) * loss_neg


def hard_neg_weight(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_hn: float = 1.0,
    tau_hn: float = 0.3,
) -> torch.Tensor:
    """W_hn = 1 + β_hn · (1-Y) · 1_{P>τ_hn}。"""
    p = torch.sigmoid(logits)
    return 1.0 + float(beta_hn) * (1.0 - target) * (p > float(tau_hn)).to(p.dtype)


def skel_recall_loss(logits: torch.Tensor, y_skel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    numer = (p * y_skel).sum()
    denom = y_skel.sum().clamp_min(eps)
    return 1.0 - numer / denom


def line_fp_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    m_line: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L_line：只罚标线域上的真背景高概率（护 A′）。"""
    p = torch.sigmoid(logits)
    mask = m_line * (1.0 - target)
    numer = (mask * p).sum()
    denom = mask.sum().clamp_min(eps)
    return numer / denom


def shadow_fn_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    m_dark: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L_shFN：暗区真缝上的漏检率。"""
    p = torch.sigmoid(logits)
    mask = m_dark * target
    numer = (mask * (1.0 - p)).sum()
    denom = mask.sum().clamp_min(eps)
    return numer / denom


def l0_cal_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float = 0.5,
    cal_lam: float = 0.5,
    cal_delta: float = 0.5,
    cal_gamma: float = 0.2,
) -> torch.Tensor:
    """定稿公共起点 L0。"""
    return cal_loss(
        logits, target, lam=cal_lam, delta=cal_delta, gamma=cal_gamma
    ) + float(dice_weight) * dice_loss(logits, target)


def l_cond_shadow(
    logits: torch.Tensor,
    target: torch.Tensor,
    m_dark: torch.Tensor,
    i_go: torch.Tensor,
    *,
    zoom_is_x4: torch.Tensor | None = None,
    mu: float = 0.2,
    dice_weight: float = 0.5,
    cal_lam: float = 0.5,
    cal_delta: float = 0.5,
    cal_gamma: float = 0.2,
) -> torch.Tensor:
    """收窄条件损失：L_cond = L0 + μ · 1_go · L_shFN（§5.5）。

    i_go: (B,) 或 (B,1,1,1)；训练可用规则 1_sh。
    X4 时 μ ← 1.25μ。
    """
    loss0 = l0_cal_dice(
        logits,
        target,
        dice_weight=dice_weight,
        cal_lam=cal_lam,
        cal_delta=cal_delta,
        cal_gamma=cal_gamma,
    )
    if float(mu) <= 0:
        return loss0
    gate = i_go.reshape(-1).float()
    if float(gate.sum().detach()) <= 0:
        return loss0
    # 逐样本 L_shFN，再按 1_go 加权平均
    p = torch.sigmoid(logits)
    mask = m_dark * target
    dims = (1, 2, 3)
    numer = (mask * (1.0 - p)).sum(dim=dims)
    denom = mask.sum(dim=dims).clamp_min(1e-6)
    per = numer / denom
    gated = (gate * per).sum() / gate.sum().clamp_min(1e-6)
    mu_eff = float(mu)
    if zoom_is_x4 is not None:
        # 仅对命中且为 X4 的样本抬 μ
        x4 = zoom_is_x4.reshape(-1).float()
        frac_x4 = ((gate * x4).sum() / gate.sum().clamp_min(1e-6)).detach()
        mu_eff = mu_eff * (1.0 + 0.25 * float(frac_x4.cpu()))
    return loss0 + mu_eff * gated


def failfix_total_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    m_line: torch.Tensor | None = None,
    m_dark: torch.Tensor | None = None,
    y_skel: torch.Tensor | None = None,
    zoom_is_x4: torch.Tensor | None = None,
    dice_weight: float = 0.5,
    cal_lam: float = 0.5,
    cal_delta: float = 0.5,
    cal_gamma: float = 0.2,
    beta_fp: float = 0.2,
    mu_sh: float = 0.2,
    alpha_sr: float = 0.05,
    use_hard_neg: bool = True,
    beta_hn: float = 1.0,
    tau_hn: float = 0.3,
) -> torch.Tensor:
    """思路二总损失 L^(2)。

    L = CAL*(W_hn) + λ Dice + β L_line + μ L_shFN + α L_sr
    X4 时 μ ← 1.25 μ（按 batch 平均缩放）。
    """
    bg_w = None
    if use_hard_neg:
        bg_w = hard_neg_weight(logits, target, beta_hn=beta_hn, tau_hn=tau_hn)
    loss = cal_loss(
        logits,
        target,
        lam=cal_lam,
        delta=cal_delta,
        gamma=cal_gamma,
        bg_weight=bg_w,
    )
    loss = loss + float(dice_weight) * dice_loss(logits, target)

    if m_line is not None and float(beta_fp) > 0:
        loss = loss + float(beta_fp) * line_fp_loss(logits, target, m_line)
    if m_dark is not None and float(mu_sh) > 0:
        mu = float(mu_sh)
        if zoom_is_x4 is not None:
            # batch 内 X4 比例抬高 μ
            frac = zoom_is_x4.float().mean().clamp(0.0, 1.0)
            mu = mu * (1.0 + 0.25 * float(frac.detach().cpu()))
        loss = loss + mu * shadow_fn_loss(logits, target, m_dark)
    if y_skel is not None and float(alpha_sr) > 0:
        loss = loss + float(alpha_sr) * skel_recall_loss(logits, y_skel)
    return loss
