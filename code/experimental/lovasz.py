"""方法③：Lovasz-Softmax 损失（直接优化 IoU）。

Berman et al., CVPR 2018: "The Lovasz-Softmax loss: A tractable surrogate
for the optimization of the intersection-over-union measure in neural networks"

核心思想
--------
IoU 是集合函数，不可直接梯度优化。但对每个类别，IoU 损失可写成
"排序后的误差累积" 的 Lovász 扩展（凸函数的分段线性扩展），从而可微。

对二分类，设：
    y* ∈ {0,1}  真值
    f ∈ R       模型输出（logit 或概率）
    e_i = 1 - f_i   （若 y_i=1）
    e_i = f_i       （若 y_i=0）
按 e 降序排列得到 π，则

    Δ_ IoU(π) = 1 - IoU(π)

    L = Σ_i Δ_i · |g_i|,   g_i = 误差排序的前缀梯度

这里用 ELU+1 把 logits 变到 [0,1] 区间（比 sigmoid 在极端值处梯度更稳）。
"""


from __future__ import annotations
import torch
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """给定按误差降序排列的真值，返回 Lovász 梯度。"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """二分类 Lovász hinge（逐样本展平）。

    标准实现（Berman et al. 2018, 官方 repo）：
        signs = 2y − 1
        errors = 1 − logits·signs         # 错误程度，越大越错
        按 errors 降序排，与 lovasz_grad 点积
        ELU 用于把 errors 平滑到 (−1, +∞)，保证在 0 附近可微
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # elu(errors) + 1 在 errors=0 处为 1，errors<0 时为 (0,1)，errors>0 时 >1
    # 保证非负且单调，可微
    return torch.dot(F.elu(errors_sorted) + 1.0, grad)


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    """(B,1,H,W) logits 与 labels。"""
    logits = logits.reshape(logits.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    if per_image:
        losses = [lovasz_hinge_flat(lg, lb) for lg, lb in zip(logits, labels)]
        return torch.stack(losses).mean()
    return lovasz_hinge_flat(logits.reshape(-1), labels.reshape(-1))


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """二分类 Lovász-softmax（probs 已在 [0,1]）。"""
    if len(probs) == 0:
        return probs.sum() * 0.0
    losses = []
    for cls in (0, 1):
        fg = (labels == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return sum(losses) / max(len(losses), 1)


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    probs = probs.reshape(probs.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    if per_image:
        losses = [lovasz_softmax_flat(p, l) for p, l in zip(probs, labels)]
        return torch.stack(losses).mean()
    return lovasz_softmax_flat(probs.reshape(-1), labels.reshape(-1))


def lovasz_plus_cal(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    lovasz_weight: float = 0.5,
    cal_delta: float = 0.5,
    cal_gamma: float = 0.2,
    cal_lam: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """L = L0(CAL+Dice) + w · Lovász。

    保留 CAL+Dice 作为主项（已验证稳定），叠加 Lovász 强化 IoU 方向。
    若纯 Lovász 单独用，早期训练极不稳定（细缝样本少时梯度稀疏）。
    """
    from losses import l0_cal_dice

    base = l0_cal_dice(
        logits,
        target,
        dice_weight=dice_weight,
        cal_lam=cal_lam,
        cal_delta=cal_delta,
        cal_gamma=cal_gamma,
    )
    lov = lovasz_hinge(logits, target, per_image=True)
    return base + float(lovasz_weight) * lov


def lovasz_only(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """纯 Lovász（对照组）。"""
    return lovasz_hinge(logits, target, per_image=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    # 自检：完美预测应接近 0，全错应较大
    logits = torch.randn(2, 1, 32, 32) * 2
    target = (torch.rand(2, 1, 32, 32) > 0.9).float()
    print("random        :", float(lovasz_hinge(logits, target)))
    print("perfect (×10) :", float(lovasz_hinge(logits * 10 * (2 * target - 1), target)))
    print("inverted      :", float(lovasz_hinge(-logits * 10 * (2 * target - 1), target)))
    print("lovasz_grad   :", lovasz_grad(torch.tensor([1.0, 1, 0, 0])).tolist())
