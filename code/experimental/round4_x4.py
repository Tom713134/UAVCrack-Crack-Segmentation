"""Round 4：X4 专项方法。

核心诊断（三轮独立验证）：
  X4 的 P=68~72 / R=84~87，过度检出；X8/X16 的 P/R 接近平衡且已达上限。
  所有全局损失改动（骨架、难负、歧义排除）都在提升 X16（65.3→65.9）
  的同时损伤 X4（62.7→61.5）。方向相反，说明二者需要的不是同一种归纳偏置。

X4 过检的物理成因假设：
  384×384 的 X4 块里，一条 2–4 px 的缝与一片 20–50 px 的暗色车辙/修补带
  在**局部**观感接近。区分它们需要**更大范围**的证据（纹理是否沿一个方向
  延伸数百像素、边界是否锐利、是否有反复的明暗条纹）。

三种给出"更多上下文"的方式：
  J: 双分辨率输入（原图 + 0.5× 缩略图拼接为 6 通道）→ 网络直接看到全局
  K: 输入 2× 放大后随机裁剪回 384（等效感受野不变但像素更细）+ 只在 X4 用
  L: 分变焦独立的 BatchNorm 统计量（X4 的纹理分布与 X16 不同）
"""


from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================ J
# 双分辨率输入


def make_dual_res(rgb_u8: np.ndarray, size: int = 384) -> np.ndarray:
    """原图 + 0.5× 缩略图（上采样回原尺寸），拼成 6 通道。

    缩略图提供"大范围布局"信息：车辙带是长条带状，缝是细线。
    在 0.5× 下缝几乎消失，但车辙带依然明显 ⇒ 网络可据此区分。
    """
    h, w = rgb_u8.shape[:2]
    small = cv2.resize(rgb_u8, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    small = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([rgb_u8, small], axis=2)


class DualResStem(nn.Module):
    """6 通道 → 3 通道的轻量融合，接在原有 3 通道主干之前。

    用 1×1 卷积把 6 通道压到 3 通道，保留原主干的全部权重可用。
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(6, 3, kernel_size=1, bias=True)
        self.base = base
        # 初始化为「原图通道恒等 + 缩略图通道零贡献」，等价于从原模型起步
        with torch.no_grad():
            self.fuse.weight.zero_()
            for c in range(3):
                self.fuse.weight[c, c, 0, 0] = 1.0
            self.fuse.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(self.fuse(x))


# ================================================================ K
# X4 专用高分辨率随机裁剪


def hr_crop_augment(
    rgb_u8: np.ndarray,
    gt: np.ndarray,
    *,
    zoom_high: float = 2.0,
    out_h: int = 378,
    out_w: int = 672,
) -> tuple[np.ndarray, np.ndarray]:
    """放大 zoom_high 倍后随机裁到 out_h × out_w（与原图同尺寸）。

    效果：同样视野内裂缝**像素宽度翻倍**，网络看到更粗更清晰的缝。
    依据 Pushing the Envelope of Thin Crack Detection (arXiv:2101.03326)。
    输出尺寸与原图一致，保证 collate 与后续 pad_bottom 逻辑不变。
    """
    h, w = rgb_u8.shape[:2]
    nh, nw = int(round(h * zoom_high)), int(round(w * zoom_high))
    big = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    big_gt = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # 若放大后仍不足，先 padding
    if nh < out_h or nw < out_w:
        pad_h = max(0, out_h - nh)
        pad_w = max(0, out_w - nw)
        big = cv2.copyMakeBorder(big, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        big_gt = cv2.copyMakeBorder(big_gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        nh, nw = big.shape[:2]

    y0 = np.random.randint(0, nh - out_h + 1)
    x0 = np.random.randint(0, nw - out_w + 1)
    out_rgb = np.ascontiguousarray(big[y0 : y0 + out_h, x0 : x0 + out_w])
    out_gt = np.ascontiguousarray(big_gt[y0 : y0 + out_h, x0 : x0 + out_w])

    # 兜底：强制尺寸一致
    if out_rgb.shape[0] != out_h or out_rgb.shape[1] != out_w:
        out_rgb = cv2.resize(out_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        out_gt = cv2.resize(out_gt, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return out_rgb, out_gt


# ================================================================ L
# 分变焦独立 BN 统计量


class ZoomBN(nn.Module):
    """为每个变焦维护独立的 BatchNorm 统计量（共享仿射参数）。

    X4 的图像统计（纹理尺度、对比度分布）与 X16 差异显著。
    共享 BN 会让二者的归一化统计相互污染。
    这里在 forward 时按 zoom_id 切换到对应变焦的 running stats。
    """

    def __init__(self, num_features: int, n_zooms: int = 3, **bn_kw) -> None:
        super().__init__()
        self.n_zooms = int(n_zooms)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features, **bn_kw) for _ in range(self.n_zooms)])

    def forward(self, x: torch.Tensor, zoom_id: int = 1) -> torch.Tensor:
        return self.bns[int(zoom_id) % self.n_zooms](x)


def convert_bn_to_zoom_bn(module: nn.Module, n_zooms: int = 3) -> nn.Module:
    """把所有 BatchNorm2d 替换为 ZoomBN（复制原统计量到每一份）。"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            z = ZoomBN(child.num_features, n_zooms=n_zooms, eps=child.eps, momentum=child.momentum)
            for i in range(n_zooms):
                z.bns[i].load_state_dict(child.state_dict())
            setattr(module, name, z)
        else:
            convert_bn_to_zoom_bn(child, n_zooms=n_zooms)
    return module


class ZoomConditionedUNet(nn.Module):
    """包装：forward 时按 zoom_id 路由到对应 BN 统计量。

    为了兼容「一个 batch 混多个变焦」，这里按变焦**分组前向**再拼回，
    保证每组用的是自己的 BN。
    """

    def __init__(self, base: nn.Module, n_zooms: int = 3) -> None:
        super().__init__()
        self.base = convert_bn_to_zoom_bn(base, n_zooms=n_zooms)
        self.n_zooms = int(n_zooms)
        self._cur_zoom = 1

    def set_zoom(self, z: int) -> None:
        self._cur_zoom = int(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ZoomBN 通过模块级全局变量读取当前 zoom
        global _CURRENT_ZOOM
        _CURRENT_ZOOM = self._cur_zoom
        return self.base(x)


_CURRENT_ZOOM = 1


def _zoom_bn_forward_patch(self, x):
    return self.bns[min(int(_CURRENT_ZOOM), self.n_zooms - 1)](x)


ZoomBN.forward = _zoom_bn_forward_patch
