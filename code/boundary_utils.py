"""边界带、软标签、局部对比（供三种辅助思路共用）。"""


from __future__ import annotations
import cv2
import numpy as np


def boundary_band(mask01: np.ndarray, r_b: int = 2) -> np.ndarray:
    """硬标注 → 骨架邻域带 B（float32 0/1）。"""
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.float32)
    # 距离到背景：裂缝内部；再对骨架邻域用膨胀近似带宽
    skel = _skeleton(m)
    if int(skel.sum()) == 0:
        skel = m
    dist = cv2.distanceTransform((1 - skel).astype(np.uint8), cv2.DIST_L2, 5)
    band = (dist <= float(r_b)).astype(np.float32)
    # 只在裂缝附近有意义：与膨胀后的裂缝相交，避免整图远域
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_b + 1, 2 * r_b + 1)))
    return (band * dil.astype(np.float32)).astype(np.float32)


def soft_label(mask01: np.ndarray, r: float = 2.0, sigma: float | None = None) -> np.ndarray:
    """硬 Y → 带宽 r 的软标签 Ỹ。"""
    m = (mask01 > 0).astype(np.uint8)
    out = m.astype(np.float32)
    if int(m.sum()) == 0:
        return out
    if sigma is None:
        sigma = max(float(r) / 2.0, 0.5)
    dist = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5)
    ring = (dist > 0) & (dist <= float(r))
    out[ring] = np.exp(-(dist[ring] ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    return out


def local_contrast(rgb_u8: np.ndarray, band: np.ndarray, ring: int = 3) -> np.ndarray:
    """边界带内局部对比 C∈[0,1]；带外为 0。"""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = int(ring) * 2 + 1
    blur = cv2.blur(gray, (k, k))
    c = np.abs(gray - blur)
    b = band > 0.5
    if not np.any(b):
        return np.zeros_like(gray, dtype=np.float32)
    vals = c[b]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-6:
        norm = np.zeros_like(c)
    else:
        norm = (c - lo) / (hi - lo)
    out = np.zeros_like(c, dtype=np.float32)
    out[b] = norm[b].astype(np.float32)
    return out


def _skeleton(mask_u8: np.ndarray) -> np.ndarray:
    """Zhang-Suen 风格细化的 OpenCV 近似：迭代腐蚀保留。"""
    skel = np.zeros_like(mask_u8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask_u8.copy()
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def skel_tube(mask01: np.ndarray, r_tube: int = 2) -> np.ndarray:
    """硬标注 → 管状骨架 Y_skel（Skeleton Recall 用）。"""
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros_like(m, dtype=np.float32)
    skel = _skeleton(m)
    if int(skel.sum()) == 0:
        skel = m
    if int(r_tube) <= 0:
        return skel.astype(np.float32)
    k = 2 * int(r_tube) + 1
    dil = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return dil.astype(np.float32)


def local_contrast_full(rgb_u8: np.ndarray, ring: int = 3) -> np.ndarray:
    """整图局部对比 C∈[0,1]，供多证据打分头作特征（非开环损失权重）。"""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = int(ring) * 2 + 1
    blur = cv2.blur(gray, (k, k))
    c = np.abs(gray - blur)
    lo, hi = float(c.min()), float(c.max())
    if hi - lo < 1e-6:
        return np.zeros_like(c, dtype=np.float32)
    return ((c - lo) / (hi - lo)).astype(np.float32)
