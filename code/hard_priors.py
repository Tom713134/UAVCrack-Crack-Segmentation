"""阴影 / 标线先验与条件增广（对齐失败分析文档 §5–§6）。

只做「可条件触发」的局部操作；禁止全局强 CLAHE / 信任第四通道。
"""


from __future__ import annotations
import cv2
import numpy as np

# §5.1
NU_DARK = 0.25
R_SHADOW = 15
T_DARK = 0.25
T_LOWC = 0.30

# §5.2
Q_BRIGHT = 0.95
LINE_MIN_AREA = 80
LINE_PCA_MIN = 0.85
LINE_ASP_MAX = 0.25

# §6.1
RHO_A = 0.01
RHO_Y = 0.005
RHO_F = 0.04
W_B = W_A = W_F = W_E = 2.0


def gray_u8(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)


def m_dark(gray: np.ndarray, nu: float = NU_DARK) -> np.ndarray:
    thr = float(np.quantile(gray.astype(np.float32), float(nu)))
    return (gray.astype(np.float32) <= thr).astype(np.float32)


def low_contrast_score(gray: np.ndarray, r_s: int = R_SHADOW) -> tuple[float, np.ndarray]:
    g = gray.astype(np.float32)
    k = int(r_s) * 2 + 1
    mean = cv2.blur(g, (k, k))
    c = np.abs(g - mean)
    thr = float(np.quantile(c, 0.25))
    mask = (c <= thr).astype(np.float32)
    return float(mask.mean()), mask


def shadow_flag(gray: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """返回 (M_dark, s_dark, s_lowC, 1_sh)。"""
    md = m_dark(gray)
    s_dark = float(md.mean())
    s_lowc, _ = low_contrast_score(gray)
    flag = int(s_dark >= T_DARK or s_lowc >= T_LOWC)
    return md, s_dark, s_lowc, flag


def _cc_pca_asp(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    if ys.size < 2:
        return 0.0, 1.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    lam2, lam1 = float(eig[0]), float(eig[1])
    pca = lam1 / (lam1 + lam2 + 1e-12)
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    asp = min(h, w) / max(h, w)
    return pca, asp


def m_line(gray: np.ndarray) -> np.ndarray:
    """亮细长连通域并集作为标线候选。"""
    thr = float(np.quantile(gray.astype(np.float32), Q_BRIGHT))
    bright = (gray.astype(np.float32) >= thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    out = np.zeros_like(gray, dtype=np.float32)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < LINE_MIN_AREA:
            continue
        ys, xs = np.where(labels == i)
        pca, asp = _cc_pca_asp(ys, xs)
        if pca >= LINE_PCA_MIN and asp <= LINE_ASP_MAX:
            out[labels == i] = 1.0
    return out


def curriculum_indicators(
    zoom: str,
    y01: np.ndarray,
    gray: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    md, s_dark, s_lowc, i_sh = shadow_flag(gray)
    ml = m_line(gray)
    y_rate = float((y01 > 0.5).mean())
    line_rate = float(ml.mean())
    i_b = int(i_sh)
    i_a = int(line_rate >= RHO_A and y_rate <= RHO_Y)
    i_f = int(y_rate >= RHO_F)
    i_e = int(str(zoom).endswith("X4") or "X4" in str(zoom))
    w = 1.0 + W_B * i_b + W_A * i_a + W_F * i_f + W_E * i_e
    return {
        "m_dark": md,
        "m_line": ml,
        "s_dark": s_dark,
        "s_lowc": s_lowc,
        "i_sh": i_sh,
        "i_b": i_b,
        "i_a": i_a,
        "i_f": i_f,
        "i_e": i_e,
        "sample_weight": float(w),
    }


def t_b_shadow(rgb: np.ndarray, m_dark_map: np.ndarray, mode: str | None = None) -> np.ndarray:
    """条件阴影光度变换 §6.2；mode=None 时均匀任选。"""
    if mode is None:
        mode = str(np.random.choice(["gamma", "v_stretch", "clahe_weak"]))
    out = rgb.copy()
    if mode == "gamma":
        g = float(np.random.uniform(0.70, 0.95))
        x = out.astype(np.float32) / 255.0
        out = np.clip((x**g) * 255.0, 0, 255).astype(np.uint8)
    elif mode == "v_stretch":
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        a = float(np.random.uniform(1.05, 1.25))
        b = float(np.random.uniform(5.0, 20.0))
        mask = m_dark_map > 0.5
        v = hsv[:, :, 2]
        v[mask] = np.clip(a * v[mask] + b, 0, 255)
        hsv[:, :, 2] = v
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    else:  # clahe_weak
        clip = float(np.random.uniform(1.0, 1.5))
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out


def t_b_fixed(rgb: np.ndarray) -> np.ndarray:
    """推理用固定弱提亮（模块 S）。"""
    x = rgb.astype(np.float32) / 255.0
    return np.clip((x**0.85) * 255.0, 0, 255).astype(np.uint8)


def t_c_weak_bridge(rgb: np.ndarray, y01: np.ndarray, p_c: float = 0.3) -> np.ndarray:
    """弱桥扰动：输入局部抹平，Y 保持完整 §6.4。"""
    if np.random.rand() >= float(p_c):
        return rgb
    m = (y01 > 0.5).astype(np.uint8)
    if int(m.sum()) == 0:
        return rgb
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ys, xs = np.where(dil > 0)
    if ys.size == 0:
        return rgb
    n_pick = max(1, int(0.15 * ys.size))
    idx = np.random.choice(ys.size, size=n_pick, replace=False)
    gray = gray_u8(rgb).astype(np.float32)
    blur = cv2.blur(gray, (15, 15))
    eta = float(np.random.uniform(0.3, 0.6))
    out = rgb.astype(np.float32)
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        g_new = (1.0 - eta) * gray[y, x] + eta * blur[y, x]
        scale = g_new / max(gray[y, x], 1.0)
        out[y, x] = np.clip(out[y, x] * scale, 0, 255)
    return out.astype(np.uint8)


def gate_prob(
    prob: np.ndarray,
    m_line_map: np.ndarray,
    y01: np.ndarray | None = None,
    *,
    kappa: float = 0.5,
    tau_keep: float = 0.6,
) -> np.ndarray:
    """模块 Gate：仅压「标线 ∧ 低置信」；Y=1 处不压（护 A′）。"""
    p = prob.astype(np.float32).copy()
    hit = (m_line_map > 0.5) & (p < float(tau_keep))
    if y01 is not None:
        hit = hit & (y01 <= 0.5)
    p[hit] = float(kappa) * p[hit]
    return p
