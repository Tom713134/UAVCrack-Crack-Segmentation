"""UAV-CrackX 读取：几何复用 ×5 + 可选失败分析课程增广 / 硬先验。"""


from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from hard_priors import curriculum_indicators, gray_u8, t_b_shadow, t_c_weak_bridge

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _build_geo_catalog() -> list[tuple[float, float, bool]]:
    return [
        (1.00, 0.0, False),
        (0.85, -10.0, False),
        (1.15, 10.0, False),
        (1.00, 0.0, True),
        (0.90, 5.0, True),
    ]


GEO_CATALOG = _build_geo_catalog()
TRAIN_EXPAND_FACTOR = len(GEO_CATALOG)  # 5
assert TRAIN_EXPAND_FACTOR == 5

# 变焦整数 id（供分变焦损失索引）
ZOOM_ID = {"UAV-CrackX4": 0, "UAV-CrackX8": 1, "UAV-CrackX16": 2}


def imread(path: Path, color: bool) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _norm_chw(rgb_u8: np.ndarray) -> np.ndarray:
    image = rgb_u8.astype(np.float32) / 255.0
    c = image.shape[2]
    mean = IMAGENET_MEAN if c == 3 else np.tile(IMAGENET_MEAN, c // 3)
    std = IMAGENET_STD if c == 3 else np.tile(IMAGENET_STD, c // 3)
    image = (image - mean) / std
    return np.transpose(image, (2, 0, 1))


class CrackSet(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: dict,
        part: str,
        *,
        train: bool,
        need_skel: bool = False,
        need_hard_masks: bool = False,
        curriculum: bool = False,
        curriculum_mode: str = "full",
        r_tube: int = 2,
        expand_factor: int | None = None,
        p_b: float = 0.7,
        p_c: float = 0.3,
        need_hn: bool = False,
        hn_dark_q: float = 0.20,
        hn_lowc_q: float = 0.30,
        need_conf_region: bool = False,
        gt_dilate: int = 2,
        ambig_band: int = 4,
        need_dual: bool = False,
        need_hr_crop: bool = False,
        hr_zoom: float = 2.0,
        hr_out_size: int = 384,
    ) -> None:
        self.data_root = data_root
        self.train = train
        self.need_skel = bool(need_skel)
        self.need_hard_masks = bool(need_hard_masks)
        self.need_hn = bool(need_hn)
        self.hn_dark_q = float(hn_dark_q)
        self.hn_lowc_q = float(hn_lowc_q)
        self.need_conf_region = bool(need_conf_region)
        self.gt_dilate = int(gt_dilate)
        self.ambig_band = int(ambig_band)
        self.need_dual = bool(need_dual)
        self.need_hr_crop = bool(need_hr_crop)
        self.hr_zoom = float(hr_zoom)
        self.hr_out_size = int(hr_out_size)
        self.curriculum = bool(curriculum) and bool(train)
        # full=旧三范式；shadow=只做阴影 T_B + 阴影过采样（收窄方案）
        self.curriculum_mode = str(curriculum_mode)
        self.r_tube = int(r_tube)
        self.p_b = float(p_b)
        self.p_c = float(p_c)
        if expand_factor is None:
            expand_factor = TRAIN_EXPAND_FACTOR if train else 1
        self.expand_factor = max(1, int(expand_factor))
        self.samples: list[tuple[str, str]] = []
        for zoom, payload in split["zooms"].items():
            for name in payload["files"][part]:
                self.samples.append((zoom, name))
        self._base_weights: list[float] | None = None

    def __len__(self) -> int:
        return len(self.samples) * self.expand_factor

    def base_sample_weights(self) -> list[float]:
        """每张原图的课程权；几何展开后复制。"""
        if self._base_weights is not None:
            return self._base_weights
        weights = []
        for zoom, name in self.samples:
            img_path = self.data_root / "leftImg8bit" / "train" / zoom / name
            gt_path = self.data_root / "gtFine" / "train" / zoom / (Path(name).stem + ".png")
            bgr = imread(img_path, True)
            gt = imread(gt_path, False)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            y01 = (gt > 0).astype(np.float32)
            info = curriculum_indicators(zoom, y01, gray_u8(rgb))
            if self.curriculum and self.curriculum_mode == "shadow":
                # 收窄：w=1+w_B·1_sh+w_E·φ7（§5.5）
                w = 1.0 + 2.0 * float(info["i_sh"]) + 1.0 * float(info["i_e"])
            else:
                w = float(info["sample_weight"])
            weights.append(float(w))
        self._base_weights = weights
        return weights

    def expanded_sample_weights(self) -> list[float]:
        base = self.base_sample_weights()
        return base * self.expand_factor

    def __getitem__(self, index: int) -> dict:
        n = len(self.samples)
        base_idx = int(index) % n
        view_id = int(index) // n
        zoom, name = self.samples[base_idx]
        img_path = self.data_root / "leftImg8bit" / "train" / zoom / name
        gt_path = self.data_root / "gtFine" / "train" / zoom / (Path(name).stem + ".png")
        bgr = imread(img_path, True)
        gt = imread(gt_path, False)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.train:
            rgb, gt = _augment(rgb, gt, view_id=view_id)
            # K：X4 专用高分辨率裁剪（缝像素变粗，更易与纹理区分）
            if self.need_hr_crop and ZOOM_ID.get(zoom, 1) == 0:
                from round4_x4 import hr_crop_augment

                hh, ww = rgb.shape[:2]
                rgb, gt = hr_crop_augment(
                    rgb, gt, zoom_high=self.hr_zoom, out_h=hh, out_w=ww
                )
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                gt = np.ascontiguousarray(gt)
                if rgb.shape[:2] != (hh, ww):
                    raise RuntimeError(f"hr_crop 尺寸异常: {rgb.shape} vs {(hh, ww)}")

        mask_np = (gt > 0).astype(np.float32)
        gray = gray_u8(rgb)
        info = curriculum_indicators(zoom, mask_np, gray)
        m_dark = info["m_dark"]
        m_line = info["m_line"]

        if self.curriculum:
            mode = self.curriculum_mode
            if mode == "shadow":
                # 只做阴影 T_B；不做标线 Y=0、不做 T_C
                if int(info["i_sh"]) == 1 or np.random.rand() < self.p_b:
                    if int(info["i_sh"]) == 1 or np.random.rand() < 0.5:
                        rgb = t_b_shadow(rgb, m_dark)
            else:
                # 标线本体强制为负（护 A′）：合成/难负时线体 Y=0
                if int(info["i_a"]) == 1 and np.random.rand() < 0.5:
                    mask_np = mask_np * (1.0 - m_line)
                if int(info["i_sh"]) == 1 or np.random.rand() < self.p_b:
                    if int(info["i_sh"]) == 1 or np.random.rand() < 0.5:
                        rgb = t_b_shadow(rgb, m_dark)
                rgb = t_c_weak_bridge(rgb, mask_np, p_c=self.p_c)
            # 增广后再算一次先验，保证与最终 RGB 对齐
            gray = gray_u8(rgb)
            info = curriculum_indicators(zoom, mask_np, gray)
            m_dark = info["m_dark"]
            m_line = info["m_line"]

        if self.curriculum and self.curriculum_mode == "shadow":
            sample_w = 1.0 + 2.0 * float(info["i_sh"]) + 1.0 * float(info["i_e"])
        else:
            sample_w = float(info["sample_weight"])

        out = {
            "image": torch.from_numpy(np.ascontiguousarray(_norm_chw(rgb))),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_np[None])),
            "zoom": zoom,
            "name": name,
            "sample_weight": float(sample_w),
            "is_x4": 1.0 if ("X4" in str(zoom)) else 0.0,
            "i_sh": float(info["i_sh"]),
            "zoom_id": ZOOM_ID.get(zoom, 1),
        }
        if self.need_hard_masks or self.curriculum:
            out["m_dark"] = torch.from_numpy(np.ascontiguousarray(m_dark[None].astype(np.float32)))
            out["m_line"] = torch.from_numpy(np.ascontiguousarray(m_line[None].astype(np.float32)))
        if self.need_skel:
            from boundary_utils import skel_tube

            tube = skel_tube(mask_np, r_tube=self.r_tube)
            out["skel"] = torch.from_numpy(np.ascontiguousarray(tube[None]))
        if self.need_hn:
            from round2_losses import dark_texture_negative

            m_hn = dark_texture_negative(
                rgb,
                mask_np,
                dark_q=self.hn_dark_q,
                lowc_q=self.hn_lowc_q,
            )
            out["m_hn"] = torch.from_numpy(np.ascontiguousarray(m_hn[None].astype(np.float32)))
        if self.need_conf_region:
            from round3_methods import confident_region

            w_map, amb = confident_region(
                mask_np, gt_dilate=self.gt_dilate, ambig_band=self.ambig_band
            )
            out["w_map"] = torch.from_numpy(np.ascontiguousarray(w_map[None].astype(np.float32)))
            out["amb"] = torch.from_numpy(np.ascontiguousarray(amb[None].astype(np.float32)))
        if self.need_dual:
            from round4_x4 import make_dual_res

            dual = make_dual_res(rgb)
            out["image_dual"] = torch.from_numpy(np.ascontiguousarray(_norm_chw(dual)))
        return out


def make_weighted_sampler(dataset: CrackSet) -> WeightedRandomSampler:
    w = dataset.expanded_sample_weights()
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


def _strong_photometric(rgb: np.ndarray) -> np.ndarray:
    """强光度扰动（无标注一致性用）：亮度/对比度/伽马/噪声。"""
    out = rgb.astype(np.float32)
    mean = float(out.mean())
    out = (out - mean) * float(np.random.uniform(0.7, 1.3)) + mean
    out = np.clip((np.clip(out, 0, 255) / 255.0) ** float(np.random.uniform(0.7, 1.4)) * 255.0, 0, 255)
    noise = np.random.normal(0.0, float(np.random.uniform(2.0, 8.0)), out.shape)
    out = np.clip(out + noise, 0, 255)
    return out.astype(np.uint8)


class UnlabeledSet(Dataset):
    """官方 val 的 300 张无标注图（思路 C 半监督）。

    只做几何/光度增广，不读 GT。weak=True 仅轻微扰动（出伪标签），
    weak=False 走强增广（算一致性损失）。
    注意：val 无标注，直接用原始数据根即可（clean 副本只镜像了 train）。
    """

    def __init__(self, data_root: Path, *, weak: bool = False) -> None:
        self.data_root = Path(data_root)
        self.weak = bool(weak)
        self.samples: list[tuple[str, str]] = []
        root = self.data_root / "leftImg8bit" / "val"
        for zdir in sorted(root.glob("*")):
            if not zdir.is_dir():
                continue
            for p in sorted(zdir.glob("*.jpg")):
                self.samples.append((zdir.name, p.name))
        if not self.samples:
            raise FileNotFoundError(f"未找到无标注图: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        zoom, name = self.samples[int(index)]
        rgb = cv2.cvtColor(
            imread(self.data_root / "leftImg8bit" / "val" / zoom / name, True), cv2.COLOR_BGR2RGB
        )
        if self.weak:
            # 弱：只翻转 + 轻微亮度
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
            rgb = np.clip(rgb.astype(np.float32) * float(np.random.uniform(0.95, 1.05)), 0, 255).astype(
                np.uint8
            )
        else:
            # 强：几何 + 强光度
            h, w = rgb.shape[:2]
            ang = float(np.random.uniform(-15.0, 15.0))
            sc = float(np.random.uniform(0.85, 1.15))
            m = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), ang, sc)
            rgb = cv2.warpAffine(
                rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[::-1])
            rgb = _strong_photometric(rgb)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(_norm_chw(rgb))),
            "zoom": zoom,
            "name": name,
            "zoom_id": ZOOM_ID.get(zoom, 1),
        }


def _apply_geo(
    rgb: np.ndarray,
    gt: np.ndarray,
    *,
    scale: float,
    angle: float,
    flip_h: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if flip_h:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        gt = np.ascontiguousarray(gt[:, ::-1])
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), float(angle), float(scale))
    rgb = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    gt = cv2.warpAffine(
        gt, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return rgb, gt


def _augment(rgb: np.ndarray, gt: np.ndarray, view_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    scale, angle, flip_h = GEO_CATALOG[int(view_id) % len(GEO_CATALOG)]
    scale = float(scale) * float(np.random.uniform(0.97, 1.03))
    angle = float(angle) + float(np.random.uniform(-2.0, 2.0))
    rgb, gt = _apply_geo(rgb, gt, scale=scale, angle=angle, flip_h=flip_h)
    if np.random.rand() < 0.5:
        rgb = np.ascontiguousarray(rgb[::-1])
        gt = np.ascontiguousarray(gt[::-1])
    brightness = float(np.random.uniform(0.85, 1.15))
    contrast = float(np.random.uniform(0.85, 1.15))
    mean = float(rgb.mean())
    out = (rgb.astype(np.float32) - mean) * contrast + mean
    out = out * brightness
    return np.clip(out, 0, 255).astype(np.uint8), gt
