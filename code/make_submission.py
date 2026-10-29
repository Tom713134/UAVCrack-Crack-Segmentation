"""VLP 提交包生成（终版：5 模型集成 + 两级后处理）。

后处理（内部 confirm/select 双向验证过的最优组合）
    ① 连通域面积过滤：剔除 |C| < min_area 的孤立小块
    ② 置信度门控细化：边界带像素阈值 + lambda

用法
----
    python make_submission.py \
        --ckpts a.pt b.pt c.pt d.pt e.pt \
        --val /path/to/leftImg8bit/val \
        --out result.zip \
        --tau 0.2 --min-area 50 --thin-lambda 0.05
"""


from __future__ import annotations
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from data import IMAGENET_MEAN, IMAGENET_STD, imread  # noqa: E402
from engine import pad_bottom  # noqa: E402
from model import ResNet34UNet  # noqa: E402

TAU = 0.2
MIN_AREA = 50
THIN_LAMBDA = 0.05


def area_filter(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """剔除面积小于 min_area 的 8-连通分量。"""
    if min_area <= 0 or mask_u8.sum() == 0:
        return mask_u8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            keep |= (labels == i).astype(np.uint8)
    return keep


def gated_thin(prob: np.ndarray, mask_u8: np.ndarray, tau: float, lam: float, r: int) -> np.ndarray:
    """边界带内提高阈值的软细化。

        Ŷ(x) = 1[ P(x) ≥ τ + λ·1{d(x) ≤ r} ]  ∧  x ∈ mask

    关键：结果**必须限制在原掩膜内**（`& mask_u8`）——
    否则 `prob ≥ τ` 会把掩膜外原本被面积过滤剔除的像素重新纳入。
    """
    if lam <= 0 or mask_u8.sum() == 0:
        return mask_u8
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    edge = (mask_u8 > 0) & (dist_in <= float(r))
    keep = (prob >= tau) & (~edge | (prob >= tau + float(lam)))
    # 严格限制在过滤后的掩膜内，防止面积过滤失去作用
    return (keep & (mask_u8 > 0)).astype(np.uint8)


class Ensemble:
    """K 个模型逐个前向，概率平均（不叠加显存）。"""

    def __init__(self, ckpts: list[Path], device: torch.device) -> None:
        self.device = device
        self.ckpts = list(ckpts)
        if not self.ckpts:
            raise RuntimeError("没有模型")

    def prob(self, image: torch.Tensor) -> torch.Tensor:
        h = image.shape[-2]
        x = pad_bottom(image.to(self.device))
        acc = None
        for ck in self.ckpts:
            m = ResNet34UNet(3, False).to(self.device)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(x)[:, :, :h])
            acc = p if acc is None else acc + p
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return acc / len(self.ckpts)

    @property
    def n(self) -> int:
        return len(self.ckpts)


def to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    x = rgb_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1))[None].copy()).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val", type=Path, required=True, help="leftImg8bit/val 目录")
    ap.add_argument("--out", type=Path, required=True, help="输出 result.zip")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--min-area", type=int, default=MIN_AREA, help="连通域最小面积（0=关闭）")
    ap.add_argument("--thin-lambda", type=float, default=THIN_LAMBDA, help="门控细化 λ（0=关闭）")
    ap.add_argument("--thin-r", type=int, default=1, help="边界带宽度（像素）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = Ensemble([Path(c) for c in args.ckpts], device)
    print(
        f"集成 K={ens.n}  τ={args.tau}  min_area={args.min_area}  "
        f"λ={args.thin_lambda}  r={args.thin_r}  device={device}",
        flush=True,
    )

    items: list[Path] = []
    for z in sorted(args.val.iterdir()):
        if z.is_dir():
            items += sorted(z.glob("*.jpg"))
    if args.limit:
        items = items[: args.limit]
    print(f"待推理 {len(items)} 张", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest, sizes, values = [], set(), set()

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, p in enumerate(items, 1):
            bgr = imread(p, True)
            if bgr is None:
                raise FileNotFoundError(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            prob = ens.prob(to_tensor(rgb))[0, 0].cpu().numpy()

            # 后处理①：面积过滤
            core = (prob >= args.tau).astype(np.uint8)
            core = area_filter(core, args.min_area)
            # 后处理②：门控细化
            final = gated_thin(prob, core, args.tau, args.thin_lambda, args.thin_r)

            assert final.shape == (h, w), f"尺寸不符 {final.shape} vs {(h, w)}"
            sizes.add((h, w))
            values.update(np.unique(final).tolist())

            ok, buf = cv2.imencode(".png", final)
            if not ok:
                raise RuntimeError(f"PNG 编码失败 {p}")
            zf.writestr(f"result/{p.stem}.png", buf.tobytes())
            manifest.append({"name": p.stem, "shape": [h, w], "crack_px": int(final.sum())})

            if i % 50 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)

    print(f"\n写出 {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")
    print(f"尺寸集合: {sizes}")
    print(f"像素值集合: {sorted(values)}")

    # ---------- 自检 ----------
    print("\n=== 提交前自检 ===")
    ok = True
    with zipfile.ZipFile(args.out) as zf:
        names = zf.namelist()

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("✓ " if cond else "✗ ") + msg)
        if not cond:
            ok = False

    chk(len(names) == len(items), f"条目数 {len(names)} == {len(items)}")
    chk(all(n.startswith("result/") for n in names), "全部位于 result/ 下")
    chk(all(n.lower().endswith(".png") for n in names), "全部为 .png")
    chk(sizes == {(378, 672)}, f"尺寸全部 (378,672)，实际 {sizes}")
    chk(values.issubset({0, 1}), f"像素值仅 0/1，实际 {sorted(values)}")

    src_stems = {p.stem for p in items}
    arc_stems = {Path(n).stem for n in names}
    chk(src_stems == arc_stems, f"文件名一一对应（差 {len(src_stems ^ arc_stems)} 个）")

    px = np.array([m["crack_px"] for m in manifest])
    chk(int((px == 0).sum()) == 0, f"无全空预测（实际 {(px == 0).sum()} 张）")
    chk(int((px == 378 * 672).sum()) == 0, f"无全满预测（实际 {(px == 378*672).sum()} 张）")
    print(f"\n裂缝像素：min {px.min()}  中位 {int(np.median(px))}  max {px.max()}")
    print(f"裂缝占比：{100*px.mean()/(378*672):.2f}%")

    print(f"\n{'全部通过，可提交' if ok else '存在问题，请检查'}")

    man = args.out.with_suffix(".manifest.json")
    man.write_text(
        json.dumps(
            {"n": len(manifest), "tau": args.tau, "K": ens.n,
             "min_area": args.min_area, "thin_lambda": args.thin_lambda,
             "items": manifest},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"清单已存 {man}")


if __name__ == "__main__":
    main()
