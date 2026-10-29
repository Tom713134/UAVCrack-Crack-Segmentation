"""对已生成的 result.zip 做提交前复核（不需要权重/数据集）。

用法
----
    python verify_result.py --zip ../result.zip --val /path/to/leftImg8bit/val
    python verify_result.py --zip ../result.zip                # 跳过文件名比对

校验 8 项（与 make_submission.py 自检一致，额外支持对已生成包离线复核）：
    1. 条目数 = 300
    2. 全部位于 result/ 下
    3. 全部为 .png
    4. 无多余层级（仅 result/<name>.png）
    5. 尺寸全部 (378, 672)
    6. 像素值仅 {0, 1}
    7. 与 val 输入文件名一一对应
    8. 无全空 / 全满预测
"""

from __future__ import annotations
import argparse
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

H, W = 378, 672


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None, help="leftImg8bit/val 目录（可选）")
    ap.add_argument("--expect", type=int, default=300)
    args = ap.parse_args()

    p = args.zip
    print(f"文件 : {p}")
    print(f"大小 : {p.stat().st_size/1048576:.4f} MB ({p.stat().st_size} 字节)")

    z = zipfile.ZipFile(p)
    names = z.namelist()
    shapes, vals, px = set(), set(), []
    for n in names:
        im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
        shapes.add(im.shape)
        vals |= set(np.unique(im).tolist())
        px.append(int((im > 0).sum()))
    px = np.array(px)

    val_stems = None
    if args.val and args.val.is_dir():
        val_stems = set()
        for d in sorted(args.val.iterdir()):
            if d.is_dir():
                val_stems |= {f.stem for f in d.glob("*.jpg")}

    checks = [
        (f"条目数 = {args.expect}", len(names) == args.expect, len(names)),
        ("全部位于 result/ 下", all(n.startswith("result/") for n in names), ""),
        ("全部为 .png", all(n.lower().endswith(".png") for n in names), ""),
        ("无多余层级", all(n.count("/") == 1 for n in names), ""),
        (f"尺寸全部 ({H},{W})", shapes == {(H, W)}, shapes),
        ("像素值仅 {0,1}", vals <= {0, 1}, sorted(vals)),
        ("无全空 / 全满", bool((px > 0).all() and (px < H * W).all()),
         f"空 {(px == 0).sum()} / 满 {(px == H*W).sum()}"),
    ]
    if val_stems is not None:
        arc = {Path(n).stem for n in names}
        ok = arc == val_stems
        checks.append(("与 val 输入一一对应", ok,
                       f"zip {len(arc)} / val {len(val_stems)}，差 {len(arc ^ val_stems)}"))

    print("\n=== 提交前复核 ===")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= bool(ok)
        print(("✓ " if ok else "✗ ") + name + (f"  [{detail}]" if detail != "" else ""))

    print(f"\n裂缝像素：min {px.min()}  中位 {int(np.median(px))}  max {px.max()}")
    print(f"裂缝占比：{100*px.mean()/(H*W):.2f}%")
    print(f"\n{'全部通过，可提交' if all_ok else '存在问题，请检查'}")


if __name__ == "__main__":
    main()
