"""包内自检：不依赖数据集，验证代码可运行。

用法
----
    python selfcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch


def sec(t: str) -> None:
    print(f"\n{'='*66}\n{t}\n{'='*66}")


def main() -> None:
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            ok = False

    sec("1. 模块导入")
    mods = [
        "model", "losses", "data", "split_data", "engine", "clean_manifest",
        "make_submission", "eval_detailed", "gated_thin", "postproc_search",
        "infer_ensemble", "hard_priors", "boundary_utils", "train", "train_vlp",
    ]
    for m in mods:
        try:
            __import__(m)
            print(f"  ✓ {m}")
        except Exception as e:
            print(f"  ✗ {m}: {type(e).__name__}: {e}")
            ok = False

    sec("2. 模型结构")
    from model import ResNet34UNet

    m = ResNet34UNet(3, False)
    n = sum(p.numel() for p in m.parameters())
    print(f"  参数量 = {n:,}")
    chk(n == 24_429_281, f"参数量应为 24,429,281，实际 {n:,}")
    x = torch.randn(1, 3, 378, 672)
    with torch.no_grad():
        y = m(x)
    chk(tuple(y.shape) == (1, 1, 378, 672), f"输出形状 {tuple(y.shape)}")
    p0 = float(torch.sigmoid(y).mean())
    print(f"  初始平均概率 = {p0:.4f}（应接近裂缝先验 0.02）")
    chk(0.005 < p0 < 0.05, "初始概率在合理区间")

    sec("3. 损失函数")
    from losses import l0_cal_dice

    logits = torch.zeros(2, 1, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 30:34, 10:54] = 1.0
    L5 = float(l0_cal_dice(logits, target))
    with torch.no_grad():
        L0 = float(l0_cal_dice(logits - 3.0, target))
    print(f"  L(p=0.5) = {L5:.4f}   L(p≈0.05) = {L0:.4f}")
    chk(L0 < L5, "降低预测概率后损失应下降（背景压制生效）")

    sec("4. 增广配置")
    from data import GEO_CATALOG, TRAIN_EXPAND_FACTOR

    print(f"  几何视角 {len(GEO_CATALOG)} 个，EXPAND={TRAIN_EXPAND_FACTOR}")
    for i, (s, a, f) in enumerate(GEO_CATALOG):
        print(f"    view {i}: scale={s:.2f} angle={a:+.0f}° flip={f}")
    chk(TRAIN_EXPAND_FACTOR == 5, "几何倍数应为 5")
    print(f"  例：831 张 → {831*5} 样本/epoch；1177 张 → {1177*5}")

    sec("5. 后处理算子")
    from make_submission import area_filter, gated_thin

    # 造一个 200px 的大团块 + 20px 的小团块
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:20] = 1        # 200 px
    mask[50:54, 50:55] = 1        # 20 px
    out = area_filter(mask, min_area=50)
    chk(int(out.sum()) == 200, f"面积过滤：保留 200 px、剔除 20 px，实际 {int(out.sum())}")

    prob = np.zeros((64, 64), np.float32)
    prob[mask > 0] = 0.6           # 核心高置信
    prob[29:32, 10:20] = 0.22      # 边界区贴阈值
    core = area_filter((prob >= 0.2).astype(np.uint8), 50)
    thinned = gated_thin(prob, core, 0.2, 0.05, 1)
    chk(int(thinned.sum()) < int(core.sum()), f"门控细化应削减像素 {int(core.sum())} → {int(thinned.sum())}")

    sec("6. 指标公式")
    from eval_detailed import agg

    rows = [{"tp": 100, "fp": 20, "fn": 30, "tn": 1000, "gt_px": 130}]
    m6 = agg(rows)
    for k in ("Crack IoU", "Crack Precision", "Crack Recall", "Crack F1", "mIoU", "aAcc"):
        print(f"  {k:16s} = {m6[k]:.4f}")
    chk(abs(m6["Crack IoU"] - 66.6667) < 0.01, "IoU = 100/(100+20+30) = 66.67")
    chk(abs(m6["Crack Precision"] - 83.3333) < 0.01, "P = 100/120 = 83.33")

    sec("7. 划分逻辑（需数据目录才能真跑）")
    from split_data import frame_key, split_zoom

    name = "937956_DJI_20231015154901_0059_Z.JPG_7_z.jpg"
    fk = frame_key(name)
    print(f"  {name}\n  → frame_key = {fk}")
    chk(fk == "937956_DJI_20231015154901_0059_Z.JPG", "帧标识提取正确")
    names = [f"X_DJI_2023010100000{i:02d}_0000_Z.JPG_{j}_z.jpg"
             for i in range(20) for j in range(5)]
    parts = split_zoom(sorted(names))
    tot = sum(len(v) for v in parts.values())
    print(f"  100 张 → train {len(parts['train'])} / select {len(parts['select'])} "
          f"/ confirm {len(parts['confirm'])}")
    chk(tot == 100, "划分不丢样本")

    sec("结论")
    print("全部通过，代码可用" if ok else "存在问题，见上方 ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
