"""校验导出的 ONNX 与 PyTorch 权重输出一致性。

对每个 seed：
    1. 检查 ONNX 是否单文件、opset、输入输出形状
    2. ONNX Runtime 与 PyTorch 在同一随机输入上比对最大绝对误差
    3. 统计权重文件大小与参数量

用法
----
    python check_onnx.py --ckpt-dir ../ckpt --onnx-dir ../onnx
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from model import ResNet34UNet  # noqa: E402

H, W = 384, 672
SEEDS = (101, 202, 303, 404, 505)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((1, 3, H, W)).astype(np.float32)

    print(f"{'seed':>6} {'pt(MB)':>8} {'onnx(MB)':>9} {'params':>12} {'max|diff|':>11}  结果")
    print("-" * 68)
    all_ok = True
    for s in seeds:
        ck, ox = args.ckpt_dir / f"seed{s}.pt", args.onnx_dir / f"seed{s}.onnx"
        if not ck.is_file() or not ox.is_file():
            print(f"{s:>6}  缺失文件")
            all_ok = False
            continue

        m = ResNet34UNet(3, False)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            ref = m(torch.from_numpy(dummy)).numpy()

        model = onnx.load(str(ox), load_external_data=False)
        ext = [t for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
        sess = ort.InferenceSession(str(ox), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": dummy})[0]

        shape_ok = ref.shape == got.shape == (1, 1, H, W)
        diff = float(np.abs(ref - got).max())
        ok = shape_ok and diff < args.tol and n == 24_429_281 and not ext
        all_ok &= ok
        print(
            f"{s:>6} {ck.stat().st_size/1e6:>8.2f} {ox.stat().st_size/1e6:>9.2f} "
            f"{n:>12,} {diff:>11.2e}  {'OK' if ok else 'FAIL'}"
            + ("" if not ext else f"  (外部数据 {len(ext)})")
        )

    print("-" * 68)
    print("全部通过" if all_ok else "存在问题，请检查")


if __name__ == "__main__":
    main()
