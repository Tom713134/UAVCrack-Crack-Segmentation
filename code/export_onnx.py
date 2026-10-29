"""导出 ONNX 模型（2026 VLP 挑战赛要求：PyTorch 模型需额外提供 ONNX 格式）。

用法
----
    # 导出单个权重
    python export_onnx.py --ckpt ckpt/seed101.pt --out ckpt/seed101.onnx

    # 导出全部 5 个 seed
    for s in 101 202 303 404 505; do
        python export_onnx.py --ckpt ckpt/seed$s.pt --out ckpt/seed$s.onnx
    done

输入 / 输出
-----------
    输入  : input       float32  (1, 3, 384, 672)   已归一化 + pad_bottom 到 384
    输出  : logits      float32  (1, 1, 384, 672)   推理时取前 378 行再做 sigmoid

注意
----
1. 模型内部有双线性上采样到输入尺寸，输入高度必须是 32 的倍数（用 384），
   推理后裁回 378。若直接输入 378 会因 5 次下采样尺寸不整除而报错。
2. 导出后权重会内嵌为**单文件** .onnx（约 93 MB），不会产生 .onnx.data 附属文件。
3. 批大小固定为 1（竞赛逐个样本推理），已去掉 dynamic_axes 以兼容 dynamo 导出路径。
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from model import ResNet34UNet  # noqa: E402

# 与 engine.pad_bottom 一致的高度
ONNX_H, ONNX_W = 384, 672


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="PyTorch 权重 .pt")
    ap.add_argument("--out", type=Path, required=True, help="输出 .onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--check", action="store_true", help="导出后与 PyTorch 输出比对")
    args = ap.parse_args()

    model = ResNet34UNet(3, False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"权重 {args.ckpt}")
    print(f"参数量 {n:,}")
    if n != 24_429_281:
        print(f"警告：参数量与终版（24,429,281）不一致", file=sys.stderr)

    dummy = torch.randn(1, 3, ONNX_H, ONNX_W)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    # torch 默认把大权重存成 <name>.onnx.data 外部文件，分发时易丢失；
    # 重新保存为单文件（权重内嵌），便于上传 / 拷贝。
    try:
        import onnx
        m = onnx.load(str(args.out))
        onnx.save(m, str(args.out), save_as_external_data=False)
        sidecar = args.out.with_suffix(args.out.suffix + ".data")
        if sidecar.is_file():
            sidecar.unlink()
            print(f"已删除外部权重文件 {sidecar.name}")
    except ImportError:
        print("未安装 onnx，跳过单文件化（可能是 .onnx + .onnx.data 两文件）", file=sys.stderr)
    print(f"已导出 {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")

    if args.check:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("跳过比对：需要 pip install onnx onnxruntime", file=sys.stderr)
            return
        onnx.checker.check_model(onnx.load(str(args.out)))
        with torch.no_grad():
            ref = model(dummy).numpy()
        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(abs(ref - got).max())
        print(f"ONNX 校验通过；与 PyTorch 最大绝对误差 = {diff:.3e}")
        if diff > 1e-3:
            print("警告：误差偏大，请检查 opset 或算子支持", file=sys.stderr)


if __name__ == "__main__":
    main()
