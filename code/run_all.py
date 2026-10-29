"""一键复现最优结果：训练 → 集成 → 后处理 → 评估 → 打包提交。

用法
----
    # 完整流程（从零训练 5 个模型，约 60 分钟）
    python run_all.py --data /path/to/UAV-CrackX --mode full

    # 已有权重，只做集成+后处理+打包（约 2 分钟）
    python run_all.py --data /path/to/UAV-CrackX --mode eval --ckpts ../ckpt/seed*.pt

    # 只测内部指标（需要 GT，用 confirm 划分）
    python run_all.py --data /path/to/UAV-CrackX --mode eval --ckpts ... --eval-split confirm

三个 mode
---------
    full  : 用全部 1177 张训练（VLP 提交用），然后对 val 推理出 result.zip
    eval  : 用 831 张训练 + confirm 留出（内部评测），报六项指标
    infer : 只推理（已有权重），产出 result.zip
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [101, 202, 303, 404, 505]
TAU = 0.2
MIN_AREA = 50
THIN_LAMBDA = 0.05


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*70}\n{desc}\n{'='*70}", flush=True)
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *[str(c) for c in cmd]], cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"失败：{desc}（exit {r.returncode}）")
    print(f"完成，用时 {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="UAV-Crack 终版一键复现")
    ap.add_argument("--data", type=Path, required=True, help="UAV-CrackX 根目录")
    ap.add_argument("--ref", type=Path, default=None, help="原始数据根（清洗时用，可选）")
    ap.add_argument("--mode", default="eval", choices=["full", "eval", "infer"])
    ap.add_argument("--ckpts", nargs="*", default=None, help="已有权重（infer/eval 用）")
    ap.add_argument("--out", type=Path, default=HERE.parent / "output")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-split", default="confirm", choices=["confirm", "select", "val"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log: dict = {"mode": args.mode, "seeds": SEEDS, "tau": TAU,
                 "min_area": MIN_AREA, "thin_lambda": THIN_LAMBDA}

    # ---------- 1. 训练 ----------
    if args.mode in ("full", "eval"):
        script = "train_vlp.py" if args.mode == "full" else "train.py"
        print(f"\n使用 {script}（{'全量 1177 张' if args.mode=='full' else '831 张 + confirm 留出'}）")
        for s in SEEDS:
            if args.mode == "full":
                run(
                    [HERE / script, "--seeds", s, "--out", ckpt_dir,
                     "--epochs", args.epochs, "--batch", args.batch,
                     "--workers", args.workers, "--data", args.data],
                    f"训练 seed {s}",
                )
            else:
                run(
                    [HERE / script, "--data", args.data, "--out", ckpt_dir / f"seed{s}",
                     "--seed", s, "--epochs", args.epochs, "--batch", args.batch,
                     "--lr", "3e-4", "--workers", args.workers],
                    f"训练 seed {s}",
                )
        if args.mode == "full":
            ckpts = sorted(ckpt_dir.glob("seed*.pt"))
        else:
            ckpts = sorted(ckpt_dir.glob("seed*/seed*.pt"))
    else:
        if not args.ckpts:
            raise SystemExit("--mode infer/eval 需要 --ckpts")
        ckpts = [Path(c) for c in args.ckpts]
        if not ckpts:
            raise SystemExit("没有找到权重")

    print(f"\n共 {len(ckpts)} 个权重：")
    for c in ckpts:
        print(f"  {c}")

    # ---------- 2. 内部评测（有 GT）----------
    if args.mode == "eval":
        run(
            [HERE / "eval_detailed.py", "--ckpts", *ckpts,
             "--data", args.data, "--part", args.eval_split,
             "--tau", TAU, "--min-area", MIN_AREA, "--thin-lambda", THIN_LAMBDA,
             "--out", args.out / "eval_result.json"],
            f"内部评测（{args.eval_split}）",
        )
        p = args.out / "eval_result.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            for k in ("summary_tau02", "by_zoom_tau02"):
                if k in d:
                    print(f"\n{k}:")
                    print(json.dumps(d[k], ensure_ascii=False, indent=1))
            log["eval"] = {k: d.get(k) for k in ("summary_tau02", "by_zoom_tau02")}

    # ---------- 3. 生成提交包 ----------
    if args.mode in ("full", "infer", "eval"):
        val_dir = args.data / "leftImg8bit" / "val"
        if val_dir.is_dir():
            run(
                [HERE / "make_submission.py", "--ckpts", *ckpts,
                 "--val", val_dir, "--out", args.out / "result.zip",
                 "--tau", TAU, "--min-area", MIN_AREA, "--thin-lambda", THIN_LAMBDA],
                "生成 VLP 提交包 result.zip",
            )
            log["submission"] = str(args.out / "result.zip")
        else:
            print(f"\n跳过提交包：未找到 {val_dir}")

    (args.out / "run_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n{'='*70}\n全部完成\n{'='*70}")
    print(f"输出目录：{args.out}")
    print(f"日志：{args.out / 'run_log.json'}")


if __name__ == "__main__":
    main()
