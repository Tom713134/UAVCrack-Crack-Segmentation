# UAV-Crack 终版代码包

> **2026 VLP 挑战赛参赛作品**
>
> 赛道 D：UAV-Crack — 基于无人机的路面裂缝分割
>
> ResNet34-UNet 五模型概率集成 + 两级后处理 · Crack IoU **65.51**（内部 confirm 193 张，micro，有 GT）
>
> 本仓库为**代码与文档**。权重不随仓库分发，获取方式见 §4.2；数据集遵循 CC BY-NC-SA 3.0，需向 VLP 平台申请。
> 生成时间：2026-10-29

---

## 0. 环境

```bash
pip install -r requirements.txt      # torch / torchvision / numpy / opencv-python
python code/selfcheck.py             # 无需数据，约 30 秒，应输出「全部通过，代码可用」
```

Python ≥ 3.9。建议 CUDA GPU（CPU 训练慢约 20 倍，推理慢约 20 倍）。

---

## 1. 一分钟了解

**训练**（5 个 seed，与定稿唯一差别是随机种子）

$$
L=\mathrm{CAL}(\delta{=}0.5,\gamma{=}0.2,\lambda{=}0.5)+0.5\cdot\mathrm{Dice}
$$

**推理**（三步）

$$
P(\mathbf{x})=\frac{1}{5}\sum_{k=1}^{5}\sigma\bigl(f_{\theta_k}(\mathrm{norm}(I))(\mathbf{x})\bigr)
$$

$$
M=\bigcup_{|C_i|\ge 50}C_i
\qquad\text{（连通域面积过滤）}
$$

$$
\hat Y(\mathbf{x})=\mathbb{1}\Bigl[P(\mathbf{x})\ge0.2+0.05\cdot\mathbb{1}\{d_M(\mathbf{x})\le1\}\Bigr]\wedge M
\qquad\text{（门控细化）}
$$

---

## 2. 目录结构

```
UAVCrack_终版代码包/
├── README.md                 本文件
├── code/
│   ├── selfcheck.py          ← 先跑这个，无需数据
│   ├── run_all.py            ← 一键流程（训练→评测→打包）
│   │
│   ├── 【核心】─────────────────────────
│   ├── model.py              ResNet34-UNet (24,429,281 参数)
│   ├── losses.py             CAL + Dice
│   ├── data.py               数据集 + 几何×5 增广
│   ├── split_data.py         按帧划分（含冻结 confirm）
│   ├── clean_manifest.py     23 张剔除清单
│   ├── engine.py             pad_bottom / evaluate / 阈值扫描
│   │
│   ├── 【训练】─────────────────────────
│   ├── train.py              内部评测版（831 train + select 早停）
│   ├── train_vlp.py          VLP 全量版（1177 train，无 holdout）
│   │
│   ├── 【推理与后处理】─────────────────
│   ├── infer_ensemble.py     集成推理（单图/批量）
│   ├── make_submission.py    ← 生成 result.zip + 8 项自检
│   ├── eval_detailed.py      ← 六项指标 + 逐图分布
│   ├── gated_thin.py         门控细化（实验版实现）
│   ├── postproc_search.py    后处理参数网格搜索
│   └── experimental/         负结果留存（见 §7）
└── docs/
    ├── 数学原理.md
    └── 实验总表.md
```

---

## 3. 快速开始

### 3.1 先自检（不需要数据，约 30 秒）

```bash
cd code
python selfcheck.py
```

预期输出「全部通过，代码可用」，检查 7 项：模块导入、模型结构、损失函数、增广配置、后处理算子、指标公式、划分逻辑。

### 3.2 完整流程

```bash
# A. 全量训练 + 生成提交包（约 60 分钟，需 GPU）
python run_all.py --data /path/to/UAV-CrackX --mode full

# B. 训练 + 内部评测（831 train / confirm 留出，约 60 分钟）
python run_all.py --data /path/to/UAV-CrackX --mode eval

# C. 已有权重，只做推理（约 2 分钟）
python run_all.py --data /path/to/UAV-CrackX --mode infer --ckpts ../ckpt/*.pt
```

输出在 `../output/`：

```
output/
├── ckpt/               5 个权重
├── eval_result.json    六项指标 + 逐图
├── result.zip          ← VLP 提交包
└── run_log.json
```

### 3.3 分步执行

```bash
# 训练 1 个模型
python train_vlp.py --seeds 101 --out ./ckpt --epochs 30 --batch 10 --workers 8 --data /path/to/UAV-CrackX

# 内部评测（6 指标）
python eval_detailed.py --ckpts ./ckpt/seed101.pt ./ckpt/seed202.pt \
  --data /path/to/UAV-CrackX --part confirm \
  --tau 0.2 --min-area 50 --thin-lambda 0.05 --out eval.json

# 生成提交包
python make_submission.py --ckpts ./ckpt/*.pt \
  --val /path/to/UAV-CrackX/leftImg8bit/val \
  --out result.zip --tau 0.2 --min-area 50 --thin-lambda 0.05
```

---

## 4. 需要自备的东西

### 4.1 数据集

Cityscapes 风格，**图像尺寸必须是 378×672**（H=378, W=672）：

```
UAV-CrackX/
├── leftImg8bit/
│   ├── train/{UAV-CrackX4,UAV-CrackX8,UAV-CrackX16}/*.jpg   各 400 张
│   └── val/{UAV-CrackX4,UAV-CrackX8,UAV-CrackX16}/*.jpg     各 100 张（无 GT）
└── gtFine/
    └── train/{UAV-CrackX4,UAV-CrackX8,UAV-CrackX16}/*.png   0=路面 1=裂缝
```

**注意**：GT 是 **uint8，值域 {0, 1}**，不是 0/255。

### 4.2 权重

> **本仓库不分发权重**（5 × 93 MB = 466 MB，超出 Git 仓库合理体积）。需自行训练或向作者索取。

训练好的 5 个权重路径（本地）：

```
ckpt/seed{101,202,303,404,505}.pt      ← 全量 1177 训练（提交用，对应 65.51）
../人工智能大赛/交付_终版/ckpt/seed{101,202,303,404,505}.pt  ← 831 训练（可复现 65.37）
```

两组是**不同的模型**，不能混用。每个权重 97,871,979 字节，加载后参数量应为 24,429,281。

**没有权重时的两条路**：

```bash
# A. 自行训练（约 60 分钟 / 5 个模型，需 GPU）
python run_all.py --data /path/to/UAV-CrackX --mode full

# B. 只用 1 个模型快速验证流程（指标约低 0.8）
python make_submission.py --ckpts single.pt --val /path/to/leftImg8bit/val --out result.zip
```

### 4.3 数据路径

代码默认从环境变量或相对路径读取：

```bash
# 方式一：环境变量
export UAV_CLEAN_DIR=/path/to/UAV-CrackX-clean
export UAV_REF_DIR=/path/to/UAV-CrackX

# 方式二：命令行参数（推荐）
python train_vlp.py --data /path/to/UAV-CrackX ...
```

---

## 5. 关键参数

```python
# 数据
IMAGE_SIZE      = (378, 672)      # (H, W)，不是 384×384
TRAIN_EXPAND    = 5               # 几何视角数
GEO_CATALOG     = [(1.00, 0, False), (0.85, -10, False), (1.15, 10, False),
                   (1.00, 0, True), (0.90, 5, True)]

# 模型
ARCH            = "ResNet34-UNet"
PARAMS          = 24_429_281
HEAD_BIAS_INIT  = -4.0            # σ(-4)=0.018 ≈ 裂缝先验

# 损失
CAL_DELTA, CAL_GAMMA, CAL_LAMBDA = 0.5, 0.2, 0.5
DICE_WEIGHT     = 0.5

# 训练
EPOCHS, BATCH, LR, WEIGHT_DECAY = 30, 10, 3e-4, 1e-4
GRAD_CLIP       = 1.0

# 推理
TAU             = 0.2
MIN_AREA        = 50              # 连通域最小面积
THIN_LAMBDA     = 0.05            # 门控细化强度
THIN_R          = 1               # 边界带宽度

# 集成
N_MODELS        = 5
SEEDS           = [101, 202, 303, 404, 505]
```

---

## 6. 指标参考

### 6.1 内部评测（confirm 193 张，有 GT）

| 方案 | Crack IoU | P | R |
|---|---|---|---|
| 单模型 | 64.58 | 79.08 | 77.89 |
| 5-seed 集成 | 65.37 | 76.65 | 81.63 |
| + 面积过滤 (a≥50) | 65.41 | — | — |
| **+ 门控细化 (λ=0.05)** | **65.51** | 77.3 | 81.3 |

> 精确值：单模型 **64.5796**、集成 **65.3719**、完整方案 **65.5053**（累计 +0.9257）。
> 早期文档中出现的 "65.55" 系四舍五入笔误，以 65.5053 为准。

分变焦（5-seed 集成）：

| 变焦 | Crack IoU | P | R |
|---|---|---|---|
| X4 | 63.24 | 70.34 | 86.24 |
| X8 | 66.23 | 78.00 | 81.44 |
| X16 | 66.04 | 80.38 | 78.73 |

### 6.2 必须知道的局限

| 问题 | 说明 |
|---|---|
| **select 与 confirm 差 6 点** | select 59.4 / confirm 65.4。划分按 DJI 时间戳取末尾 15%，那批帧更易 |
| **val 与训练集帧重叠 96.5%** | 137/142 帧相同（不同 patch），官方分会偏高 |
| **X4 是瓶颈** | 比 X8/X16 低 3 点；FP 有 72.7% 贴 GT 3px 内（"画粗"非"混淆"） |
| **300 张无标注图未用** | 半监督两次失败；标注它们是最大潜力 |

---

## 7. experimental/ 目录说明

这些是**失败方向**的完整实现，保留备查，**不建议再跑**：

| 文件 | 方法 | 结果 |
|---|---|---|
| `train_ema.py` | EMA 权重平均 | 65.41 vs 65.43（**−0.015**）|
| `train_zoom.py` | 分变焦专用模型 | X4 62.81 vs 63.24（**−0.43**）|
| `lovasz.py` | Lovász-Softmax | w=0.5 → 62.81；w=0.03 → 64.85 |
| `round2_losses.py` | 难负挖掘 / 骨架+护栏 | −0.50 / −0.32 |
| `round3_methods.py` | 多尺度 TTA / 边缘歧义排除 | **−5.25** / −0.31 |
| `round4_x4.py` | 双分辨率输入 / 分变焦 BN | −0.55 / −0.61 |
| `run_round5.py` | 多种子集成训练 | **+0.79**（这个有效）|
| `run_final.py` | 通用训练入口 | 基准 |

**判定标尺**：$\sigma_{\text{seed}}=0.24$，任何 $|\Delta|<0.48$ 的差异都是噪声。

---

## 8. 常见问题

### Q1 指标比 65.51 低 1–2 点

按可能性排查：

1. 图像不是 378×672（`pad_bottom(384)` 假设 H=378）
2. 划分没按帧分组（同帧 patch 跨集会严重泄漏）
3. `head.bias` 未设为 −4
4. 用的权重是「831 训练」而非「1177 全量」（或反之）
5. 后处理顺序错（必须先面积过滤，再门控细化）

### Q2 门控细化让结果变差

检查 `gated_thin` 是否限制了 `& mask`。若不限制，`prob >= tau` 会把面积过滤剔除的像素重新纳入，后处理一失效。

```python
# 必须这样
return (keep & (mask_u8 > 0)).astype(np.uint8)
```

### Q3 loss 下降但 IoU 不涨

正常。CAL 的负类项让背景越判越对，loss 持续下降，但 IoU 由正类决定。实测 loss 0.50→0.24（−52%），IoU 只从 64.4→64.8。

**判据应看 select IoU，不看 loss。**

### Q4 想再提升

按性价比：

| 方向 | 预期 | 成本 | 依据 |
|---|---|---|---|
| 标注 300 张无标注 val | +1~2 | 人工 | 数据量是唯一反复验证有效的因素 |
| 扩集成到 K=10 并筛选成员 | +0.1 | 5 次训练 | K 曲线实测 |
| ImageNet 预训练（若禁用则相反）| 未知 | 需重训 | 当前已用 |

**不建议**：调损失权重（6 次全失败）、多尺度 TTA（−5.25）、半监督（2 次崩溃）、分变焦阈值（方向矛盾）。

---

## 9. 完整流程（技术细节）

见 `docs/数学原理.md`，涵盖：

1. 问题定义与数据集
2. 符号表
3. 数据划分：为什么必须按帧分组（**空间泄漏推导**）
4. 数据增广：几何×5 的数学含义
5. 网络结构（**head bias = −4 的推导**）
6. 损失函数：CAL + Dice（**$p^\gamma$ 负类加权的机制**）
7. 训练过程（**早停 0.5 而评测 0.2 的原因**）
8. 模型集成：**$\mathrm{Var}(\bar p)=\sigma^2[\rho+\frac{1-\rho}{K}]$ 完整推导**
9. 后处理一：连通域面积过滤
10. 后处理二：置信度门控细化（**与骨架重建失败的对比**）
11. 完整数据流图
12. 评测指标（micro vs macro）
13. 11 条关键设计决策与实验依据
14. 6 项已知局限
15. 复现检查清单

---

## 10. 引用

- **CAL 损失**：Shan et al., *UAV-Based Pavement Image Stitching without Occlusion, Crack Semantic Segmentation, and Quantification*, IEEE T-ITS 25(11):17038–17053, 2024.
- **数据集**：UAV-CrackX，<https://github.com/SHAN-JH/UAV-CrackX-Datasets>
- **评测平台**：VLP，<http://vlp.chd.edu.cn/tasks/4>
