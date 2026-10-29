# VLP 平台提交说明（UAV-Crack 任务）

来源：<http://vlp.chd.edu.cn/tasks/4>

---

## 1. 提交要求（官方原文摘要）

| 项 | 要求 |
|---|---|
| 评测对象 | `leftimg8bit/val/` 下 **300 张**图 |
| 分辨率 | 必须与输入一致：**672×378**（宽×高，即 H=378, W=672） |
| 文件名 | 与输入 `.jpg` **完全一致**，仅扩展名改为 `.png` |
| 像素值 | **0 = 路面，1 = 裂缝**（uint8 单通道） |
| 压缩包名 | **`result.zip`** |
| 内部结构 | `result/<name>.png`，**只含这 300 张**，不得有其他文件 |

官方给出的示例：

```
result/
├── 937956_DJI_20231015154707_0002_Z.JPG_12_z.png
├── 937956_DJI_20231015154713_0005_Z.JPG_5_z.png
├── 937956_DJI_20231015154717_0007_Z.JPG_7_z.png
└── ...
```

> **Important**: Do not include any extra files or images in the submission zip. Only include the 300 required predicted mask images. Store 300 mask images in a folder named 'result'.

---

## 2. 评测指标（六项）

| 指标 | 定义 |
|---|---|
| mIoU | Mean IoU = (Crack IoU + Background IoU) / 2 |
| Crack F1 | $2PR/(P+R)$ |
| Crack IoU | $TP/(TP+FP+FN)$ |
| Crack Precision | $TP/(TP+FP)$ |
| Crack Recall | $TP/(TP+FN)$ |
| aAcc | $(TP+TN)/(TP+TN+FP+FN)$ |

---

## 3. 本项目的提交方案

### 3.1 训练数据的变化

| | 之前（内部评测版） | **现在（VLP 提交版）** |
|---|---|---|
| 训练集 | 831 张（1177 的 70%） | **1177 张（全部）** |
| select | 153 张（早停用） | **无** |
| confirm | 193 张（报告用） | **无** |
| 早停 | 按 select IoU 选最优 epoch | **固定跑满 30 epoch** |
| 模型数 | 5（seed 101/202/303/404/505） | 5（同） |

**理由**：VLP 用独立的 300 张 val 做官方评测，内部再留 holdout 只会浪费 30% 的标注数据。全量训练是竞赛提交的标准做法。

**代价**：失去了早停保护。依据历史实验，30 epoch 时 select IoU 仍在平台区（未过拟合），所以取最后 epoch 是安全的。

### 3.2 数据重叠的重要发现

检查发现 `leftImg8bit/val` 的 300 张与训练集存在**帧级重叠**：

| 项 | 数量 |
|---|---|
| 训练集独立帧 | 197 |
| val 独立帧 | 142 |
| **重叠帧** | **137** |
| 文件名重叠 | 0（不同 patch） |

即 val 中有 **137/142 = 96.5%** 的帧在训练集里出现过（不同 patch）。

$$
\text{val 的 300 张中，绝大多数来自训练集同一批飞行帧}
$$

**影响**：
- 官方指标会偏高（模型见过同帧的相邻 patch，存在空间相关性）；
- 这**不是我们造成的问题**，是数据集划分本身如此；
- 但意味着 VLP 上的分数**不应与 internal confirm 的 65.37 直接比较**。

**我们无法改变这一点**（val 的 GT 不可见，且必须提交全 300 张）。

### 3.3 推理配置

$$
P(\mathbf{x})=\frac{1}{5}\sum_{k=1}^{5}\sigma\bigl(f_{\theta_k}(\mathrm{norm}(I))(\mathbf{x})\bigr),
\qquad
\hat Y=\mathbb{1}[P\ge 0.2]
$$

- 5 个模型概率平均（与内部评测版一致）
- $\tau=0.2$（CAL 训练出的概率分布偏保守，0.5 会严重漏检）
- 输入按 `pad_bottom` 补到 384 推理，输出裁回 378

---

## 4. 生成提交包的流程

```bash
# 1. 训练 5 个模型（全量 1177 张）
python train_vlp.py --seeds 101 --out <dir>   # 重复 5 次，或 --seeds 101,202,... 串行

# 2. 生成 result.zip
python make_submission.py \
  --ckpts <dir>/seed101.pt <dir>/seed202.pt <dir>/seed303.pt \
          <dir>/seed404.pt <dir>/seed505.pt \
  --val /path/to/UAV-CrackX/leftImg8bit/val \
  --out result.zip \
  --tau 0.2
```

`make_submission.py` 会自动做 **6 项自检**：

1. 条目数 = 300
2. 全部位于 `result/` 下
3. 全部为 `.png`
4. 尺寸全部 378×672
5. 像素值仅 {0, 1}
6. 文件名与 val 输入一一对应

全部通过才打印「可提交」。同时生成 `result.manifest.json`（含每张图的裂缝像素数，便于复查）。

---

## 5. 提交包结构

```
result.zip
└── result/
    ├── 937958_DJI_20231015160532_0003_Z.JPG_2_z.png
    ├── ...
    └── (共 300 个 .png)
```

附带生成：

```
result.manifest.json      # 300 条记录：arcname / src / shape / crack_px
```

---

## 6. 已知风险

| 风险 | 说明 | 应对 |
|---|---|---|
| **val 与训练集帧重叠 96.5%** | 官方分数会偏高，不代表泛化能力 | 报告中明确说明 |
| 无早停 | 固定 30 epoch，可能非最优 | 历史曲线显示 30 epoch 仍在平台区 |
| 无内部验证 | 无法预知提交分数 | 可先用 4 个模型训练 + 1 个 holdout 快速验证流程 |
| 分辨率约定 | 官方写 672×378（宽×高），实际 H=378, W=672 | 我们的输出 `mask.shape == (h, w) = (378, 672)`，与输入一致 |
| 像素值 | 官方要 0/1，但很多工具显示为全黑 | 已用 uint8 0/1；若平台按 0/255 解析会全判为裂缝，需按官方要求保持不变 |

---

## 7. 若需回退到内部评测版

内部评测版（confirm 65.37）的权重在：

```
/root/autodl-tmp/uav_upload_20261028/exp_r5_ens/seed{101,202,303,404,505}.pt
```

区别仅是**训练数据少了 346 张**（831 vs 1177）。两者推理代码完全相同，可互换。
