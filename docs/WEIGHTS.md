# 权重文件说明（WEIGHTS）

> 2026 VLP 挑战赛 · 赛道 D（UAV-Crack）· 本仓库为 <https://github.com/Tom713134/UAVCrack-Crack-Segmentation>

**下载链接**：[待填写 — 见文末"下载地址"一节]

---

## 1. 文件清单

| 文件 | 大小 | 格式 | SHA256 |
|---|---|---|---|
| `seed101.pt` | 97,871,979 B (93.34 MB) | PyTorch | `48bf567cba8b761632b3d59ba7c4201ca54e58dfdfe8aca54c6034e8ba8af365` |
| `seed202.pt` | 97,871,979 B (93.34 MB) | PyTorch | `23d54c6c9b606fd530669f6109dab9b77567c765673ae4a15da56e1f630b1886` |
| `seed303.pt` | 97,871,979 B (93.34 MB) | PyTorch | `6b83fe6d26b998d127e288eac3a0edd966a8c8078fa62c84425af0e82ecd78d9` |
| `seed404.pt` | 97,871,979 B (93.34 MB) | PyTorch | `6ccf5fff5dbe6636b14885cb698819db91d46dce802022688d2e14826b2d831f` |
| `seed505.pt` | 97,871,979 B (93.34 MB) | PyTorch | `2e50510239f6f65a0ff35027ffd956024915ee786b26fe672a90311788ea571a` |
| `seed101.onnx` | 97,878,144 B (93.34 MB) | ONNX | `6b33a6e225354b981a3875ba2df0205eb3e3e23a0cbd992673d5cf9f2b533f63` |
| `seed202.onnx` | 97,878,144 B (93.34 MB) | ONNX | `30e790fce8b7f2043301fbf7c237574054e2a5c6c33b9914292e21cad6b05dda` |
| `seed303.onnx` | 97,878,144 B (93.34 MB) | ONNX | `9a979672a83a85f729e6a913cfa443b343d3f2f2c3afce74164f838b18b12bf7` |
| `seed404.onnx` | 97,878,144 B (93.34 MB) | ONNX | `4cc853029afc27f6b9c08204f1d3f39928caa9787f0c0414c950a2ace4d64348` |
| `seed505.onnx` | 97,878,144 B (93.34 MB) | ONNX | `f48e97b3647a9383ec8c4c1fb57941a0deed4ce10bc0bda1e9256f235b035e7a` |

合计 **10 个文件，933.6 MB**。

- `.pt`：PyTorch `state_dict`（纯权重字典，非整模型），用 `model.load_state_dict(torch.load(...))` 加载
- `.onnx`：单文件、权重内嵌（无 `.onnx.data` 附属文件），opset 18

全部权重的参数量均为 **24,429,281**（ResNet34-UNet）。训练数据为 **1177 张**全量标注图，对应内部 confirm（193 张，micro）Crack IoU **65.5053**。

---

## 2. 完整性校验

下载后建议核对 SHA256，任一不符请重新下载：

```bash
# Windows PowerShell
Get-FileHash seed101.pt -Algorithm SHA256

# Linux / macOS
sha256sum seed101.pt
```

一次性校验全部（把上面的表存成 `SHA256SUMS`）：

```bash
sha256sum -c SHA256SUMS
```

---

## 3. PyTorch 用法

```python
import torch
from model import ResNet34UNet      # 来自本仓库 code/model.py

m = ResNet34UNet(3, False)          # in_ch=3, use_attention=False
m.load_state_dict(torch.load("seed101.pt", map_location="cpu"))
m.eval()
```

**5 模型集成的完整推理**（与提交包一致）：

```bash
cd code
python make_submission.py \
  --ckpts ../ckpt/seed101.pt ../ckpt/seed202.pt ../ckpt/seed303.pt \
          ../ckpt/seed404.pt ../ckpt/seed505.pt \
  --val /path/to/UAV-CrackX/leftImg8bit/val \
  --out ../result.zip \
  --tau 0.2 --min-area 50 --thin-lambda 0.05
```

---

## 4. ONNX 用法

输入必须**先归一化并补齐到 384 行**（模型内部上采样到输入尺寸，378 会因 5 次下采样不整除而报错），推理后裁回 378：

```python
import numpy as np, cv2, onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

sess = ort.InferenceSession("seed101.onnx", providers=["CPUExecutionProvider"])

bgr = cv2.imread("xxx.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]                                  # 378, 672

x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD      # HWC
x = np.transpose(x, (2, 0, 1))[None]                  # 1CHW
x = np.pad(x, ((0,0),(0,0),(0,384-h),(0,0)), mode="reflect")   # 378 -> 384

logits = sess.run(None, {"input": x})[0][:, :, :h]     # 裁回 378
prob = 1.0 / (1.0 + np.exp(-logits))[0, 0]
```

5 个 ONNX 的输出**取算术平均**后再做后处理，等价于 PyTorch 版集成。

---

## 5. 精度验证

导出后已逐一对拍验证（`python code/check_onnx.py --ckpt-dir ../ckpt --onnx-dir ../onnx`）：

| seed | 参数量 | ONNX vs PyTorch 最大绝对误差 | 结果 |
|---|---|---|---|
| 101 | 24,429,281 | 3.05e-05 | OK |
| 202 | 24,429,281 | 2.19e-05 | OK |
| 303 | 24,429,281 | 7.44e-05 | OK |
| 404 | 24,429,281 | 2.10e-05 | OK |
| 505 | 24,429,281 | 5.34e-05 | OK |

误差量级约 1e-5，对二值化结果无影响。

复现导出：

```bash
cd code
python export_onnx.py --ckpt ../ckpt/seed101.pt --out ../onnx/seed101.onnx
```

---

## 6. 注意事项

1. **两组权重不可混用**。本目录是 **1177 张全量训练**版（提交用）。另有一组 831 张训练版用于内部评测复现 65.37，未在此提供。
2. **后处理顺序**：先连通域面积过滤（a≥50），再门控细化（λ=0.05）。`make_submission.py` 已按正确顺序实现，勿自行调换。
3. **`gated_thin` 必须限制在原掩膜内**（`& mask`），否则面积过滤会被阈值判定抵消。
4. 数据集 UAV-CrackX 遵循 CC BY-NC-SA 3.0，需向 VLP 平台申请，**不在本仓库或权重包内**。

---

## 下载地址

> 待填写。要求：huggingface / Google 云盘 / 百度网盘，链接有效期 ≥ 90 天。

| 平台 | 链接 | 提取码 | 有效期 |
|---|---|---|---|
| （待填写） | | | |
