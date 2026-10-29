"""ResNet-34 U-Net。B 加输入通道，C 在跳跃连接上放 Attention Gate。"""


from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """粗层特征决定跳跃连接里哪些位置留下。"""

    def __init__(self, skip_ch: int, gate_ch: int) -> None:
        super().__init__()
        inter = max(skip_ch // 2, 16)
        self.theta = nn.Conv2d(skip_ch, inter, 1, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter, 1, bias=False)
        self.psi = nn.Conv2d(inter, 1, 1, bias=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.theta(skip) + self.phi(gate))))
        return skip * alpha


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention: bool) -> None:
        super().__init__()
        self.gate = AttentionGate(skip_ch, in_ch) if attention else None
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.gate is not None:
            skip = self.gate(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet34UNet(nn.Module):
    def __init__(self, in_channels: int = 3, attention: bool = False) -> None:
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        if in_channels == 3:
            self.conv1 = backbone.conv1
        else:
            conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :3].copy_(backbone.conv1.weight)
            self.conv1 = conv
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up3 = UpBlock(512, 256, 256, attention)
        self.up2 = UpBlock(256, 128, 128, attention)
        self.up1 = UpBlock(128, 64, 64, attention)
        self.up0 = UpBlock(64, 64, 32, attention)
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = x.shape[-2]
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.up3(x4, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        y = F.interpolate(y, size=(height, x.shape[-1]), mode="bilinear", align_corners=False)
        return self.head(y)

    def dark_channel_norm(self) -> float:
        weight = self.conv1.weight[:, 3:]
        if weight.numel() == 0:
            return 0.0
        return float(weight.detach().norm().cpu())


class MultiEvidenceUNet(nn.Module):
    """主头 + 浅层侧输出 + 1x1 多证据融合打分（思路三）。

    证据：p_main, p_side, soft_skel(p_main), contrast(外部传入)。
    forward 返回最终 logits；train 时可取 .aux 拿侧输出。
    """

    def __init__(self, in_channels: int = 3, attention: bool = False, skel_iter: int = 5) -> None:
        super().__init__()
        self.backbone = ResNet34UNet(in_channels=in_channels, attention=attention)
        self.side_head = nn.Conv2d(64, 1, 1)
        nn.init.constant_(self.side_head.bias, -4.0)
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
        )
        nn.init.constant_(self.fuse[-1].bias, -2.0)
        self.skel_iter = int(skel_iter)
        self.last_side: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, contrast: torch.Tensor | None = None) -> torch.Tensor:
        from losses import soft_skeleton

        height = x.shape[-2]
        b = self.backbone
        x0 = b.relu(b.bn1(b.conv1(x)))
        x1 = b.layer1(b.maxpool(x0))
        x2 = b.layer2(x1)
        x3 = b.layer3(x2)
        x4 = b.layer4(x3)
        y = b.up3(x4, x3)
        y = b.up2(y, x2)
        y = b.up1(y, x1)
        y = b.up0(y, x0)
        y = F.interpolate(y, size=(height, x.shape[-1]), mode="bilinear", align_corners=False)
        logits_main = b.head(y)
        side = self.side_head(x0)
        side = F.interpolate(side, size=(height, x.shape[-1]), mode="bilinear", align_corners=False)
        self.last_side = side
        p_main = torch.sigmoid(logits_main)
        p_side = torch.sigmoid(side)
        skel = soft_skeleton(p_main, self.skel_iter)
        if contrast is None:
            contrast = torch.zeros_like(p_main)
        evidence = torch.cat([p_main, p_side, skel, contrast], dim=1)
        return self.fuse(evidence)
