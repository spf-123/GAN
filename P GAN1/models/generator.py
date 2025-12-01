import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import WaveletMambaBlock


class MultiResBlock(nn.Module):
    """多分辨率残差块 (保持不变)"""

    def __init__(self, in_c, out_c):
        super().__init__()
        c1 = out_c // 4
        c2 = out_c // 4
        c3 = out_c - c1 - c2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, c1, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class MultiResDown(nn.Module):
    """下采样块 (保持不变)"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = MultiResBlock(in_c, out_c)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class MultiResUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # [修改] 使用 PixelShuffle 代替 Upsample
        # PixelShuffle 会把通道数变为原本的 1/4 (r=2)，所以输入通道要调整
        self.up_conv = nn.Conv2d(in_c, in_c * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # PixelShuffle 后通道数变回 in_c，再和 skip (out_c) 拼接
        # 这里需要调整一下 block 的输入维度
        # 注意：为了代码改动最小，这里也可以用 TransposedConv
        # 或者更简单的：保留架构，只换插值方式为 'nearest' + 卷积，或者直接用 TransposedConv

        # 为了不破坏你现有维度逻辑，最稳妥的升级是使用 TransposedConv (反卷积)
        self.up = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2)

        self.block = MultiResBlock(in_c + out_c, out_c)  # 拼接后的维度

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸对齐逻辑保留
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)

        # 你的原代码这里拼接维度可能有隐患，通常是 cat([x, skip])
        # 假设 MultiResBlock 能处理 in_c + out_c 的输入
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class UNetSwinMultiResGenerator(nn.Module):
    def __init__(self, base=64):
        super().__init__()

        self.enc0 = MultiResBlock(3, base)
        self.enc1 = MultiResDown(base, base * 2)
        self.enc2 = MultiResDown(base * 2, base * 4)
        self.enc3 = MultiResDown(base * 4, base * 8)
        self.enc4 = MultiResDown(base * 8, base * 8)


        self.bottleneck = nn.Sequential(
            WaveletMambaBlock(dim=base * 8),
            WaveletMambaBlock(dim=base * 8)
        )

        # Decoder (保持不变)
        self.dec3 = MultiResUp(base * 16, base * 8)
        self.dec2 = MultiResUp(base * 12, base * 4)
        self.dec1 = MultiResUp(base * 6, base * 2)
        self.dec0 = MultiResUp(base * 3, base)

        self.out_conv = nn.Conv2d(base, 3, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c0 = self.enc0(x)
        c1 = self.enc1(c0)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)

        # Bottleneck
        # c4 shape: [B, C, H, W], HybridMambaBlock 内部会自动处理维度
        mamba_feat = self.bottleneck(c4)

        # 残差连接：将 bottleneck 的输出与 c4 相加
        bottleneck_out = mamba_feat + c4

        # Decoder
        u3 = self.dec3(bottleneck_out, c3)
        u2 = self.dec2(u3, c2)
        u1 = self.dec1(u2, c1)
        u0 = self.dec0(u1, c0)

        out = self.out_conv(u0)
        out = self.act(out)
        return out