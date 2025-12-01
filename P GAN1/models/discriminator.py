import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(feat))
        return attn


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        ca = self.ca(x)
        x = x * ca
        sa = self.sa(x)
        x = x * sa
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.act(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_down = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res = ResBlock(out_c)
        self.cbam = CBAMBlock(out_c)

    def forward(self, x):
        x = self.act(self.bn(self.conv_down(x)))
        x = self.res(x)
        x = self.cbam(x)
        return x


class SwinLikeBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.scale = (dim // num_heads) ** -0.5

        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W


        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_perm.view(B, N, C)


        shortcut = x_flat
        x_norm = self.norm1(x_flat)
        qkv = self.qkv(x_norm).view(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        x_flat = out + shortcut


        shortcut2 = x_flat
        x_norm2 = self.norm2(x_flat)
        x_mlp = self.mlp(x_norm2)
        x_flat = x_mlp + shortcut2

        x_out = x_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_c=3, base=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1

        self.first = nn.Sequential(
            nn.Conv2d(in_c, base, kw, 2, padw),
            nn.LeakyReLU(0.2, True)
        )

        nf = base
        blocks = []
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(512, nf * 2)
            blocks.append(DownBlock(nf_prev, nf))

        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Conv2d(nf, 1, kw, 1, padw)

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_c=3, base=64, n_layers=3, num_D=2, use_swin=True):
        super().__init__()
        assert num_D >= 1
        self.num_D = num_D
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(in_c=in_c, base=base, n_layers=n_layers)
            for _ in range(num_D)
        ])

    def forward(self, x):
        results = []
        inp = x
        for D in self.discriminators:
            results.append(D(inp))
            inp = self.downsample(inp)
        return results


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_c=1, base=64, n_layers=3, use_swin=True):
        super().__init__()
        kw = 4
        padw = 1

        self.first = nn.Sequential(
            nn.Conv2d(in_c, base, kw, 2, padw),
            nn.LeakyReLU(0.2, True)
        )

        nf = base
        blocks = []
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(512, nf * 2)
            blocks.append(DownBlock(nf_prev, nf))

        self.blocks = nn.Sequential(*blocks)
        self.use_swin = use_swin
        if use_swin:
            self.swin = SwinLikeBlock(dim=nf, num_heads=4, mlp_ratio=4.0)

        self.final_conv = nn.Conv2d(nf, 1, kw, 1, padw)

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        if self.use_swin:
            x = self.swin(x)
        x = self.final_conv(x)
        return x


class MultiScaleFrequencyDiscriminator(nn.Module):
    def __init__(self, in_c=1, base=64, n_layers=3, num_D=2, use_swin=True):
        super().__init__()
        assert num_D >= 1
        self.num_D = num_D
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )
        self.discriminators = nn.ModuleList([
            FrequencyDiscriminator(in_c=in_c, base=base, n_layers=n_layers, use_swin=use_swin)
            for _ in range(num_D)
        ])

    def forward(self, x):
        results = []
        inp = x
        for D in self.discriminators:
            results.append(D(inp))
            inp = self.downsample(inp)
        return results

class DualDiscriminator(nn.Module):
    def __init__(self, in_c=3, base=64, n_layers=3, num_D=2, use_swin=True):
        super().__init__()
        self.imgD = MultiScaleDiscriminator(
            in_c=in_c, base=base, n_layers=n_layers, num_D=num_D, use_swin=use_swin
        )
        self.freqD = MultiScaleFrequencyDiscriminator(
            in_c=1, base=base, n_layers=n_layers, num_D=num_D, use_swin=use_swin
        )

    def forward_img(self, x):
        return self.imgD(x)

    def forward_freq(self, x):
        return self.freqD(x)
