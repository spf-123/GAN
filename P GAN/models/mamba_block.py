import torch
import torch.nn as nn

# [修改点 1] 尝试导入官方 Mamba，如果环境没有装，才回退到 PyTorch 版
try:
    from mamba_ssm import Mamba

    IS_OFFICIAL_MAMBA = True
    print("[Mamba] Using Official CUDA Implementation (Fast!)")
except ImportError:
    IS_OFFICIAL_MAMBA = False
    print("[Mamba] Using Pure PyTorch Implementation (Slower but Compatible)")
    # 这里保留 PurePyTorchSSM 类定义，防止报错，但实际上如果安装了库就不会用到它
    import math
    import torch.nn.functional as F


    class PurePyTorchSSM(nn.Module):
        def __init__(self, dim, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.dim = dim
            self.d_inner = int(expand * dim)
            self.dt_rank = math.ceil(dim / 16)
            self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                    bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
            self.activation = nn.SiLU()
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        def forward(self, x):
            B, L, C = x.shape
            xz = self.in_proj(x)
            x_proj, z = xz.chunk(2, dim=-1)
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.conv1d(x_proj)[:, :, :L]
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.activation(x_proj)
            ssm_params = self.x_proj(x_proj)
            dt, B_ssm, C_ssm = torch.split(ssm_params, [self.dt_rank, 16, 16], dim=-1)
            dt_scaling = torch.sigmoid(self.dt_proj(dt))
            global_context = x_proj.mean(dim=1, keepdim=True) * dt_scaling.mean(dim=1, keepdim=True)
            y = x_proj * dt_scaling + global_context
            out = y * self.activation(z)
            return self.out_proj(out)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    x_high = torch.cat([x_HL, x_LH, x_HH], dim=1)
    return x_LL, x_high


def idwt_init(x_LL, x_high):
    r = 2
    in_batch, in_channel, in_height, in_width = x_LL.size()
    out_channel, out_height, out_width = in_channel, r * in_height, r * in_width
    x_HL, x_LH, x_HH = torch.chunk(x_high, 3, dim=1)
    h_ll, h_hl, h_lh, h_hh = x_LL / 2, x_HL / 2, x_LH / 2, x_HH / 2
    y1 = h_ll - h_hl - h_lh + h_hh
    y2 = h_ll - h_hl + h_lh - h_hh
    y3 = h_ll + h_hl - h_lh - h_hh
    y4 = h_ll + h_hl + h_lh + h_hh
    y = torch.zeros((in_batch, out_channel, out_height, out_width), device=x_LL.device)
    y[:, :, 0::2, 0::2] = y1
    y[:, :, 1::2, 0::2] = y2
    y[:, :, 0::2, 1::2] = y3
    y[:, :, 1::2, 1::2] = y4
    return y


class WaveletMambaBlock(nn.Module):
    """
    [SCI 一区级创新] 小波-Mamba 频域解耦模块
    自动切换官方 CUDA Mamba 或 PyTorch 实现
    """

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()

        # [修改点 2] 根据环境自动选择 Mamba 实现
        if IS_OFFICIAL_MAMBA:
            # 官方 Mamba 接口: d_model=dim
            self.mamba_ll = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            # 纯 PyTorch 接口
            self.mamba_ll = PurePyTorchSSM(dim)

        self.norm_ll = nn.LayerNorm(dim)

        self.conv_high = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False),
            nn.BatchNorm2d(dim * 3),
            nn.SiLU(),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1, bias=False)
        )

        self.norm_fusion = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.fusion_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        # 1. DWT
        x_ll, x_high = dwt_init(x)

        # 2. Mamba Branch (Low Freq)
        # Reshape: [B, C, H/2, W/2] -> [B, L, C]
        x_ll_in = x_ll.permute(0, 2, 3, 1).contiguous()
        x_ll_in = self.norm_ll(x_ll_in)
        b_ll, h_ll, w_ll, c_ll = x_ll_in.shape
        x_ll_flat = x_ll_in.view(b_ll, -1, c_ll)

        # Mamba Forward
        x_ll_out = self.mamba_ll(x_ll_flat)

        # Reshape back
        x_ll_out = x_ll_out.view(b_ll, h_ll, w_ll, c_ll).permute(0, 3, 1, 2)
        x_ll_out = x_ll + x_ll_out

        # 3. CNN Branch (High Freq)
        x_high_out = self.conv_high(x_high)
        x_high_out = x_high + x_high_out

        # 4. IDWT & Fusion
        x_recon = idwt_init(x_ll_out, x_high_out)
        x_recon = self.fusion_conv(x_recon)
        x = shortcut + x_recon

        # FFN
        shortcut_2 = x
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm_fusion(x_norm)
        x_mlp = self.mlp(x_norm).permute(0, 3, 1, 2)

        return shortcut_2 + x_mlp