import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy import signal

class CharbonnierLoss(nn.Module):
    """L1 Loss 的鲁棒变体，SOTA 标配"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = sqrt(diff^2 + eps^2)
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        return torch.mean(loss)

class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        assert gan_mode in ['lsgan', 'hinge'], f"Unsupported gan_mode: {gan_mode}"
        self.gan_mode = gan_mode
        self.mse = nn.MSELoss()

    def d_loss(self, pred_real, pred_fake):
        if self.gan_mode == 'lsgan':
            return (self.mse(pred_real, torch.ones_like(pred_real)) +
                    self.mse(pred_fake, torch.zeros_like(pred_fake))) * 0.5
        else:
            return (torch.relu(1 - pred_real).mean() +
                    torch.relu(1 + pred_fake).mean())

    def g_loss(self, pred_fake):
        if self.gan_mode == 'lsgan':
            return self.mse(pred_fake, torch.ones_like(pred_fake))
        else:
            return -pred_fake.mean()


class VGGPerceptualLoss(nn.Module):
    """基于 VGG19 的感知损失"""

    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self._norm(x)
        y = self._norm(y)
        fx = self.vgg(x)
        fy = self.vgg(y)
        return F.mse_loss(fx, fy)


class SobelLoss(nn.Module):
    """Sobel 边缘损失（带soft mask）"""

    def __init__(self, reduction='mean'):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('kernel_x', kx.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', ky.view(1, 1, 3, 3))
        self.reduction = reduction

    def forward(self, x):
        B, C, H, W = x.shape
        kx = self.kernel_x.to(x.device)
        ky = self.kernel_y.to(x.device)

        kx = kx.expand(C, 1, 3, 3)
        ky = ky.expand(C, 1, 3, 3)

        grad_x = F.conv2d(x, kx, padding=1, groups=C)
        grad_y = F.conv2d(x, ky, padding=1, groups=C)

        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        mask = torch.sigmoid(5 * (grad_mag - grad_mag.mean(dim=(2, 3), keepdim=True)))
        weighted_grad = grad_mag * mask

        if self.reduction == 'mean':
            return weighted_grad.mean()
        else:
            return weighted_grad.sum()


class WaveletFrequencyLoss(nn.Module):
    """小波频域损失 - 多尺度细节保留"""

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def forward(self, fake, clear):
        """计算多尺度小波损失"""
        loss = 0.0
        fake_gray = fake.mean(dim=1, keepdim=True)
        clear_gray = clear.mean(dim=1, keepdim=True)

        for scale in range(2):
            # 应用高斯滤波器作为Sobel变体
            fake_edges = self._compute_edges(fake_gray)
            clear_edges = self._compute_edges(clear_gray)

            loss += F.l1_loss(fake_edges, clear_edges)

            # 下采样到下一尺度
            fake_gray = F.avg_pool2d(fake_gray, 2)
            clear_gray = F.avg_pool2d(clear_gray, 2)

        return loss / 2.0

    def _compute_edges(self, x):
        """计算边缘特征"""
        # 使用Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)


class FFTFrequencyLoss(nn.Module):
    """FFT频域损失 - 全局频谱约束"""

    def __init__(self, cutoff=0.25):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, fake, clear):
        """计算FFT幅度损失"""
        fake_gray = fake.mean(dim=1, keepdim=True)
        clear_gray = clear.mean(dim=1, keepdim=True)

        # 计算FFT
        fake_fft = torch.fft.fft2(fake_gray)
        clear_fft = torch.fft.fft2(clear_gray)

        fake_mag = torch.abs(fake_fft)
        clear_mag = torch.abs(clear_fft)

        # 对数域上的L1损失
        fake_mag_log = torch.log1p(fake_mag)
        clear_mag_log = torch.log1p(clear_mag)

        loss = F.l1_loss(fake_mag_log, clear_mag_log)

        # 高频损失（细节保留）
        hf_loss = self._high_frequency_loss(fake_fft, clear_fft)

        return loss + 0.5 * hf_loss

    def _high_frequency_loss(self, fake_fft, clear_fft):
        """高频分量损失"""
        B, C, H, W = fake_fft.shape

        # 创建高频掩码
        mask = torch.ones((1, 1, H, W), device=fake_fft.device)
        cy, cx = H // 2, W // 2
        r = int(min(H, W) * self.cutoff * 0.5)

        if r > 0:
            y1, y2 = max(cy - r, 0), min(cy + r, H)
            x1, x2 = max(cx - r, 0), min(cx + r, W)
            mask[:, :, y1:y2, x1:x2] = 0.0

        fake_hf = fake_fft * mask
        clear_hf = clear_fft * mask

        loss = F.l1_loss(torch.abs(fake_hf), torch.abs(clear_hf))
        return loss

class GuidedLoss(nn.Module):
    """引导式损失：基于图像引导的细节增强"""

    def __init__(self, guide_weight=0.5):
        super().__init__()
        self.guide_weight = guide_weight

    def forward(self, fake, clear):
        """使用梯度引导的损失"""
        # 计算清晰图的梯度作为引导
        clear_grad_x = torch.abs(torch.diff(clear, dim=2))
        clear_grad_y = torch.abs(torch.diff(clear, dim=3))

        # 补齐维度
        clear_grad_x = F.pad(clear_grad_x, (0, 0, 0, 1), mode='constant', value=0)
        clear_grad_y = F.pad(clear_grad_y, (0, 1, 0, 0), mode='constant', value=0)

        guide = torch.sqrt(clear_grad_x ** 2 + clear_grad_y ** 2 + 1e-8)

        # 在高梯度区域加权
        loss = torch.abs(fake - clear)
        weighted_loss = loss * (1.0 + 2.0 * guide)

        return weighted_loss.mean() * self.guide_weight


# ========== 简化版本 ConsistencyLoss ==========
class ConsistencyLoss(nn.Module):
    """自监督一致性损失 - 用于预训练"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, dehazed, re_dehazed):
        """
        重建一致性：去雾图再去雾应接近自身
        """
        return self.l1(dehazed, re_dehazed)  # 直接返回 Tensor


class PatchSimilarityLoss(nn.Module):
    """Patch自相似性损失 - 改进版"""

    def __init__(self, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape

        if H < self.patch_size or W < self.patch_size:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)

        # 提取patches
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)

        if patches.numel() == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)

        num_patches = patches.shape[-1]
        patches = patches.reshape(B, C * self.patch_size * self.patch_size, num_patches)
        patches = patches.permute(0, 2, 1)

        # 计算余弦相似度矩阵
        patches_norm = F.normalize(patches, p=2, dim=2)
        sim_matrix = torch.bmm(patches_norm, patches_norm.transpose(1, 2))

        mask = torch.triu(torch.ones(sim_matrix.shape[1], sim_matrix.shape[2],
                                     device=x.device), diagonal=1)
        sim_matrix = sim_matrix * mask.unsqueeze(0)

        # 策略1：惩罚不相似性（推荐用于去雾）
        target_sim = 0.8  # 相同内容的patches应该相似
        loss = F.mse_loss(sim_matrix[sim_matrix > 0],
                          torch.ones_like(sim_matrix[sim_matrix > 0]) * target_sim)

        return loss

class GradientPenaltyLoss(nn.Module):
    """梯度惩罚（WGAN-GP）"""

    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, D, real, fake):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interpolates = D(interpolates)
        fake_output = torch.ones(d_interpolates.size(), device=real.device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty