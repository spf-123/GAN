import os
import csv
import math
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

from datasets import build_loader
from models.generator import UNetSwinMultiResGenerator
from models.discriminator import DualDiscriminator
from utils.losses import (
    CharbonnierLoss,
    GANLoss,
    VGGPerceptualLoss,
    FFTFrequencyLoss,
)
from utils.images import save_samples


def to_numpy01(t):
    """BCHW [0,1] -> HWC [0,1]"""
    x = t.detach().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return x


def compute_batch_metrics(pred, gt):
    """pred/gt: BCHW [0,1]"""
    pred_np = to_numpy01(pred)
    gt_np = to_numpy01(gt)
    B = pred_np.shape[0]
    psnr_list, ssim_list = [], []
    for i in range(B):
        p = pred_np[i]
        g = gt_np[i]
        try:
            ssim = ssim_fn(g, p, data_range=1.0, channel_axis=-1)
        except TypeError:
            ssim = ssim_fn(g, p, data_range=1.0, multichannel=True)
        ps = psnr_fn(g, p, data_range=1.0)
        psnr_list.append(ps)
        ssim_list.append(ssim)
    return psnr_list, ssim_list


def fft_mag_gray(x):
    """计算灰度图的 FFT 幅度，用于频域判别器输入"""
    x_gray = x.mean(dim=1, keepdim=True)
    X = torch.fft.fft2(x_gray)
    X = torch.fft.fftshift(X)
    mag = torch.abs(X)
    mag = torch.log1p(mag)
    # 归一化到 [0, 1] 附近，防止数值过大
    max_val = mag.amax(dim=(2, 3), keepdim=True) + 1e-8
    mag = mag / max_val
    return mag


@torch.no_grad()
def validate(G, val_loader, device, save_preview=True, epoch=0):
    """验证阶段"""
    G.eval()
    l1 = nn.L1Loss(reduction='none')
    all_l1, all_psnr, all_ssim = [], [], []
    first_batch = True

    # 使用 ncols 固定进度条宽度，防止刷屏
    with tqdm(val_loader, desc='validate', leave=False, ncols=100) as pbar:
        for batch in pbar:
            if isinstance(batch, dict):
                hazy = batch['hazy_rgb'].to(device)
                clear = batch['clear_rgb'].to(device)
            else:
                hazy, clear = batch[0].to(device), batch[1].to(device)

            fake = G(hazy).clamp(0, 1)

            b_l1 = l1(fake, clear).mean(dim=(1, 2, 3)).cpu().tolist()
            all_l1 += b_l1

            b_psnr, b_ssim = compute_batch_metrics(fake, clear)
            all_psnr += b_psnr
            all_ssim += b_ssim

            if save_preview and first_batch:
                os.makedirs('samples', exist_ok=True)
                n = min(8, hazy.size(0))
                # 拼接对比图: Hazy | Fake | Clear
                triplet = torch.cat(
                    [hazy[:n].clamp(0, 1), fake[:n].clamp(0, 1), clear[:n].clamp(0, 1)],
                    dim=3
                )
                save_samples(triplet, f'samples/epoch{epoch:03d}_triplet.png')
                first_batch = False

    mean_l1 = float(sum(all_l1) / max(1, len(all_l1)))
    mean_psnr = float(sum(all_psnr) / max(1, len(all_psnr)))
    mean_ssim = float(sum(all_ssim) / max(1, len(all_ssim)))
    return {'L1': mean_l1, 'PSNR': mean_psnr, 'SSIM': mean_ssim}


def train_one_epoch(
        G, D, loader, opt_g, opt_d,
        ganloss, l1_loss_fn, device,
        vgg_loss_fn, fft_loss_fn,
        lambda_l1, lambda_vgg, lambda_fft
):
    """单轮训练"""
    G.train()
    D.train()

    # 初始化日志字典
    log = {'G': 0.0, 'D': 0.0, 'L1': 0.0, 'VGG': 0.0, 'FFT': 0.0}

    with tqdm(loader, desc='train', leave=False, ncols=100) as pbar:
        for batch in pbar:
            # 1. 数据准备
            if isinstance(batch, dict):
                hazy = batch['hazy_rgb'].to(device)
                clear = batch['clear_rgb'].to(device)
            else:
                hazy, clear = batch[0].to(device), batch[1].to(device)

            # ------------------------------------------------------------------
            #  Train Discriminator (D)
            # ------------------------------------------------------------------
            D.zero_grad(set_to_none=True)

            # 生成假图 (不计算G的梯度)
            with torch.no_grad():
                fake_detach = G(hazy).clamp(0, 1)

            # 准备频域数据
            real_freq = fft_mag_gray(clear)
            fake_freq = fft_mag_gray(fake_detach)

            # D - Forward Image (图像域判别)
            pred_real_img = D.forward_img(clear)
            pred_fake_img = D.forward_img(fake_detach)

            if isinstance(pred_real_img, (list, tuple)):
                loss_d_img = 0.0
                for pr, pf in zip(pred_real_img, pred_fake_img):
                    loss_d_img += ganloss.d_loss(pr, pf)
                loss_d_img /= len(pred_real_img)
            else:
                loss_d_img = ganloss.d_loss(pred_real_img, pred_fake_img)

            # D - Forward Frequency (频域判别)
            pred_real_freq = D.forward_freq(real_freq)
            pred_fake_freq = D.forward_freq(fake_freq)

            if isinstance(pred_real_freq, (list, tuple)):
                loss_d_freq = 0.0
                for pr, pf in zip(pred_real_freq, pred_fake_freq):
                    loss_d_freq += ganloss.d_loss(pr, pf)
                loss_d_freq /= len(pred_real_freq)
            else:
                loss_d_freq = ganloss.d_loss(pred_real_freq, pred_fake_freq)

            # D 总损失
            loss_d = loss_d_img + loss_d_freq
            loss_d.backward()

            # 梯度裁剪与更新
            nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            opt_d.step()

            # ------------------------------------------------------------------
            #  Train Generator (G)
            # ------------------------------------------------------------------
            G.zero_grad(set_to_none=True)

            # 重新生成以追踪梯度
            fake = G(hazy).clamp(0, 1)

            # G - GAN Loss
            pred_fake_img_for_g = D.forward_img(fake)
            if isinstance(pred_fake_img_for_g, (list, tuple)):
                loss_g_adv_img = 0.0
                for pf in pred_fake_img_for_g:
                    loss_g_adv_img += ganloss.g_loss(pf)
                loss_g_adv_img /= len(pred_fake_img_for_g)
            else:
                loss_g_adv_img = ganloss.g_loss(pred_fake_img_for_g)

            fake_freq_for_g = fft_mag_gray(fake)
            pred_fake_freq_for_g = D.forward_freq(fake_freq_for_g)
            if isinstance(pred_fake_freq_for_g, (list, tuple)):
                loss_g_adv_freq = 0.0
                for pf in pred_fake_freq_for_g:
                    loss_g_adv_freq += ganloss.g_loss(pf)
                loss_g_adv_freq /= len(pred_fake_freq_for_g)
            else:
                loss_g_adv_freq = ganloss.g_loss(pred_fake_freq_for_g)

            loss_g_adv = loss_g_adv_img + loss_g_adv_freq

            # G - Reconstruction Losses
            loss_l1 = l1_loss_fn(fake, clear) * lambda_l1
            loss_vgg = vgg_loss_fn(fake, clear) * lambda_vgg
            loss_fft = fft_loss_fn(fake, clear) * lambda_fft

            # G 总损失
            loss_g = loss_g_adv + loss_l1 + loss_vgg + loss_fft

            loss_g.backward()

            # 梯度裁剪与更新
            nn.utils.clip_grad_norm_(G.parameters(), 10.0)
            opt_g.step()

            # Logging
            log['G'] += loss_g.item()
            log['D'] += loss_d.item()
            log['L1'] += loss_l1.item()
            log['VGG'] += loss_vgg.item()
            log['FFT'] += loss_fft.item()

            pbar.set_postfix({'G': f"{loss_g.item():.4f}", 'D': f"{loss_d.item():.4f}"})

    # 计算 Epoch 平均值
    n = max(len(loader), 1)
    for k in log:
        log[k] /= n
    return log


def append_csv(path, fieldnames, row):
    existed = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not existed:
            w.writeheader()
        w.writerow(row)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='./data')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=200)

    # 学习率配置
    p.add_argument('--lr_g', type=float, default=2e-4)
    p.add_argument('--lr_d', type=float, default=2e-4)
    # Cosine Annealing 不需要 step_size，但保留 gamma 参数兼容
    p.add_argument('--lr_step_size', type=int, default=30)
    p.add_argument('--lr_gamma', type=float, default=0.5)

    # 损失权重配置
    p.add_argument('--lambda_l1', type=float, default=10.0)
    p.add_argument('--lambda_vgg', type=float, default=0.1)
    p.add_argument('--lambda_fft', type=float, default=0.1)

    # [策略 A] 验证频率：默认每 5 个 Epoch 验证一次
    p.add_argument('--val_every', type=int, default=5, help='Run validation every N epochs')

    # [策略 C] 保存频率：默认每 10 个 Epoch 强制保存一个备份
    p.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')

    p.add_argument('--resume', default='', help='path to checkpoint to resume')
    p.add_argument('--pretrain_ckpt', default='', help='path to pretrain checkpoint')
    p.add_argument('--num_workers', type=int, default=8)

    args = p.parse_args()

    # 初始化目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_loader = build_loader(args.data_root, 'train',
                                size=args.size, batch_size=args.batch_size,
                                num_workers=args.num_workers)

    # 验证集加载 (注意: 1000张验证比较耗时)
    val_loader = build_loader(args.data_root, 'val',
                              size=args.size, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # 模型初始化
    print("Initializing models...")
    G = UNetSwinMultiResGenerator(base=96).to(device)
    D = DualDiscriminator(in_c=3, base=64, n_layers=3, num_D=2).to(device)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[Model] G: {g_params:.2f}M params, D: {d_params:.2f}M params")

    # 优化器
    opt_g = optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=1e-4)

    # [优化] 使用 CosineAnnealingLR 替代 StepLR，更适合大数据集训练
    scheduler_g = lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-7)
    scheduler_d = lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-7)

    # 加载权重
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=device)
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        if 'opt_g' in state and 'opt_d' in state and 'epoch' in state:
            opt_g.load_state_dict(state['opt_g'])
            opt_d.load_state_dict(state['opt_d'])
            start_epoch = int(state['epoch']) + 1
        print(f'Resumed from {args.resume}, epoch {start_epoch}')
    elif args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        state = torch.load(args.pretrain_ckpt, map_location=device)
        if 'G' in state:
            G.load_state_dict(state['G'])
        else:
            G.load_state_dict(state)
        print(f'Loaded pretrain checkpoint from {args.pretrain_ckpt}')

    # 损失函数初始化
    ganloss = GANLoss('lsgan').to(device)
    l1_loss_fn = CharbonnierLoss().to(device)
    vgg_loss_fn = VGGPerceptualLoss(device).to(device)
    fft_loss_fn = FFTFrequencyLoss(cutoff=0.25).to(device)

    best_psnr = -math.inf
    log_csv = 'logs/train_log.csv'

    print(f"Start training from epoch {start_epoch} (Val every {args.val_every}, Save every {args.save_every})...")

    for epoch in range(start_epoch, args.epochs + 1):
        # 1. Train
        tr_log = train_one_epoch(
            G, D, train_loader,
            opt_g, opt_d,
            ganloss, l1_loss_fn, device,
            vgg_loss_fn, fft_loss_fn,
            args.lambda_l1, args.lambda_vgg, args.lambda_fft
        )

        # 2. Validate (策略 A: 减少验证频率)
        # 只有在符合间隔或最后一个 epoch 时才跑验证
        do_validation = (epoch % args.val_every == 0) or (epoch == args.epochs)

        if do_validation:
            val_log = validate(G, val_loader, device, save_preview=True, epoch=epoch)
            val_msg = f"Val(PSNR={val_log['PSNR']:.2f} SSIM={val_log['SSIM']:.4f})"
        else:
            # 如果跳过验证，用 None 占位
            val_log = None
            val_msg = "(Val Skipped)"

        # 3. Print & Log
        cur_lr_g = opt_g.param_groups[0]['lr']
        cur_lr_d = opt_d.param_groups[0]['lr']

        log_message = (f"[Ep {epoch:3d}] "
                       f"Train(G={tr_log['G']:.3f} D={tr_log['D']:.3f}) | {val_msg}")
        tqdm.write(log_message)

        # 写入 CSV
        fieldnames = ['epoch', 'lr_g', 'lr_d',
                      'train_G', 'train_D', 'train_L1', 'train_VGG', 'train_FFT',
                      'val_PSNR', 'val_SSIM', 'val_L1']

        row = {
            'epoch': epoch,
            'lr_g': cur_lr_g,
            'lr_d': cur_lr_d,
            'train_G': tr_log['G'],
            'train_D': tr_log['D'],
            'train_L1': tr_log['L1'],
            'train_VGG': tr_log['VGG'],
            'train_FFT': tr_log['FFT'],
            'val_PSNR': val_log['PSNR'] if val_log else '',
            'val_SSIM': val_log['SSIM'] if val_log else '',
            'val_L1': val_log['L1'] if val_log else '',
        }
        append_csv(log_csv, fieldnames, row)

        # 4. Save Checkpoints
        ckpt = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'opt_g': opt_g.state_dict(),
            'opt_d': opt_d.state_dict(),
            'epoch': epoch,
        }

        # 始终保存最新的，防止中断
        torch.save(ckpt, f'checkpoints/latest.pt')

        # [策略 C] 按照 save_every 频率保存历史记录 (默认每10轮)
        if epoch % args.save_every == 0:
            torch.save(ckpt, f'checkpoints/epoch{epoch:03d}.pt')

        # 仅在跑了验证且结果更好的时候更新 best.pt
        if val_log and val_log['PSNR'] > best_psnr:
            best_psnr = val_log['PSNR']
            torch.save(ckpt, 'checkpoints/best.pt')
            tqdm.write(f"  >>> New Best PSNR: {best_psnr:.2f} <<<")

        scheduler_g.step()
        scheduler_d.step()

    print("Training completed!")