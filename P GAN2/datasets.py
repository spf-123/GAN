import os
import re
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

_SUFFIX_PAT = re.compile(r"(_hazy|_Hazy|_HAZY|_gt|_GT|_clean|_CLEAR|_clear|_GT\d*|_h\d*|_\d*\.\d*|_\d{1,3}x\d{1,3})$",
                         re.IGNORECASE)


def _key(stem: str):
    """将文件名去除后缀，生成用于匹配的主键"""
    stem = os.path.splitext(stem)[0]
    if '_' in stem:
        stem = stem.split('_')[0]
    return stem


def rgb_to_ycbcr(img_tensor):
    """RGB [0,1] -> YCbCr"""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    B, C, H, W = img_tensor.shape
    r, g, b = img_tensor[:, 0], img_tensor[:, 1], img_tensor[:, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.419 * g - 0.081 * b + 0.5

    ycbcr = torch.stack([y, cb, cr], dim=1)
    return ycbcr


def ycbcr_to_rgb(img_tensor):
    """YCbCr -> RGB [0,1]"""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    B, C, H, W = img_tensor.shape
    y, cb, cr = img_tensor[:, 0], img_tensor[:, 1], img_tensor[:, 2]

    r = y + 1.402 * (cr - 0.5)
    g = y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)

    rgb = torch.stack([r, g, b], dim=1)
    return rgb.clamp(0, 1)


class DehazePairs(Dataset):
    def __init__(self, hazy_dir: str, clear_dir: str, size: int = 256,
                 augment: bool = True, augment_prob: float = 0.5,
                 use_ycbcr: bool = False):
        """
        构建数据集
        """
        assert os.path.isdir(hazy_dir), f"hazy_dir not found: {hazy_dir}"
        assert os.path.isdir(clear_dir), f"clear_dir not found: {clear_dir}"

        self.hazy_paths = sorted([p for p in glob.glob(os.path.join(hazy_dir, '*')) if os.path.isfile(p)])
        self.clear_paths = sorted([p for p in glob.glob(os.path.join(clear_dir, '*')) if os.path.isfile(p)])

        assert len(self.hazy_paths) and len(self.clear_paths), \
            f"Empty dirs.  hazy={hazy_dir}({len(self.hazy_paths)}), clear={clear_dir}({len(self.clear_paths)})"

        hmap = {_key(os.path.basename(p)): p for p in self.hazy_paths}
        cmap = {_key(os.path.basename(p)): p for p in self.clear_paths}

        keys = sorted(set(hmap.keys()) & set(cmap.keys()))

        assert len(keys) > 0, (
            "No matched pairs. Check filename keys.\n"
            f"Examples hazy: {os.path.basename(self.hazy_paths[0])}\n"
            f"Examples clear: {os.path.basename(self.clear_paths[0])}"
        )

        self.pairs = [(hmap[k], cmap[k]) for k in keys]
        self.use_ycbcr = use_ycbcr
        self.size = size
        self.augment = augment
        self.augment_prob = augment_prob

        # 仅用于最后的 Tensor 转换
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        hazy_path, clear_path = self.pairs[idx]

        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        # [优化点] 1. 确保图片足够大，如果小于 size 就放大
        w, h = hazy_img.size
        if w < self.size or h < self.size:
            # 保持比例放大到最小边等于 self.size
            scale = max(self.size / w, self.size / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            hazy_img = hazy_img.resize((new_w, new_h), Image.BICUBIC)
            clear_img = clear_img.resize((new_w, new_h), Image.BICUBIC)

        # [优化点] 2. 使用 RandomCrop 替代 Resize
        # 注意：这里需要手动写 crop 逻辑以保证 hazy 和 clear 裁剪位置一致
        i, j, h, w = T.RandomCrop.get_params(hazy_img, output_size=(self.size, self.size))
        hazy_img = TF.crop(hazy_img, i, j, h, w)
        clear_img = TF.crop(clear_img, i, j, h, w)

        # [优化点] 3. 数据增强 (翻转/旋转)
        if self.augment and random.random() < self.augment_prob:
            if random.random() < 0.5:
                hazy_img = TF.hflip(hazy_img)
                clear_img = TF.hflip(clear_img)
            if random.random() < 0.5:
                hazy_img = TF.vflip(hazy_img)
                clear_img = TF.vflip(clear_img)
            # 旋转建议去掉或只用90度旋转，任意角度旋转会引入黑色填充，影响去雾

        hazy_tensor = self.to_tensor(hazy_img)
        clear_tensor = self.to_tensor(clear_img)

        return hazy_tensor, clear_tensor  # (后面 YCbCr 逻辑保留)


def build_loader(root: str, split: str = 'train', size: int = 128,
                 batch_size: int = 8, shuffle: bool = True,
                 num_workers: int = 4, augment: bool = True,
                 use_ycbcr: bool = False):
    """
    创建数据加载器
    """
    hazy_dir = os.path.join(root, split, 'hazy')
    clear_dir = os.path.join(root, split, 'clear')

    # 只在训练集开启增强
    use_augment = augment and (split == 'train')

    ds = DehazePairs(hazy_dir, clear_dir, size=size, augment=use_augment, use_ycbcr=use_ycbcr)

    # 打印数据集信息
    if split == 'train':
        print(f"[datasets] split={split:5s} pairs={len(ds):4d} size={size} aug={use_augment}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if split == 'train' else 0,
        pin_memory=True,
        drop_last=(split == 'train'),
    )