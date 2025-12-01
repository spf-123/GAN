import torch
import torchvision.utils as vutils

def to_range01(x):
    return x

def save_samples(imgs, path, nrow=4):
    vutils.save_image(to_range01(imgs).clamp(0,1), path, nrow=nrow)