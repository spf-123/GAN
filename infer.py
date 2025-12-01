import os, glob
import torch
from PIL import Image
import torchvision.transforms as T
from models.generator import UNetSwinMultiResGenerator
from utils.images import to_range01
import argparse


@torch.no_grad()
def load_model(ckpt='checkpoints/best.pt', device='cuda'):
    G = UNetSwinMultiResGenerator(base=64).to(device)
    state = torch.load(ckpt, map_location=device)
    G.load_state_dict(state['G'])
    G.eval()
    torch.no_grad()
    return G


@torch.no_grad()
def dehaze_folder(input_dir, out_dir, size=256, ckpt='checkpoints/best.pt', device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    G = load_model(ckpt, device)
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    toPIL = T.ToPILImage()

    paths = sorted(glob.glob(os.path.join(input_dir, '*')))
    print(f"Processing {len(paths)} images...")

    for i, p in enumerate(paths):
        if not os.path.isfile(p):
            continue
        try:
            img = Image.open(p).convert('RGB')
            x = tf(img).unsqueeze(0).to(device)
            y = G(x)[0].cpu()
            out_img = toPIL(to_range01(y).clamp(0, 1))
            out_path = os.path.join(out_dir, os.path.basename(p))
            out_img.save(out_path)
            print(f"  [{i + 1}/{len(paths)}] {os.path.basename(p)} -> {out_path}")
        except Exception as e:
            print(f"  ❌ Error processing {p}: {e}")


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--inp', required=True, help='Input directory with hazy images')
    a.add_argument('--out', required=True, help='Output directory for dehazed images')
    a.add_argument('--size', type=int, default=256, help='Image resize size')
    a.add_argument('--ckpt', default='checkpoints/best.pt', help='Checkpoint path')
    a.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = a.parse_args()
    dehaze_folder(args.inp, args.out, args.size, args.ckpt, args.device)