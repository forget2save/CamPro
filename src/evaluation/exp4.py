import sys
import os.path as osp

root = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
)
sys.path.append(root)

import pandas as pd
import torch
import torch.nn.functional as F
from rich.progress import track
from IQA_pytorch import SSIM, MS_SSIM

from isp.pipeline import Pipeline
from unet import UNet
from unet.data import coco_val_mask_loader
from misc import AverageMeter


@torch.no_grad()
def main():
    camera = Pipeline.load("../checkpoints/default.pt")

    ckpt = "../checkpoints/UNet.pt"
    enhancer = UNet()
    enhancer.load_state_dict(torch.load(ckpt))
    enhancer.eval().cuda()

    loader = coco_val_mask_loader(batch_size=1, num_workers=0)[1]

    ssim = SSIM().cuda()
    msssim = MS_SSIM().cuda()

    m_captured = [
        AverageMeter("RMSE"),
        AverageMeter("PSNR"),
        AverageMeter("SSIM"),
        AverageMeter("MS-SSIM"),
    ]

    m_enhanced = [
        AverageMeter("RMSE"),
        AverageMeter("PSNR"),
        AverageMeter("SSIM"),
        AverageMeter("MS-SSIM"),
    ]

    table = []
    for img_ori, mask in track(loader):
        img_ori = img_ori.cuda()
        mask = mask.cuda()
        img_cap = camera.forward(img_ori)
        img_enh = enhancer(img_cap).clamp(0, 1)

        score = F.mse_loss(img_ori, img_cap)
        m_captured[0].update(score.item() ** 0.5)
        score = -10 * torch.log10(F.mse_loss(img_ori, img_cap))
        m_captured[1].update(score.item())
        score = ssim.forward(img_ori, img_cap, as_loss=False)
        m_captured[2].update(score.item())
        score = msssim.forward(img_ori, img_cap, as_loss=False)
        m_captured[3].update(score.item())

        img_ori = img_ori * mask
        score = F.mse_loss(img_ori, img_enh)
        m_enhanced[0].update(score.item() ** 0.5)
        score = -10 * torch.log10(F.mse_loss(img_ori, img_enh))
        m_enhanced[1].update(score.item())
        score = ssim.forward(img_ori, img_enh, as_loss=False)
        m_enhanced[2].update(score.item())
        score = msssim.forward(img_ori, img_enh, as_loss=False)
        m_enhanced[3].update(score.item())

    s1 = [m.avg for m in m_captured]
    s1 = list(map(lambda o: f"{o:.6f}", s1))
    s2 = [m.avg for m in m_enhanced]
    s2 = list(map(lambda o: f"{o:.6f}", s2))

    table.append(["Captured"] + s1)
    table.append(["Enhanced"] + s2)

    df = pd.DataFrame(table, columns=["Image Type"] + [m.name for m in m_captured])
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv("../results/4.csv", index=False)


if __name__ == "__main__":
    main()
