import sys
import os.path as osp

root = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
)
sys.path.append(root)

from datetime import datetime
import pandas as pd
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from datetime import datetime

from rich.progress import track
from isp.pipeline import Pipeline

from privacy.data import ClosedSetCelebA
from privacy.backbone import pretrained_backbone
from privacy.metric import CMC

from unet import UNet


@torch.no_grad()
def evaluate(d_class, t_func, backbone):
    normalize = tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
    loader = DataLoader(d_class(), batch_size=64, num_workers=8)
    gallery, probe = [], []
    for a, b in track(loader):
        a, b = a.cuda(), b.cuda()
        b = t_func(b)
        gallery.append(backbone(normalize(a)))
        probe.append(backbone(normalize(b)))
    gallery = torch.cat(gallery, dim=0)
    probe = torch.cat(probe, dim=0)
    label = torch.arange(gallery.shape[0], device=gallery.device)
    return gallery, probe, label


def main():
    # Load optimized ISP parameters
    camera = Pipeline.load("../checkpoints/ablation.pt")

    # Load trained image enhancer
    ckpt = "../checkpoints/UNet.pt"
    enhancer = UNet()
    enhancer.load_state_dict(torch.load(ckpt))
    enhancer.eval().cuda()

    # Experiment Configs
    # You can arbitrarily comment or uncomment some lines to enable or disable some experiments.
    models = [
        "facenet",
        "arcface_ir18",
        "arcface_irse50",
        "arcface_ir152",
        "magface_ir18",
        "magface_ir50",
        "magface_ir100",
        "adaface_ir18",
        "adaface_ir50",
        "adaface_ir100",
    ]
    d_class = ClosedSetCelebA
    t_func = camera.forward
    classifer = lambda g, p, l: CMC(g, p, l).cmc_curve()[1][0]
    # Expected computation time: 30 minutes for 1 repeat time
    repeat = 10

    # Start evaluation
    table = []
    tic = datetime.now()
    for m in models:
        row = [m]
        backbone = pretrained_backbone(m).cuda().eval()
        if m == "adaface_ir18":
            backbone.load_state_dict(torch.load("../checkpoints/whitebox.pt"))
            row.append(False)
        else:
            row.append(True)
        avg_acc = []
        for _ in range(repeat):
            gallery, probe, label = evaluate(d_class, t_func, backbone)
            acc = classifer(gallery, probe, label)
            avg_acc.append(acc)
        avg_acc = sum(avg_acc) / len(avg_acc)
        row.append(avg_acc)
        table.append(row)
    toc = datetime.now()
    print(toc - tic)

    df = pd.DataFrame(
        table,
        columns=["Name", "Black-box", "Protector-Only"],
    )
    df = df.reset_index(drop=True)
    df.sort_values(by=["Black-box", "Protector-Only"], inplace=True)

    avg = df.iloc[1 : len(models), 2].mean()
    df.loc[len(models)] = ["Average Black-box Accuracy", None, avg]
    ratio = avg / df.iloc[0, 2]
    df.loc[len(models) + 1] = ["Accuracy ratio (Black-box/White-box)", None, ratio]

    print(df.head())
    df.to_csv("../results/2.csv", index=False)


if __name__ == "__main__":
    main()
