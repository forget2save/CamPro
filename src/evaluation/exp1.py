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
from torch import nn, optim
from datetime import datetime

from rich.progress import track
from isp.pipeline import Pipeline

from privacy.data import ClosedSetCelebA, ClosedSetLFW
from privacy.backbone import pretrained_backbone
from privacy.metric import CMC

from unet import UNet


def linear_classifier(gallery, probe, label):
    out_channels, in_channels = gallery.shape
    f = nn.Linear(in_channels, out_channels).cuda()
    g = nn.CrossEntropyLoss()
    opt = optim.SGD(f.parameters(), 5)
    best_acc = 0
    for _ in range(200):
        pred = f(gallery)
        with torch.no_grad():
            pred2 = f(probe)
        loss = g(pred, label)
        test_acc = (pred2.max(dim=1)[1] == label).float().mean().item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        best_acc = max(test_acc, best_acc)
    return best_acc


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
    camera = Pipeline.load("../checkpoints/default.pt")

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
    datasets = [
        ("CelebA", ClosedSetCelebA),
        ("LFW", ClosedSetLFW),
    ]
    transforms = [
        ("Raw", lambda o: o),
        ("Captured", camera.forward),
        ("Enhanced", lambda o: enhancer(camera.forward(o)).clamp(0, 1)),
    ]
    classifiers = [
        ("Nearest", lambda g, p, l: CMC(g, p, l).cmc_curve()[1][0]),
        ("Linear", lambda g, p, l: linear_classifier(g, p, l)),
    ]
    # Expected computation time: 30 minutes for 1 repeat time
    repeat = 10

    # Start evaluation
    table = []
    for d_name, d_class in datasets:
        for c_name, classifer in classifiers:
            for t_name, t_func in transforms:
                tic = datetime.now()
                row = [d_name, c_name, t_name]
                for m in models:
                    backbone = pretrained_backbone(m).cuda().eval()
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

    df = pd.DataFrame(table, columns=["Dataset", "Image Type", "Classifier"] + models)
    df["Average"] = df.iloc[:, 3 : 3 + len(models)].mean(axis=1)
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv("../results/1.csv", index=False)


if __name__ == "__main__":
    main()
