import sys
import os.path as osp

root = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
)
sys.path.append(root)

import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch import optim, nn
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
from rich.progress import track

from isp.pipeline import Pipeline
from privacy.data import celeba_dataloader, ClosedSetCelebA2
from privacy.backbone import pretrained_backbone
from privacy.head import SoftmaxHead, ArcFaceHead
from privacy.metric import CMC
from misc import AverageMeter, timer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=int, default=7)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    @timer
    def train_epoch(transform=lambda x: x):
        loss_fn = nn.CrossEntropyLoss()
        extractor.train()
        clshead.train()
        optimizer.zero_grad()
        meter = AverageMeter("CrossEntropyLoss")
        for x, y in track(trainloader, description="Training..."):
            x, y = x.cuda(), y.cuda()
            x = transform(x)
            x = normalize(x)
            x = extractor(x)
            x = clshead(x, y)
            loss = loss_fn(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            meter.update(loss.item())
        return meter.avg

    @timer
    @torch.no_grad()
    def val_epoch(transform=lambda x: x):
        extractor.eval()
        acc = 0
        repeat = 10
        for _ in range(repeat):
            valloader = DataLoader(
                ClosedSetCelebA2(deepcopy(testimgs)),
                batch_size=64,
                num_workers=8,
            )
            gallery, probe = [], []
            for a, b in track(valloader, description="Validating..."):
                a, b = a.cuda(), b.cuda()
                a = transform(a)
                b = transform(b)
                gallery.append(extractor(normalize(a)))
                probe.append(extractor(normalize(b)))
            gallery = torch.cat(gallery, dim=0)
            probe = torch.cat(probe, dim=0)
            label = torch.arange(gallery.shape[0], device=gallery.device)
            _, y = CMC(gallery, probe, label).cmc_curve()
            acc += y[0]
        acc = acc / 10
        return acc

    args = parse_args()

    camera = Pipeline.load("../checkpoints/default.pt")
    camera.ccm.ccm.requires_grad_(False)
    camera.gamma.keypoints.requires_grad_(False)

    trainloader, testloader = celeba_dataloader(batch_size=128, num_workers=8)
    d = testloader.dataset
    testimgs = [d.dataset.imgs[i][0] for i in d.indices]
    normalize = tv.transforms.Normalize([0.5] * 3, [0.5] * 3)

    backbones = [
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
    heads = [
        ("Softmax", SoftmaxHead, 1e-1),
        ("ArcFace", ArcFaceHead, 1e-1),
    ]
    modes = [
        ("Finetune", True),
        ("Train-From-Scratch", False),
    ]
    transform = camera.forward

    backbone = backbones[args.backbone]
    h_name, head, lr = heads[args.head]
    p_name, pretrained = modes[args.mode]

    extractor = pretrained_backbone(backbone, pretrained=pretrained).cuda()
    clshead = head(10177).cuda()

    optimizer = optim.SGD(
        [
            {"params": extractor.parameters()},
            {"params": clshead.parameters()},
        ],
        lr=lr,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.1)

    accs = []
    for e in range(1, 21):
        loss = train_epoch(transform)
        acc = val_epoch(transform)
        accs.append(acc)
        scheduler.step()
        print(e, loss, acc)
    print("Backbone:", backbone)
    print("Head:", h_name)
    print("Mode:", p_name)
    print("Highest Accuracy:", np.array(accs).max())


if __name__ == "__main__":
    main()
