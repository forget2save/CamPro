import os.path as osp
import sys
import torch
from torch import optim
from tqdm import tqdm
import torchvision as tv

yolov5_root = osp.join(osp.dirname(osp.abspath(__file__)), "..", "yolov5")
if yolov5_root not in sys.path:
    sys.path.append(yolov5_root)

from utils.general import non_max_suppression

from .data import coco_person_dataloader
from .loss import ComputeLoss
from .model import pretrained_yolov5
from .metric import cal_ap, parse_stats

sys.path.remove(yolov5_root)

trainloader, testloader = None, None
detector, optimizer, loss_fn = None, None, None

__all__ = [
    "utility_init",
    "utility_sample",
    "utility_forward",
    "utility_backward",
    "utility_validate",
]


def utility_init(cfg):
    global trainloader, testloader
    global detector, optimizer, loss_fn

    trainloader, testloader = coco_person_dataloader(cfg.UTILITY.BATCH, cfg.WORKERS)
    detector = pretrained_yolov5().cuda()
    optimizer = optim.SGD(
        detector.parameters(),
        lr=cfg.UTILITY.LR,
        weight_decay=5e-4,
    )
    loss_fn = ComputeLoss(detector)


def utility_sample(loader):
    x, y, _, _ = next(loader)
    x, y = x.cuda().float().div(255), y.cuda()
    return x, y


def utility_forward(x, y=None):
    x = detector(x)
    if y is None:
        x = non_max_suppression(x[0])
    else:
        x, _ = loss_fn(x, y)
    return x


def utility_backward():
    optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def utility_validate(transform=lambda x: x, fast=1):
    detector.eval()
    stats = []
    n = len(testloader)
    n = n // fast
    loader = iter(testloader)
    for i in tqdm(range(n), ncols=60, leave=False):
        img, lbl, _, shapes = next(loader)
        img, lbl = img.cuda().float().div(255), lbl.cuda()
        img = transform(img)
        out, _ = detector(img)
        out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.6)
        parse_stats(img, lbl, shapes, out, stats)
    ap50, ap = cal_ap(stats)
    return ap50, ap
