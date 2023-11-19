from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import torchvision as tv

from .data import celeba_dataloader, ClosedSetLFW
from .backbone import pretrained_backbone
from .head import head_type
from .metric import CMC
from .loss import loss_fn_adv, loss_fn_neg

trainloader, valloader, testloader = None, None, None
extractor, clshead, optimizer = None, None, None

# loss function
loss_fn = nn.CrossEntropyLoss()
normalize = tv.transforms.Normalize([0.5] * 3, [0.5] * 3)

__all__ = [
    "privacy_init",
    "privacy_sample",
    "privacy_forward",
    "privacy_backward",
    "privacy_validate",
    "privacy_pretrain",
]


def privacy_init(cfg):
    global trainloader, valloader, testloader
    global extractor, clshead, optimizer

    trainloader, testloader = celeba_dataloader(cfg.PRIVACY.BATCH, cfg.WORKERS)
    valloader = DataLoader(ClosedSetLFW(), batch_size=128, num_workers=cfg.WORKERS)
    extractor = pretrained_backbone(cfg.PRIVACY.BACKBONE).cuda()
    clshead = head_type(cfg.PRIVACY.HEAD)(10177).cuda()
    optimizer = optim.SGD(
        [
            {"params": extractor.parameters()},
            {"params": clshead.parameters()},
        ],
        lr=cfg.PRIVACY.LR,
        weight_decay=5e-4,
    )


# wrappers
def privacy_sample(loader):
    x, y = next(loader)
    x, y = x.cuda(), y.cuda()
    return x, y


def privacy_forward(x, y=None, loss="nor"):
    """`loss` in [`"nor"`, `"adv"`, `"neg"`, `"ent"`]"""
    x = normalize(x)
    x = extractor(x)
    if y is not None:
        x = clshead(x, y)
        if loss == "nor":
            x = loss_fn(x, y)
        elif loss == "adv":
            x = loss_fn_adv(x, y)
        elif loss == "neg":
            x = loss_fn_neg(x, y)
        elif loss == "acc":
            x = (x.max(dim=1)[1] == y).float().mean().item()
        else:
            raise ValueError()
    return x


def privacy_backward():
    optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def privacy_validate(transform=lambda x: x, model=None, mode="tt"):
    if model is None:
        model = extractor
    model.eval()

    gallery, probe = [], []
    for a, b in tqdm(valloader, ncols=60, leave=False):
        a = a.cuda()
        if mode[0] == "t":
            a = transform(a)
        a = normalize(a)
        a = model(a)
        gallery.append(a)

        b = b.cuda()
        if mode[1] == "t":
            b = transform(b)
        b = normalize(b)
        b = model(b)
        probe.append(b)
    gallery = torch.cat(gallery, dim=0)
    probe = torch.cat(probe, dim=0)
    label = torch.arange(len(gallery), device=gallery.device)
    metric = CMC(gallery, probe, label)
    return metric.cmc_curve()


def privacy_pretrain(transform=lambda x: x):
    extractor.train()
    clshead.train()
    optimizer.zero_grad()
    for x, y in tqdm(trainloader, ncols=60, leave=False):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x = transform(x)
        loss = privacy_forward(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
