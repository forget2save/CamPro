import torch
import numpy as np
from utils.general import xywh2xyxy, scale_coords
from utils.metrics import box_iou, ap_per_class


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(
        detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
    )
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where(
        (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
    )  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = (
            torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        )  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def parse_stats(imgs, targets, shapes, out, stats: list):
    targets = targets.clone()
    out = [x.clone() for x in out]

    height, width = imgs.shape[-2:]
    targets[:, 2:] *= torch.Tensor([width, height, width, height]).cuda()
    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    iouv = torch.linspace(0.5, 0.95, 10).cuda()
    niou = iouv.numel()

    for si, pred in enumerate(out):
        infer_shape = imgs[si].shape[1:]
        shape = shapes[si][0]
        ratio_pad = shapes[si][1]
        labels = targets[targets[:, 0] == si, 1:]
        tbox = labels[:, 1:5]
        pbox = pred[:, :4]
        scale_coords(infer_shape, pbox, shape, ratio_pad)
        scale_coords(infer_shape, tbox, shape, ratio_pad)
        tcls = labels[:, 0].tolist() if len(labels) else []  # target class

        if len(pred) == 0:
            if len(labels):
                stats.append(
                    (
                        torch.zeros(0, niou, dtype=torch.bool),
                        torch.Tensor(),
                        torch.Tensor(),
                        tcls,
                    )
                )
            continue

        # Evaluate
        if len(labels):
            correct = process_batch(pred, labels, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append(
            (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
        )  # (correct, conf, pcls, tcls)


def cal_ap(stats):
    names = {0: "person"}
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0].mean(), ap.mean(1).mean()
        return ap50, ap
    else:
        return 0, 0
