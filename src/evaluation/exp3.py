import sys
import os.path as osp

root = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
)
sys.path.append(root)

import pandas as pd
import numpy as np
import torch
from rich.progress import track

from utility.data import coco_person_dataloader
from utility.model import pretrained_yolov5
from utility.metric import parse_stats
from utils.general import non_max_suppression

from isp.pipeline import Pipeline
from baseline import LowResolution, Defocus


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if j == 0 or j == 5:
                    py.append(
                        np.interp(px, mrec, mpre)
                    )  # precision at mAP@0.5 and mAP@0.75
    f1 = 2 * p * r / (p + r + eps)
    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    ap = ap[0]
    return ap, p, r, f1


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


@torch.no_grad()
def main():
    camera = Pipeline.load("../checkpoints/default.pt")
    lowres = LowResolution(16)
    defocus = Defocus(25, 7)

    detector = pretrained_yolov5().cuda().eval()
    transforms = [
        ("Raw Images", lambda o: o),
        ("Low-Resolution", lowres),
        ("Defocused", defocus),
        ("CamPro", camera.forward),
    ]
    _, loader = coco_person_dataloader(batch_size=16, num_workers=8)

    table = []
    for t_name, t_func in transforms:
        if t_name == "CamPro":
            ckpt = "../checkpoints/captured.pt"
            detector.load_state_dict(torch.load(ckpt))
        elif t_name == "Low-Resolution":
            ckpt = "../checkpoints/lowres.pt"
            detector.load_state_dict(torch.load(ckpt))
        elif t_name == "Defocused":
            ckpt = "../checkpoints/defocus.pt"
            detector.load_state_dict(torch.load(ckpt))
        stats = []
        for img, lbl, _, shapes in track(loader):
            img, lbl = img.cuda().float().div(255), lbl.cuda()
            img = t_func(img)
            out, _ = detector(img)
            out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.6)
            parse_stats(img, lbl, shapes, out, stats)
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        ap, p, r, f1 = ap_per_class(*stats)
        ap_50 = ap[0]
        ap_75 = ap[5]
        ap = ap.mean()
        data = [ap, ap_50, ap_75, p, r, f1]
        data = list(map(lambda o: f"{float(o):.6f}", data))
        table.append([t_name] + data)

    df = pd.DataFrame(
        table, columns=[" ", "AP", "AP@0.5", "AP@0.75", "Precision", "Recall", "F1"]
    )
    df = df.reset_index(drop=True)
    print(df.head())
    df.to_csv("../results/3.csv", index=False)


if __name__ == "__main__":
    main()
