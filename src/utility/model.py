import yaml
import os.path as osp
import torch
from models.yolo import Model

weight_folder = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
    "..",
    "weights",
)
pretrained = osp.join(weight_folder, "yolov5m.pt")

yolov5_root = osp.join(osp.dirname(osp.abspath(__file__)), "..", "yolov5")
model_yaml = osp.join(yolov5_root, "models", "yolov5m.yaml")
hyp_yaml = osp.join(yolov5_root, "data", "hyps", "hyp.scratch-low.yaml")
with open(hyp_yaml, "r") as f:
    hyp = yaml.safe_load(f)

def pretrained_yolov5():
    ckpt = torch.load(pretrained, map_location="cpu")["model"].state_dict()
    model = Model(model_yaml, nc=1)
    model.load_state_dict(ckpt)
    model.nc = 1
    model.hyp = hyp
    return model
