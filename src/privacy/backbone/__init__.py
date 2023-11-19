from .facenet import InceptionResnetV1
from .ir152 import IR_152, IR_SE_50
from .adaface import build_model
from .magface import builder_inf
import torch
import os.path as osp


weight_folder = osp.join(
    osp.dirname((osp.abspath(__file__))),
    "..",
    "..",
    "..",
    "weights",
)


def get_available_models():
    return [
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


def pretrained_backbone(varient: str, pretrained=True):
    """varient should be in [facenet, mobile, ir152, irse50, ir18]"""
    if varient == "facenet":
        net = InceptionResnetV1(pretrained="vggface2")
    elif varient == "arcface_ir18":
        net = builder_inf(varient, pretrained=pretrained)
        return net
    elif varient == "arcface_ir152":
        net = IR_152((112, 112))
    elif varient == "arcface_irse50":
        net = IR_SE_50((112, 112))
    elif varient.startswith("magface"):
        net = builder_inf(varient, pretrained=pretrained)
        return net
    elif varient.startswith("adaface"):
        net = build_model(varient.split("_")[1])
    else:
        print(varient)
        raise NotImplementedError()
    if pretrained:
        weights = osp.join(weight_folder, f"{varient}.pth")
        net.load_state_dict(torch.load(weights))
    return net
