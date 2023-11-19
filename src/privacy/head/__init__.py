from .softmax import SoftmaxHead
from .arcface import ArcFaceHead


def head_type(varient):
    if varient == "softmax":
        return SoftmaxHead
    elif varient == "arcface":
        return ArcFaceHead
    else:
        raise ValueError()
