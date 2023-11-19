import torch
from torch import nn
from collections import OrderedDict
from .ccm import CCM
from .gamma import Gamma, degamma


class Pipeline:
    def __init__(self, module, param=None) -> None:
        self.ops = []
        self.module = module
        if param is None:
            param = [None] * len(module)
        for m, p in zip(module, param):
            if p is None:
                if m == "ccm":
                    self.ops.append(CCM().cuda())
                    self.ccm = self.ops[-1]
                elif m == "gamma":
                    self.ops.append(Gamma().cuda())
                    self.gamma = self.ops[-1]
                elif m == "degamma":
                    self.ops.append(degamma)
                else:
                    raise NotImplementedError()
            else:
                if m == "ccm":
                    self.ops.append(CCM(p).cuda())
                    self.ccm = self.ops[-1]
                elif m == "gamma":
                    self.ops.append(Gamma(p).cuda())
                    self.gamma = self.ops[-1]
                elif m == "degamma":
                    self.ops.append(degamma)
                else:
                    raise NotImplementedError()

    def parameters(self):
        ret = []
        for op in self.ops:
            if isinstance(op, nn.Module):
                ret.append({"params": op.parameters()})
        return ret

    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x

    def clamp(self):
        for op in self.ops:
            if isinstance(op, nn.Module) and hasattr(op, "clamp"):
                op.clamp()

    def save(self, save_path):
        d = OrderedDict()
        for m, op in zip(self.module, self.ops):
            if isinstance(op, nn.Module):
                d[m] = op.state_dict()
            else:
                d[m] = None
        torch.save(d, save_path)

    @staticmethod
    def load(save_path):
        d = torch.load(save_path)
        module = list(d.keys())
        # print(module)
        param = []
        for m in module:
            if m == "gamma":
                param.append(d[m]["keypoints"].shape[0])
            else:
                param.append(None)
        # print(param)
        p = Pipeline(module, param)
        for m, op in zip(p.module, p.ops):
            if isinstance(op, nn.Module):
                try:
                    op.load_state_dict(d[m])
                except Exception:
                    print(m, "fail to load")
        return p
