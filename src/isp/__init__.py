from torch import optim

from .pipeline import Pipeline


camera, optimizer = None, None

__all__ = ["camera_init", "camera_forward", "camera_backward"]


def camera_init(cfg):
    global camera, optimizer

    camera = Pipeline(cfg.ISP.MODULE, cfg.ISP.PARAM)
    if cfg.ISP.OPT == "sgd":
        optimizer = optim.SGD(
            camera.parameters(),
            lr=cfg.ISP.LR,
        )
    elif cfg.ISP.OPT == "adam":
        optimizer = optim.Adam(
            camera.parameters(),
            lr=cfg.ISP.LR,
        )
    else:
        raise NotImplementedError()


def camera_forward(x):
    x = camera.forward(x)
    return x


def camera_backward():
    optimizer.step()
    optimizer.zero_grad()
    camera.clamp()
