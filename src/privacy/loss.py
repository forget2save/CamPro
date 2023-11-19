import torch
import torch.nn.functional as F


def loss_fn_neg(logit: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
    """negative cross entropy loss"""
    loss = -F.cross_entropy(logit, label)
    return loss


def loss_fn_adv(logit: torch.Tensor, label: torch.LongTensor) -> torch.Tensor:
    """adversarial cross entropy loss"""
    eps = 1e-12
    prob = F.softmax(logit, dim=1)
    index = torch.arange(len(prob)).cuda()
    prob = prob[index, label]
    loss = -torch.log(1 - prob + eps).mean()
    return loss

