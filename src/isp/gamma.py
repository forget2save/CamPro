import torch
import torch.nn as nn


def degamma(x: torch.Tensor):
    return torch.clamp(torch.pow(x, 2.2), 0, 1)


class Gamma(nn.Module):
    def __init__(self, n_point=32) -> None:
        super().__init__()
        self.keypoints = nn.Parameter(torch.linspace(0, 1, n_point))
        self.n_space = n_point - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        x_ = torch.clamp(x, eps, 1 - eps)
        i = torch.floor(x_ * self.n_space).long()
        alpha = x * self.n_space - i
        y = self.keypoints[i] * (1 - alpha) + self.keypoints[i + 1] * alpha
        return y

    @torch.no_grad()
    def clamp(self) -> None:
        self.keypoints.data.clamp_(0, 1)

