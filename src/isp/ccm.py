import torch
import torch.nn as nn


class CCM(nn.Module):
    def __init__(self, noise=0.1) -> None:
        super().__init__()
        self.ccm = nn.Parameter(
            torch.eye(3, dtype=torch.float32) 
            + noise * (torch.rand((3, 3), dtype=torch.float32) - 0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        x = x.permute(1, 0, 2, 3).flatten(start_dim=1)
        x = torch.mm(self.ccm, x)
        x = torch.clamp(x, 0, 1)
        y = x.view(3, n, h, w).permute(1, 0, 2, 3)
        return y

    @torch.no_grad()
    def clamp(self) -> None:
        self.ccm.data.clamp_(-5, 5)
