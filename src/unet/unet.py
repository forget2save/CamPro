import torch
import torchvision as tv
from torch import nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope, True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        return x


class UNetUpsample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = tv.transforms.functional.center_crop(x2, x1.shape[-2:])
        x = torch.cat([x1, x2], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, depth=5, base_channels=32) -> None:
        super().__init__()
        self.depth = depth
        self.pool = nn.MaxPool2d(2)
        x = 3
        y = base_channels
        for i in range(1, depth + 1):
            self.add_module(f"block_d{i}", UNetBlock(x, y))
            x, y = y, y * 2
        y = x // 2
        for i in range(1, depth):
            self.add_module(f"block_u{i}", UNetBlock(x, y))
            self.add_module(f"upsample{i}", UNetUpsample(x, y))
            x, y = y, y // 2
        self.head = nn.Conv2d(x, 3, 1)

    def forward(self, x: torch.Tensor):
        feat = []
        for i in range(1, self.depth):
            tmp = self.get_submodule(f"block_d{i}")(x)
            feat.append(tmp)
            x = self.pool(tmp)
        x = self.get_submodule(f"block_d{self.depth}")(x)

        for i in range(1, self.depth):
            x = self.get_submodule(f"block_u{i}")(
                self.get_submodule(f"upsample{i}")(x, feat[-i])
            )
        x = self.head(x)
        return x

