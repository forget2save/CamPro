import torch
import torch.nn.functional as F


__all__ = ["LowResolution", "Defocus"]


class LowResolution:
    def __init__(self, sample_rate=4) -> None:
        self.s = sample_rate

    def __call__(self, x):
        x = F.interpolate(x, scale_factor=1 / self.s, mode="area")
        x = F.interpolate(x, scale_factor=self.s, mode="bilinear")
        return x


class Defocus:
    def __init__(self, kernel_size=5, std_dev=5.0, device="cuda:0") -> None:
        self.padding = (kernel_size // 2, kernel_size // 2)

        kernel = torch.zeros((3, 3, kernel_size, kernel_size))
        kernel_template = torch.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel_template[i, j] = torch.exp(
                    -torch.tensor(
                        (i - kernel_size // 2) ** 2 / (2 * std_dev**2)
                        + (j - kernel_size // 2) ** 2 / (2 * std_dev**2)
                    )
                )
        # Normalize kernel to sum to 1
        kernel_template = kernel_template / torch.sum(kernel_template)
        for i in range(3):
            kernel[i, i] = kernel_template
        self.kernel = kernel.to(device)

    def __call__(self, x):
        x = F.conv2d(x, self.kernel, padding=self.padding)
        return x
