import torch.nn as nn
import sys

sys.path.append(".")


class ResBlock1D(nn.Module):
    def __init__(self, width: int, depth: int, res_scale: int = 1, dilation: int = 1):
        super(ResBlock1D, self).__init__()
        self.res_scale = res_scale
        self.model = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(width, depth, 3, 1, padding=dilation, dilation=dilation),
            nn.LeakyReLU(),
            nn.Conv1d(depth, width, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.res_scale * self.model(x)
