import torch.nn as nn

from model.vqvae.resid import ResBlock1D
from model.vqvae.lstm import LSTMBlock1D


class Encoder(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_dim: int,
        layers: int,
        stride: int,
        width: int,
        depth: int,
        lstm: bool,
    ):
        super(Encoder, self).__init__()

        kernel, padding = stride * 2, stride // 2

        blocks = []
        # downsampling
        for i in range(layers):
            block = nn.Sequential(
                nn.Conv1d(
                    channels if i == 0 else width, width, kernel, stride, padding
                ),
                ResBlock1D(width, depth),
            )
            blocks.append(block)

        if lstm:
            block = LSTMBlock1D(width)
            blocks.append(block)

        block = nn.Conv1d(width, codebook_dim, 3, 1, 0)
        blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
