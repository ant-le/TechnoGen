import sys

sys.path.append(".")
import torch.nn as nn
from model.vqvae.resid import ResBlock1D
from model.vqvae.lstm import LSTMBlock1D


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

        kernel, padding = stride * 2, stride // 2

        blocks = []
        block = nn.ConvTranspose1d(codebook_dim, width, 3, 1, 0)
        blocks.append(block)

        if lstm:
            block = LSTMBlock1D(width)
            blocks.append(block)

        # upsampling
        for i in range(layers):
            block = nn.Sequential(
                ResBlock1D(width, depth),
                nn.ConvTranspose1d(
                    width,
                    channels if i == (layers - 1) else width,
                    kernel,
                    stride,
                    padding,
                ),
            )
            blocks.append(block)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
