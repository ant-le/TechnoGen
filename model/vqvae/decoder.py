import torch.nn as nn

from model.vqvae.resid import ResBlock1D


class Decoder(nn.Module):
    def __init__(
        self,
        channels: int,
        codebook_dim: int,
        layers: int,
        kenel_size: int,
        stride: int,
        width: int,
        depth: int,
    ):
        super(Decoder, self).__init__()

        kernel, padding = stride * kenel_size, stride // kenel_size

        blocks = []
        block = nn.ConvTranspose1d(codebook_dim, width, 3, 1, 1)
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
