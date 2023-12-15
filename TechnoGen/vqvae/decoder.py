import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, channels: [int], kernel_size: int):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=kernel_size,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(),
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.decoder_layers:
            out = layer(out)
        return out
