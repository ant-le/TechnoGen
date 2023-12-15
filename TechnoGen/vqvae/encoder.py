import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, channels: [int], kernel_size: int):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
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
        for layer in self.encoder_layers:
            out = layer(out)
        return out
