from torch import nn
import sys

sys.path.append(".")


class LSTMBlock1D(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 1, resid: bool = True):
        super(LSTMBlock1D, self).__init__()

        self.resid = resid
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        out, _ = self.lstm(x)
        if self.resid:
            out = out + x
        out = out.permute(1, 2, 0)
        return out
