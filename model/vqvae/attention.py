import torch.nn as nn
from torch.nn import functional as F
import torch, sys, math

sys.path.append(".")


class MultiHeadAttention(nn.Module):
    def __init__(
        self, dimension: int, n_head: int = 2, project: bool = False, bias: bool = False
    ):
        super(MultiHeadAttention, self).__init__()

        assert dimension % n_head == 0

        # key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(dimension, 3 * dimension, bias=bias)
        self.anton = 2
        self.n_head = n_head
        self.dimension = dimension
        self.project = project

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        # (B,E,T) -> (B,T,E)
        x = x.permute(0, 2, 1).contiguous()
        (
            B,
            T,
            E,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.project:
            query, key, value = self.c_attn(x).split(self.dimension, dim=2)
        else:
            query = key = value = x

        key = key.view(B, T, self.n_head, E // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        query = query.view(B, T, self.n_head, E // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        value = value.view(B, T, self.n_head, E // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and x.device.type == "cuda":
            # efficient attention using Flash Attention CUDA kernels
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                is_causal=False,
            )
        else:
            # manual implementation of attention
            att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            att = F.softmax(att, dim=-1)
            out = att @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = (
            out.transpose(1, 2).contiguous().view(B, E, T)
        )  # re-assemble all head outputs side by side

        return out
