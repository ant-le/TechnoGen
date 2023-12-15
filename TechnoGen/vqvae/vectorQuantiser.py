import torch
import torch.nn as nn
from einops import einsum


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        n_embedding: int,
    ):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(n_embedding, dim_embedding)

    def forward(self, x):
        B, C, H, W = x.shape
        x_premuted = x.permute(0, 2, 3, 1)
        x_flattened = x_premuted.reshape(x_premuted.size(0), -1, x_premuted.size(-1))

        distances = torch.cdist(
            x_flattened,
            self.embedding.weight[None, :].repeat((x_flattened.size(0), 1, 1)),
        )
        min_encoding_indices = torch.argmin(distances, dim=-1)

        #
        quant_out = torch.index_select(
            self.embedding.weight, 0, min_encoding_indices.view(-1)
        )

        x_permuted = x_flattened.reshape((-1, x_flattened.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x_permuted) ** 2)
        codebook_loss = torch.mean((quant_out - x_permuted.detach()) ** 2)
        losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commmitment_loss,
        }
        quant_out = x_permuted + (quant_out - x_permuted).detach()

        x = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape(
            (-1, x.size(-2), x.size(-1))
        )
        return x, losses, min_encoding_indices

    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, "b n h w, n d -> b d h w")
