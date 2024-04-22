    import torch.nn as nn
import torch.nn.functional as F
import torch, sys

sys.path.append(".")

from model.vqvae.encoder import Encoder
from model.vqvae.decoder import Decoder
from model.vqvae.vectorQuantizer import VectorQuantizer
from model.utils import get_spectral_loss


class VQVAE(nn.Module):
    def __init__(
        self,
        input_shape=(1, 44_000 * 8),
        layers: int = 6,
        stride: int = 2,
        width: int = 128,
        depth: int = 3,
        codebook_dim: int = 64,
        codebook_size: int = 512,
        discard_vec_threshold: float = 1.0,
        codebook_loss_weight: float = 0.8,
        spectral_loss_weight: float = 1.0,
        commit_loss_weight: float = 0.8,
        init_random: bool = True,
        lstm: bool = False,
    ):
        """Vector Qunatized Convolutional Variational Autoencoder
        Args:
            input_shape (tuple, optional): _description_.
            layers (int, optional): Defines compression rate
            kernel_size (int, optional): _description_.
            stride (int, optional): _description_.
            width (int, optional): _description_.
            depth (int, optional): _description_.
            codebook_dim (int, optional): _description_.
            codebook_size (int, optional): _description_.
            discard_vec_threshold (float, optional):
            codebook_loss_weight (float, optional):
            spectral_loss_weight (float, optional):
            commit_loss_weight (float, optional):
            init_random (bool, optional):
        """
        super(VQVAE, self).__init__()
        print("--- Initiate VQ VAE model...")

        self.channels, self.input_size = input_shape
        self.compression_level = self.input_size // (stride**layers)

        self.spectral_loss_weight = spectral_loss_weight
        self.commit_loss_weight = commit_loss_weight
        self.codebook_dim = codebook_dim

        self.encoder = Encoder(
            self.channels, codebook_dim, layers, stride, width, depth, lstm
        )

        self.quantizer = VectorQuantizer(
            codebook_dim,
            codebook_size,
            codebook_loss_weight,
            discard_vec_threshold,
            init_random,
        )

        self.decoder = Decoder(
            self.channels, codebook_dim, layers, stride, width, depth, lstm
        )

        print(
            "--- Model running with parameter size: %.2f"
            % int(
                sum(p.numel() for p in self.parameters()) + codebook_dim * codebook_size
            )
        )

    def forward(self, x):
        # store resulting metrics in dict
        metrics = {}

        # Encode (B,C,T) -> (B,E,t)
        x_encoded = self.encoder(x)

        # Vector Quantization
        x_reencoded, commit_loss, metrics = self.quantizer(x_encoded)

        # Decode
        out = self.decoder(x_reencoded)

        # Losses -> change to multispectral loss?
        reconstruction_loss = torch.mean(
            (x.permute(0, 2, 1) - out.permute(0, 2, 1)) ** 2
        )
        spectral_loss = get_spectral_loss(out, x)
        loss = (
            reconstruction_loss
            + self.spectral_loss_weight * spectral_loss
            + self.commit_loss_weight * commit_loss
        )

        metrics.update(
            dict(
                recons_loss=reconstruction_loss,
                spectral_loss=spectral_loss,
                commit_loss=commit_loss,
            )
        )

        for key, val in metrics.items():
            metrics[key] = val.detach()

        if not self.training:
            out = out.detach()

        return out, loss, metrics

    @torch.no_grad()
    def generate(self):
        generated_vectors = self.quantizer.get_random_codebook_vetors(
            self.compression_level
        )
        out = self.decoder(generated_vectors)
        return out

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x)

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)
