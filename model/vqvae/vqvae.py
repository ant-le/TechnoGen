import torch.nn as nn
import torch

from model.vqvae.encoder import Encoder
from model.vqvae.decoder import Decoder
from model.vqvae.vectorQuantiser import VectorQuantizer


def get_spectral_loss(
    out,
    x,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    epsilon: int = 2e-3,
):
    """Specral Convergence between input and output spectrograms"""

    # mps backend is not implemented for these operations
    x = x.to("cpu") if torch.backends.mps.is_available() else x
    out = out.to("cpu") if torch.backends.mps.is_available() else out

    out, x = (  # (B,C,T) -> (B,T,C)
        torch.mean(out.permute(0, 2, 1).float(), -1),
        torch.mean(x.permute(0, 2, 1).float(), -1),
    )
    spec_x = torch.stft(
        x,
        n_fft,
        hop_length=hop_length,
        return_complex=True,
        win_length=win_length,
    )

    spec_x = torch.norm(spec_x, p=2, dim=-1)
    gt_norm = (spec_x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()

    spec_out = torch.stft(
        out,
        n_fft,
        hop_length=hop_length,
        return_complex=True,
        win_length=win_length,
    )

    spec_out = torch.norm(spec_out, p=2, dim=-1)

    residual = spec_x - spec_out
    residual_norm = (residual.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()

    mask = (gt_norm > epsilon).float()
    losses = (residual_norm * mask) / torch.clamp(gt_norm, min=epsilon)
    return torch.mean(losses)


class VQVAE(nn.Module):
    def __init__(
        self,
        input_shape=(1, 44_100 * 8),
        layers: int = 5,
        kernel_size: int = 2,
        stride: int = 2,
        width: int = 16,
        depth: int = 8,
        codebook_dim: int = 32,
        codebook_size: int = 64,
        discard_vec_threshold: float = 1.0,
        codebook_loss_weight: float = 0.5,
        spectral_loss_weight: float = 1.0,
        commit_loss_weight: float = 1.0,
    ):
        super(VQVAE, self).__init__()
        print("--- Initiate VQ VAE model...")
        self.channels, self.inut_size = input_shape
        self.spectral_loss_weight = spectral_loss_weight
        self.commit_loss_weight = commit_loss_weight

        self.encoder = Encoder(
            self.channels, codebook_dim, layers, kernel_size, stride, width, depth
        )

        self.quantizer = VectorQuantizer(
            codebook_dim,
            codebook_size,
            codebook_loss_weight,
            discard_vec_threshold,
        )

        self.decoder = Decoder(
            self.channels, codebook_dim, layers, kernel_size, stride, width, depth
        )

        print(
            "--- Model running with parameter size: %.2f"
            % int(
                sum(p.numel() for p in self.parameters()) + codebook_dim * codebook_size
            )
        )

    def forward(self, x):
        # TODO: add (mutlihead) attention in encoder/decoder
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

        return out, loss, metrics
