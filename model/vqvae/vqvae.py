import torch.nn as nn
import torch
from torchaudio.transforms import InverseSpectrogram, Spectrogram

from model.vqvae.encoder import Encoder
from model.vqvae.decoder import Decoder
from model.vqvae.vectorQuantiser import VectorQuantizer


def multispectral_loss(out, x, n_fft: int = 1024, hop_length: int = 256):
    """RMSE between input and output spectrograms"""
    spec_x = torch.stft(x.float(), n_fft, hop_length, device=x.device)
    spec_out = torch.stft(out.float(), n_fft, hop_length, device=x.device)
    diff = spec_out - spec_x
    return (diff.view(diff.shape[0], -1) ** 2).sqrt()


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
        train: bool = False,
        discard_vec_threshold: float = 1.0,
        codebook_loss_weight: float = 0.8,
        spectral_loss_weight: float = 0.3,
        commit_loss_weight: float = 1.0,
    ):
        super(VQVAE, self).__init__()
        self.channels, self.inut_size = input_shape
        self.train = train
        self.spectral_loss_weight = spectral_loss_weight
        self.commit_loss_weight = commit_loss_weight

        self.encoder = Encoder(
            self.channels, codebook_dim, layers, kernel_size, stride, width, depth
        )

        self.quantizer = VectorQuantizer(
            codebook_dim,
            codebook_size,
            codebook_loss_weight,
            train,
            discard_vec_threshold,
        )

        self.decoder = Decoder(
            self.channels, codebook_dim, layers, kernel_size, stride, width, depth
        )

        print(
            "number of parameters: %.2f"
            % int(sum(p.numel() for p in self.parameters()))
        )

    def forward(self, x):
        # TODO: add (mutlihead) attention in encoder/decoder
        metrics = {}

        # Encode (B,C,T) -> (B,E,t)
        x_encoded = self.encoder(x)

        # Vector Quantization
        x_reencoded, codebook_idxsm, commit_loss, metrics = self.quantizer(x_encoded)

        # Decode
        out = self.decoder(x_reencoded)

        # Losses -> change to multispectral loss?
        reconstruction_loss = torch.mean((out - x) ** 2)
        spectral_loss = multispectral_loss(x, out)
        loss = (
            reconstruction_loss
            + self.spectral_loss_weight * spectral_loss(x, out)
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
