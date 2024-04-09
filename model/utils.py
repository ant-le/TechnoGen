import torch, sys

sys.path.append(".")

from einops import rearrange, repeat


def kmeans(x, codebook_size: int, num_iters: int = 10):
    print("--- Initialising codebook with k-means: May take some time...")
    # code was adapted from and slightly modified from
    # https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py
    # (BT, E)
    BT, E = x.shape
    dtype = x.dtype

    # radnomly sample E initial vectors
    if BT >= codebook_size:
        indices = torch.randperm(BT, device=x.device)[:codebook_size]
    else:
        indices = torch.randint(0, BT, (codebook_size,), device=x.device)
    # (E_bins, E)
    means = x[indices]

    # iteratively find cluster centers
    for _ in range(num_iters):
        diffs = rearrange(x, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=codebook_size)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(codebook_size, E, dtype=dtype)
        new_means = new_means / bins_min_clamped[..., None]
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=E), x)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means


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
    x = x.to("cpu") if x.device.type == "mps" else x
    out = out.to("cpu") if out.device.type == "mps" else out

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
