import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=E), x)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_dim: int,
        codebook_size: int,
        codebook_loss_weight: float,
        discard_vec_threshold: int = 1,
        init_random: bool = True,
    ):
        super(VectorQuantizer, self).__init__()

        # codebook dimensions
        self.codebook_dim = codebook_dim  # CD_dim = E
        self.codebook_size = codebook_size  # CB_size
        self.loss_weight = codebook_loss_weight
        self.discard_vec_threshold = discard_vec_threshold
        self.init_random = init_random

        # instantiate codebook instance in buffer -> no model param
        codebook = torch.zeros(codebook_size, codebook_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("codebook_avg", codebook.clone())
        self.register_buffer("vector_usage", torch.ones(codebook_size))
        self.register_buffer("inited", torch.Tensor([False]))

    def inititalise_codeboook(self, x):
        assert x.shape[1] == self.codebook_dim and len(x.shape) == 2
        if not self.init_random:
            try:
                # find cluster means serving as initial vectors in codeboook
                codebook = kmeans(x, self.codebook_size)
            except RuntimeError:
                print("--- Memory Size too small: Initialsing randomly instead!")
                codebook = x[torch.randperm(x.shape[0])][: self.codebook_size]
        else:
            codebook = x[torch.randperm(x.shape[0])][: self.codebook_size]

        # find cluster means serving as initial vectors in codeboook

        # Update codebook
        self.codebook = codebook
        self.codebook_avg = codebook.clone()

    def update_codebook(self, x: torch.Tensor, codebook_idxs: []):
        # get params for better readability
        weight = self.loss_weight
        cb_size = self.codebook_size
        cb_dim = self.codebook_dim
        threshold = self.discard_vec_threshold

        # we do not backpropagate through quantiser
        with torch.no_grad():
            # create representation of codebook vectors over input (CB_size, BT)
            codebook_idxs_onehot = torch.zeros(
                self.codebook_size, x.shape[0], device=x.device
            )
            # (CB_size, BT) and (1,BT) -> (CB_size, BT) with onehot codebook vectors
            codebook_idxs_onehot.scatter_(0, codebook_idxs.view(1, x.shape[0]), 1)

            # (CB_size, BT) * (BT, E) -> (CB_size, E) -> (CB_size, 1)
            # count information for each codebook vector over
            # codebook dimensions and count for each vector
            codebook_avg = torch.matmul(codebook_idxs_onehot, x)
            vector_usage = codebook_idxs_onehot.sum(dim=-1)

            # update codebook instances
            old_codebook = self.codebook
            # information about vector quality (CB_size, E)
            self.codebook_avg = weight * self.codebook_avg + (1 - weight) * codebook_avg

            # information about vector usage (CB_size, 1)
            self.vector_usage = weight * self.vector_usage + (1 - weight) * vector_usage

            # check if usage of certain codebook vectors is beyond threshold
            # in last iteration (false -> 0. & true -> 1.)
            usage = (self.vector_usage.view(cb_size, 1) >= threshold).float()
            assert x.shape[0] >= cb_size  # BT >= number of codebook vectors
            random_shift = x[torch.randperm(x.shape[0])][:cb_size]

            # update vectors according and change unimportant vecotors randomly
            self.codebook = (
                usage
                * (
                    self.codebook_avg.view(cb_size, cb_dim)
                    / self.vector_usage.view(cb_size, 1)
                )
                + (1 - usage) * random_shift
            )

            # calculate codebook evaluations
            vector_prob = vector_usage / torch.sum(vector_usage)
            # asses variablity in vector selection
            entropy = -torch.sum(vector_prob * torch.log(vector_prob + 1e-8))
            # number of used vecs in current iteration
            usage_curr = (vector_usage >= threshold).sum()
            # number if used vecs of last iterations
            usage_last = torch.sum(usage)
            #
            codebook_diff = torch.norm(self.codebook - old_codebook) / np.sqrt(
                np.prod(old_codebook.shape)
            )
            return dict(
                vector_prob=vector_prob,
                entropy=entropy,
                usage_curr=usage_curr,
                usage_last=usage_last,
                codebook_diff=codebook_diff,
            )

    def preprocess(self, x):
        # (B,E,T) -> (B,T,E)
        x = x.permute(0, 2, 1).contiguous()
        # (B,T,E) -> (BT,E)
        x = x.view(-1, x.shape[-1])

        # we must be now have our embedding
        # dimension hyperparameter value
        assert x.shape[-1] == self.codebook_dim

        # normalise x for stability
        x_norm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        return x, x_norm

    def postprocess(self, out, old_shape):
        B, T = old_shape
        # (BT, E) -> (B,T,E)
        out = out.view(B, T, -1)
        # (B,T,E) -> (B,E,T)
        out = out.permute(0, 2, 1).contiguous()
        return out

    def forward(self, x):
        B, E, T = x.shape
        # preprocess to get right shape
        # (B,E,T) -> (BT, E)
        x, x_norm = self.preprocess(x)

        if not self.inited and self.training:
            self.inititalise_codeboook(x)
            self.inited.data.copy_(torch.Tensor([True]))

        # get latent vectors from codebook
        codebook_vecs = self.codebook.t()
        # pairwise distances between vectors and all
        # downsampled timeperiods in all batches
        # (BT, E) x (E, CB_size) -> (BT, CB_size)
        distances = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, codebook_vecs)
            + torch.sum(codebook_vecs**2, dim=0, keepdim=True)
        )

        # for each batch - time combination find the closest
        # codebook vector and store their distance
        min_distances, codebook_idxs = torch.min(distances, dim=-1)

        # how well do found codebook vectors match input
        fit = torch.mean(min_distances)

        # return the corresponding codebook vector for
        # time-batch combination (BT, E)
        out = F.embedding(codebook_idxs, self.codebook)

        # update codebook vectors
        codebook_updates = (
            self.update_codebook(x, codebook_idxs) if self.training else {}
        )

        # calculate commit loss and stop gradients from going
        # through the quantiser
        # Loss: scaled difference between preprocessed input and
        # codebook vectors
        commit_loss = torch.norm(out.detach() - x) ** 2 / np.prod(x.shape)

        # Feed foward the vectors -> and make sure that
        # encoder weights are not changed
        out = x + (out - x)
        out = out.detach()

        # post data back to right shape
        # (BT, E) -> (B,E,T)
        out = self.postprocess(out, (B, T))

        # codebook indices for each point
        # (E, CB_size) -> (B, T)
        codebook_idxs = codebook_idxs.view(B, T)

        if not self.training:
            out = out.detach()
        return (
            out,
            commit_loss,
            dict(fit=fit, x_norm=x_norm, **codebook_updates),
        )
