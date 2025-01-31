#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import math

import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor, nn

# gaussian mixture model denoisers.


class GMMEqualVarianceDenoiser(nn.Module):
    """GMM denoiser, all components same weight/variance, OU time cond."""

    # Use OU convention dx_t = -x_t dt + sqrt(2) dw_t
    # Might conflict with usage in other functions (...)

    def __init__(
        self,
        input_size: tuple[int, int, int],
        num_clusters: int,
        init_variance: float = 1.0,
        init_means: Float[Tensor, "num_clusters dim"] | None = None,
    ):
        super().__init__()
        # Parameters
        self.channels, self.height, self.width = input_size
        self.dim = self.channels * self.height * self.width
        self.num_clusters = num_clusters
        self.init_variance = init_variance
        self.one_sparse_score = 0.0
        self.uniform_score = 0.0
        # Trainable parameters
        normalize = lambda x: x / torch.norm(x, dim=(1, 2, 3), p=2, keepdim=True)
        if init_means is None:
            self.means = nn.Parameter(
                # 1 / math.sqrt(self.dim) * torch.randn(self.num_clusters, self.dim),
                normalize(torch.randn((self.num_clusters,) + input_size)),
                requires_grad=True,
            )
        else:
            self.means = nn.Parameter(
                init_means.detach(),
                requires_grad=True,
            )
        # self.standard_dev = 0 is valid.
        # But there will be numerical issues with self.score
        self.standard_dev = nn.Parameter(
            math.sqrt(self.init_variance) * torch.ones((1,)),
            requires_grad=True,
        )
        self.variance = self.standard_dev**2
        # Layers
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)
        self.unflatten = nn.Unflatten(-1, (self.channels, self.height, self.width))

    def forward(self, x: Tensor, t: Tensor, compute_scores: bool = False) -> Tensor:
        # Flatten the input
        x = self.flatten(x)
        # timescale signal
        scale, variance_t = self.get_time_scaling(t)
        scaled_clusters = self.flatten(self.means) * scale  # N x dim
        # Calculate softmax weights
        dists = torch.cdist(x[None, ...], scaled_clusters[None, ...])[0, ...]  # B x N
        weights = -0.5 * dists**2 / variance_t
        # Perform softmax autoregression
        normalized_weights = self.softmax(weights)
        cluster_term = einsum(scaled_clusters, normalized_weights, "n d,b n->b d")
        if compute_scores == True:
            max_idx = torch.argmax(weights, dim=-1)
            one_sparse = torch.zeros_like(normalized_weights)
            one_sparse[torch.arange(one_sparse.shape[0]), max_idx] = 1
            uniform = torch.ones_like(normalized_weights) / normalized_weights.shape[-1]
            # for i in range(normalized_weights.shape[0]):
            #    one_sparse[i, max_idx[i]] = 1
            one_sparse_errors = torch.sum((normalized_weights - one_sparse) ** 2, -1)
            uniform_errors = torch.sum((normalized_weights - uniform) ** 2, -1)
            self.one_sparse_score = torch.mean(one_sparse_errors)
            self.uniform_score = torch.mean(uniform_errors)
        # Combine with input
        denoiser_weight = scale**2 * self.standard_dev**2 / variance_t
        x = denoiser_weight * x + (1 - denoiser_weight) * cluster_term
        x = x / scale
        # Lift and return
        x = self.unflatten(x)
        return x

    def get_time_scaling(self, t: Tensor) -> tuple[Tensor, Tensor]:
        scale = torch.exp(-t)  # controls the OU parameterization convention
        variance = 1 - scale**2
        variance_t = scale**2 * self.standard_dev**2 + variance
        return scale, variance_t

    def score(self, x: Tensor, t: Tensor) -> Tensor:
        """Call on raw samples (instead of calling forward)."""
        # PERF: can be numerical stability issues
        scale, variance_t = self.get_time_scaling(t)
        variance = 1 - scale**2
        denoised = self.forward(x, t)
        return (denoised * scale - x) / variance

    def get_means(self, t: Tensor = torch.zeros((1,))) -> Tensor:
        """Returns mean parameters, detached from computational graph."""
        scale, variance_t = self.get_time_scaling(t)
        return scale * self.means.detach()

    def get_variances(self, t: Tensor = torch.zeros((1,))) -> Tensor:
        """Returns variance parameters, detached from computational graph."""
        scale, variance_t = self.get_time_scaling(t)
        scale, variance_t = self.get_time_scaling(t)
        return variance_t * torch.ones((self.num_clusters,))

    def generate_samples(
        self, num_samples: int, t: Tensor = torch.zeros((1,))
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate samples from the GMM. For Debug."""
        # time stuff
        scale, variance_t = self.get_time_scaling(t)
        # cluster sampling
        cluster_idxs = torch.randint(self.num_clusters, (num_samples,))
        means = scale * self.flatten(
            self.means[cluster_idxs, ...].detach()
        )  # num_samples x dim
        # cluster-conditional sampling
        gaussians = torch.randn(num_samples, self.dim)
        samples = means + gaussians * torch.sqrt(variance_t)
        samples = self.unflatten(samples)
        return samples, self.means[cluster_idxs, ...].detach().clone(), cluster_idxs
