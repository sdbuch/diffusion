#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
from torch.nn.functional import softmax

from denoisers.gmm import GMMEqualVarianceDenoiser


# For this experiment:
# - K=1 GMM with cluster center = 0.
# - Look at d_{1j} random variables.
def eval_softmax_approx(
    dims: torch.Tensor,
    times: torch.Tensor,
    samples: torch.Tensor,
    beta: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    results_mean = torch.zeros((len(samples), len(dims), len(times)), device=device)
    results_stdev = torch.zeros((len(samples), len(dims), len(times)), device=device)
    T = 1000
    model = GMMEqualVarianceDenoiser((1, 1, 1), 1)
    for idx_n in range(len(samples)):
        for idx_d in range(len(dims)):
            for idx_t in range(len(times)):
                n = int(samples[idx_n])
                d = int(dims[idx_d])
                t = times[idx_t]
                scale, variance_t = model.get_time_scaling(t)
                variance = 1 - scale**2
                G = scale * beta * torch.randn((T, int(n), d), device=device)
                g = G[:, 0, :].unsqueeze(1)
                w = torch.sqrt(variance) * torch.randn((T, 1, d), device=device)
                dists_sq = torch.cdist(G, g + w).squeeze(-1) ** 2
                weights = -0.5 * dists_sq / variance
                output = softmax(weights, dim=-1)
                one_sparse = torch.zeros((1, n), device=device)
                one_sparse[0, 0] = 1
                error = torch.cdist(output, one_sparse).squeeze(-1)
                results_mean[idx_n, idx_d, idx_t] = torch.mean(error)
                results_stdev[idx_n, idx_d, idx_t] = torch.std(error)
    return results_mean, results_stdev


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta = torch.sqrt(torch.tensor(0.25, device=device))
    dims = torch.arange(128, 256)
    times = torch.logspace(-6, 0, 100)
    samples = torch.arange(16, 64, 2)
    results_mean, results_stdev = eval_softmax_approx(
        dims=dims, times=times, samples=samples, beta=beta, device=device
    )
    # TODO:
    # 1. Set the times variable based on beta (in general)
    # 2. Pick a hypothesis that a time around c \beta^2 is when the approximation breaks.
    # 3. For different values of c, perform a dims-samples heatmap plot. Examine the transition, in particular whether the approximation can be made arbitrarily bad by increasing c in a reasonable part of the plane (e.g., n <= d^2; label it)
