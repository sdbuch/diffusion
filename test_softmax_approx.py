#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
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


# For this experiment:
# - K=1 GMM with cluster center = 0.
# - Look at random variables corresponding to certain random inits.
# - Want to see if there is still a gap...
def eval_softmax_approx_randinit(
    dims: torch.Tensor,
    times: torch.Tensor,
    samples: torch.Tensor,
    beta: torch.Tensor,
    device: torch.device,
    init_variance: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    results_mean = torch.zeros((len(samples), len(dims), len(times)), device=device)
    results_stdev = torch.zeros((len(samples), len(dims), len(times)), device=device)
    T = 1000
    model = GMMEqualVarianceDenoiser((1, 1, 1), 1, init_variance = 0.25)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    for idx_n in range(len(samples)):
        for idx_d in range(len(dims)):
            for idx_t in range(len(times)):
                n = int(samples[idx_n])
                d = int(dims[idx_d])
                t = times[idx_t]
                scale, variance_t = model.get_time_scaling(t)
                variance = 1 - scale**2
                if init_variance is None:
                    # This case attempts to sphere normalize.
                    G = scale * sqrt(1/d) * torch.randn((T, int(n), d), device=device)
                else:
                    G = scale * sqrt(init_variance) * torch.randn((T, int(n), d), device=device)
                g = scale * beta * torch.randn((T, 1, d), device=device)
                w = torch.sqrt(variance) * torch.randn((T, 1, d), device=device)
                dists_sq = torch.cdist(G, g + w).squeeze(-1) ** 2
                weights = -0.5 * dists_sq / variance_t
                output = softmax(weights, dim=-1)
                one_sparse = torch.zeros((1, n), device=device)
                one_sparse[0, 0] = 1
                error = torch.cdist(output, one_sparse).squeeze(-1)
                results_mean[idx_n, idx_d, idx_t] = torch.mean(error)
                results_stdev[idx_n, idx_d, idx_t] = torch.std(error)
    return results_mean, results_stdev



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta = torch.sqrt(torch.tensor(0.25))
    dims = torch.arange(30, 32, 2)
    # Regime 1: small times
    times = beta**2 * torch.logspace(-0, 1, 25)
    # Regime 2: large times. Test if it works.
    # times = torch.logspace(-6, 1, 25)
    samples = torch.arange(16, 64, 2)
    results_mean, results_stdev = eval_softmax_approx_randinit(
        dims=dims, times=times, samples=samples, beta=beta, device=device, init_variance=1.0
    )
    # 2. Pick a hypothesis that a time around c \beta^2 is when the approximation breaks.
    #    for OU, more like c log(1 + C beta^2), which may linearize.
    const = 5
    # Regime 1: target the cut time
    target_time = 0.5 * torch.log(1 + const * beta**2)
    # Regime 2: more arbitrary
    # target_time = 1.0
    closest_time_idx = torch.argmin(torch.abs(times - target_time))
    if closest_time_idx == 0 or closest_time_idx == len(times) - 1:
        warnings.warn("Closest time on the edge of times array: might be OOB.")
    slice_mean, slice_std = (
        results_mean[:, :, closest_time_idx],
        results_stdev[:, :, closest_time_idx],
    )
    # 3. For different values of c, perform a dims-samples heatmap plot.
    # Examine the transition, in particular whether the approximation can be
    # made arbitrarily bad by increasing c in a reasonable part of the plane
    # (e.g., n <= d^2; label it)
    # Convert the tensor to a NumPy array for plotting
    data_np = slice_mean.cpu().numpy()
    xval_np = dims.cpu().numpy()
    yval_np = samples.cpu().numpy()
    # Define the curve y = f(x) (e.g., a simple quadratic curve)
    x_curve = np.linspace(xval_np.min(), xval_np.max(), 100)
    y_curve = x_curve**2  # Example curve: quadratic function

    # Plot the heatmap with custom x and y axis values
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data_np,
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=1,
        extent=[xval_np[0], xval_np[-1], yval_np[-1], yval_np[0]],
    )
    # Overlay the curve on top of the heatmap
    plt.plot(x_curve, y_curve, color="red", label="samples = dims^2")

    # Set axes limits to ensure the heatmap and curve align properly
    plt.xlim(xval_np[0], xval_np[-1])
    plt.ylim(yval_np[-1], yval_np[0])

    # Set custom tick labels
    subsample_rate_y = 2
    subsample_rate_x = 2
    plt.xticks(
        ticks=xval_np[::subsample_rate_x],
        labels=[f"{val:.0f}" for val in xval_np[::subsample_rate_x]],
    )
    plt.yticks(
        ticks=yval_np[::subsample_rate_y],
        labels=[f"{val:.0f}" for val in yval_np[::subsample_rate_y]],
    )
    plt.colorbar(label="Value")
    plt.title(f"Heatmap of Averaged ell-2 Softmax ApproxRates (1-Sparse, c={const:.1f})")
    plt.xlabel("Ambient dimension")
    plt.ylabel("Number of samples")
    plt.gca().invert_yaxis()
    plt.show()

    # Make another plot of the time-wise behavior
    # Do it for smallest samples; largest dim
    slice_time_mean, slice_time_std = results_mean[0, -1, :].cpu(), results_stdev[0, -1, :].cpu()
    plt.errorbar(times, slice_time_mean, yerr=slice_time_std, label=f'target time={target_time:.3f}')
    plt.title(f"d={dims[-1]}, n={samples[0]}")
    plt.xlabel('time')
    plt.ylabel('softmax ell2 approximation error (averaged)')
    plt.legend()
    plt.show()
