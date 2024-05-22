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
import tyro


def gaussian_forward():
    # Implement euler-maruyama scheme
    # Parameters: discretization points (and/or step size + scheme), max time T

    device = torch.device("mps")
    d = 1
    num_trials = 1000
    X0 = torch.zeros((d, num_trials), device=device)

    # scheme: uniform discretization; specify number of points
    T = 2
    num_times = 100
    times = torch.linspace(0, T, 100)

    # sample trajectories
    trajectories = torch.zeros((d, num_trials, num_times), device=device)
    trajectories[:, :, 0] = X0  # CHECK: broadcasts?
    for time_idx in range(1, num_times):
        # time_idx corresponds to iter i+1
        # the loop generates X_{i+1} from X_{i}
        time = times[time_idx - 1]
        next_time = times[time_idx]
        delta = next_time - time
        current_X = trajectories[:, :, time_idx - 1]
        drift = -0.5 * current_X
        trajectories[:, :, time_idx] = (
            current_X
            + drift * delta
            + torch.sqrt(delta) * torch.randn((d, num_trials), device=device)
        )

    # eval
    # visualize a trajectory
    plt.plot(times, trajectories[0, 0, :].to("cpu"))
    plt.show()
    # visualize a few histograms
    downsample_rate = num_times // 10
    fast_times = times[::downsample_rate]
    fast_trajectories = trajectories[..., ::downsample_rate]
    for time_idx in range(len(fast_times)):
        time = fast_times[time_idx]
        plt.hist(fast_trajectories[0, :, time_idx].to("cpu"), density=True)
        # Plot a gaussian density for comparison
        num_pts_gaussian = 100
        gaussian_pts = torch.linspace(-3, 3, num_pts_gaussian)
        plt.plot(
            gaussian_pts,
            1
            / torch.sqrt(2 * torch.tensor(torch.pi))
            * torch.exp(-0.5 * gaussian_pts**2),
        )
        plt.title(f'empirical distribution at time {time.to("cpu")}')
        plt.show()


def gaussian_reverse():
    # Implement euler-maruyama scheme
    # Parameters: discretization points (and/or step size + scheme), max time T

    device = torch.device("mps")
    d = 1
    num_trials = 1000
    X0 = torch.randn((d, num_trials), device=device)

    # scheme: uniform discretization; specify number of points
    # backwards-in-time, but reparameterized
    # early stop
    T = 2
    num_times = 1000
    # early_stop_time = 1 / torch.sqrt(torch.tensor(num_times, device=device))
    early_stop_time = 0.1
    fwd_times = torch.linspace(
        early_stop_time, T, num_times
    )  # times for the forward process
    times = T - torch.flip(fwd_times, (0,))
    # breakpoint()

    # sample trajectories
    trajectories = torch.zeros((d, num_trials, num_times), device=device)
    trajectories[:, :, 0] = X0  # CHECK: broadcasts?
    for time_idx in range(1, num_times):
        # time_idx corresponds to iter i+1
        # the loop generates X_{i+1} from X_{i}
        # NOTE: backward process
        #  The law of the forward process at time t for this initialization is
        #   N(0, (1 - e^{-t}) I)
        #  So its score is
        #   x \mapsto -x / (1 - e^{-t})
        time = times[time_idx - 1]
        next_time = times[time_idx]
        delta = next_time - time
        current_X = trajectories[:, :, time_idx - 1]
        fwd_score = -current_X / (
            1 - torch.exp(-(T - time))
        )  # forward process score at the reversed time
        drift = 0.5 * current_X + fwd_score
        trajectories[:, :, time_idx] = (
            current_X
            + drift * delta
            + torch.sqrt(delta) * torch.randn((d, num_trials), device=device)
        )

    # eval
    # visualize a trajectory
    plt.plot(times, trajectories[0, 0, :].to("cpu"))
    plt.show()
    # visualize a few histograms
    # For downsampling, we want to be sure we always 'catch' the final time
    #  to do this without a bunch of flips, assume that the downsample rate divides
    #  num_times, and sample exactly out of phase
    # TODO: this would be easier if we did N+1 point sampling rather than the current...
    downsample_rate = num_times // 10
    fast_times = times[(downsample_rate - 1) :: downsample_rate]
    fast_trajectories = trajectories[..., (downsample_rate - 1) :: downsample_rate]
    for time_idx in range(len(fast_times)):
        time = fast_times[time_idx]
        plt.hist(fast_trajectories[0, :, time_idx].to("cpu"), density=True, bins=31)
        # Plot a gaussian density for comparison
        num_pts_gaussian = 100
        variance = 1 - torch.exp(-(T - time))
        gaussian_pts = torch.linspace(-3, 3, num_pts_gaussian)
        plt.plot(
            gaussian_pts,
            1
            / torch.sqrt(2 * torch.tensor(torch.pi) * variance)
            * torch.exp(-0.5 * gaussian_pts**2 / variance),
        )
        plt.title(
            f'empirical distribution at time {time.to("cpu"):.2f} (fwd time {(T-time).to("cpu"):.2f})'
        )
        plt.show()


if __name__ == "__main__":
    # tyro.cli(gaussian_forward)
    tyro.cli(gaussian_reverse)
