#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp, sqrt

import matplotlib.pyplot as plt
import torch
import tyro

from denoisers.empirical import OptimalEmpiricalDenoiserConstantEnergy
from samplers.sde import BasicOUSampler
from util.visualization import plot_histogram
from util.metrics import chamfer


def test_ou_1d():
    # etc
    device = torch.device("cpu")

    dims = range(10, 200, 10)
    num_samples = 1000

    results = torch.zeros((num_samples, len(dims)), device=device)

    for idx_dim in range(len(dims)):
        # target empirical measure
        dim = dims[idx_dim]
        X = torch.zeros((2, dim), device=device)
        X[0, :dim//2] += 1.0
        X[1, dim//2:] += 1.0
        # create score function
        denoiser = OptimalEmpiricalDenoiserConstantEnergy(X)
        score = (
            lambda y, forward_time: (
                1.0
                / (1 - exp(-2 * forward_time))
                * (
                    exp(-forward_time)
                    * denoiser(exp(-forward_time) * y, 1 - exp(-2 * forward_time))
                    - y
                )
            )
        )  # TODO: seems not ideal that score function depends so much on the process (but maybe unavoidable? possibly interface should be different...)
        # sampler
        min_time = 1e-2
        num_points = 1000
        sampler = BasicOUSampler(dim, score, min_time, num_points)
        samples = sampler.sample(num_samples)

        # Test the results by visualizing (1D case)
        variance = 1 - exp(-2 * min_time)
        # print(variance)
        num_pts_gaussian = 100
        gaussian_pts = torch.linspace(-2, 2, num_pts_gaussian)
        plt.plot(
            gaussian_pts,
            0.5
            * (
                1
                / torch.sqrt(2 * torch.tensor(torch.pi) * variance)
                * torch.exp(-0.5 * (gaussian_pts - 1) ** 2 / variance)
                + 1
                / torch.sqrt(2 * torch.tensor(torch.pi) * variance)
                * torch.exp(-0.5 * (gaussian_pts + 1) ** 2 / variance)
            ),
        )
        # plot_histogram(samples)
        # print(samples)
        distances = chamfer(X, samples)
        results[:, idx_dim] = distances
        print(f'done dim {dim}')
        # breakpoint()
    breakpoint()



if __name__ == "__main__":
    tyro.cli(test_ou_1d)
