#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp, sqrt

import matplotlib.pyplot as plt
import torch
import tyro
from tqdm import tqdm

import wandb
from data.square import SquareDataset, TranslatedSquareDataset
from denoisers.empirical import OptimalEmpiricalDenoiserConstantEnergy
from samplers.sde import BasicOUSampler
from util.metrics import chamfer
from util.visualization import plot_histogram


def test_ou():
    # etc
    device = torch.device("cuda")

    # Experiment params
    dims = range(10, 1000, 10)
    num_samples = 1000
    data = TranslatedSquareDataset(
            dimension=10,
            dataset_size=1,
            device=device)
    tmp = []
    for idx in range(len(data)):
        sample = data[idx]
        tmp.append(sample[0])

    # sampler params
    min_time = 1e-2
    num_points = 1000

    run = wandb.init(
        project="pixel-effect-on-wass-incoherent-mixture",
        config={
            "dimensions": list(dims),
            "num_samples": num_samples,
            "early_stopping_time": min_time,
            "discretization_length": num_points,
        },
        # | dataclasses.asdict(config),
    )
    wandb.define_metric("dimension")
    wandb.define_metric("distance", step_metric="dimension")
    wandb.define_metric("mean distance", step_metric="dimension")
    wandb.define_metric("normalized distance", step_metric="dimension")

    # We try to estimate progress based on time complexity of the sampler
    # auxiliary progress bar...
    adjusted_total = sum(dims)

    with tqdm(total=adjusted_total) as pbar:
        for idx_dim in range(len(dims)):
            # target empirical measure
            dim = dims[idx_dim]
            X = torch.zeros((2, dim), device=device)
            X[0, : dim // 2] += 1.0
            X[1, dim // 2 :] += 1.0
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
            sampler = BasicOUSampler(dim, score, min_time, num_points, device=device)
            samples = sampler.sample(num_samples)

            # calculate ell^2 distances
            # It corresponds to an upper bound on W_1 for ell^2
            # Hence the normalization factor is 1/sqrt(dim)
            distances = chamfer(X, samples)

            wandb.log(
                {
                    "dimension": dim,
                    "distance": distances,
                    "mean distance": torch.mean(distances),
                    "normalized distance": distances / sqrt(dim)
                }
            )
            # TODO: anything else to log?
            pbar.update(dim)


if __name__ == "__main__":
    tyro.cli(test_ou)
