#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp, sqrt, log

import matplotlib.pyplot as plt
import torch
import tyro
from numpy.random import default_rng
from tqdm import tqdm

import wandb
from data.motions import TranslationalImageManifoldify
from data.square import StepDataset
from denoisers.empirical import OptimalEmpiricalDenoiserConstantEnergy
from samplers.sde import BasicOUSampler
from util.metrics import chamfer
from util.visualization import plot_histogram


# TODO: write a sweep?
def test_ou():
    # etc
    device = torch.device("cuda")

    # Experiment params
    dims = range(10, 1000, 10)
    num_samples = 10000
    downsample_factor = 0  # HACK: setting to 0 means no downsampling

    # sampler params
    base_min_time = 1.0
    base_num_points = 10
    debias = False

    run = wandb.init(
        project="pixel-effect-on-wass-incoherent-mixture",
        config={
            "dimensions": list(dims),
            "num_samples": num_samples,
            "early_stopping_time": base_min_time,
            "discretization_length": base_num_points,
            "debias": debias,
            "downsample_factor": downsample_factor,
        },
        # | dataclasses.asdict(config),
    )
    wandb.define_metric("dimension")
    wandb.define_metric("distance", step_metric="dimension")
    wandb.define_metric("mean distance", step_metric="dimension")
    wandb.define_metric("normalized distance", step_metric="dimension")
    wandb.define_metric("uniformness", step_metric="dimension")

    rng = default_rng()

    # We try to estimate progress based on time complexity of the sampler
    # auxiliary progress bar...
    adjusted_total = sum(dims)

    with tqdm(total=adjusted_total) as pbar:
        for idx_dim in range(len(dims)):
            # target empirical measure
            dim = dims[idx_dim]
            if downsample_factor == 0:
                downsample = 1
            else:
                downsample = dim // downsample_factor
            data = TranslationalImageManifoldify(
                base_dataset=StepDataset(dimension=dim, device=device),
                device=device,
                downsample_factor=downsample,
            )
            X = [sample[0][None, ...] for sample in data]
            X = torch.cat(X)
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
            seed = int(rng.integers(0, high=2**32))
            sampler = BasicOUSampler(
                dimension=X.shape[1:],
                score_estimator=score,
                min_time=base_min_time / dim**2,
                num_points=100,#2*int(base_num_points * log(dim)),
                device=device,
                seed=seed,
            )
            samples = sampler.sample(num_samples, debias=debias)

            # calculate ell^2 distances
            # It corresponds to an upper bound on W_1 for ell^2
            # Hence the normalization factor is 1/sqrt(dim)
            distances, minimizer_args = chamfer(X, samples)
            histogram = torch.tensor(
                [minimizer_args.tolist().count(i) for i in range(dim // downsample)],
                dtype=torch.float32,
            )
            uniformness = torch.exp(torch.mean(torch.log(histogram))) / torch.mean(
                histogram
            )

            wandb.log(
                {
                    "dimension": dim,
                    "distance": distances,
                    "mean distance": torch.mean(distances),
                    "uniformness": uniformness,
                    "normalized distance": distances / sqrt(dim),
                }
            )
            # TODO: anything else to log?
            pbar.update(dim)


if __name__ == "__main__":
    tyro.cli(test_ou)
