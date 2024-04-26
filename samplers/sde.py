#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from collections.abc import Callable, Iterator
from math import exp, sqrt
from time import sleep
from typing import Iterable

import torch
from util.visualization import plot_histogram

from samplers.discretizations import (ExpoLinearDiscretization,
                                      LinearDiscretization)
from samplers.integrators import euler_maruyama

""" This module contains classes for sampling from stochastic differential equations (SDEs).
We employ the following time conventions:
    1. times are floats
    2. score estimators/denoisers have time arguments corresponding to the time of the forward process. So an argument of t=0.0 corresponds to the data distribution
    3. time arguments for sampling processes (min_time and max_time) are defined with respect to the forward process. So min_time larger than 0 corresponds to an "early stopping". max_time is the final time of the forward process.
"""


def init_generator(seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@dataclasses.dataclass(frozen=True)
class SDESampler:
    """Base class for SDE-based samplers."""

    # Process configuration
    dimension: int  # dimension of ambient space TODO: allow to be size?
    drift: Callable[[torch.Tensor, float], torch.Tensor]  # (x, t) -> drift
    diffusion_coeff: float  # only support scalars!
    max_time: float
    min_time: float
    discretization: Callable[[], Iterator[float]]
    integrator: Callable[
        [torch.Tensor, float, float], tuple[torch.Tensor, float]
    ]  # (x, t, dt) -> x_next

    # Randomness configuration
    device: torch.device = torch.device("cpu")
    seed: int = 0
    generator: torch.Generator | None = None

    def __post_init__(self):
        if self.generator is None:
            # post-init hack to initialize
            object.__setattr__(
                self, "generator", init_generator(self.seed, self.device)
            )

    def step(
        self, current_x: torch.Tensor, current_t: float, next_t: float
    ) -> torch.Tensor:
        # one round of the sampler
        current_drift = self.drift(current_x, current_t)
        integrated_drift, integrated_diffu = self.integrator(
            current_drift, current_t, next_t
        )
        data_shape = current_x.shape
        noise = torch.randn(data_shape, generator=self.generator, device=self.device)
        return (
            current_x
            + integrated_drift
            + self.diffusion_coeff * integrated_diffu * noise
        )

    def sample(self, num_samples: int) -> torch.Tensor:
        # full iteration of the sampler
        # TODO: Expose initial distribution
        initialization = torch.randn(
            (num_samples, self.dimension), generator=self.generator, device=self.device
        )
        current_x = initialization
        discretization = self.discretization()
        current_t = discretization.__next__()
        for next_t in discretization:
            next_x = self.step(current_x, current_t, next_t)
            current_x = next_x
            current_t = next_t
        return current_x

    def sample_trajectory(self):
        # full iteration + log every intermediate
        raise NotImplementedError("not implemented yet")


class BasicOUSampler(SDESampler):
    """A simple Ornstein-Uhlenbeck process sampler. dx_t = -x_t dt + sqrt(2) dw_t."""

    def __init__(
        self,
        dimension: int,
        score_estimator: Callable[[torch.Tensor, float], torch.Tensor],  # same as drift
        min_time: float,
        num_points: int,
        device: torch.device | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        assert dimension > 0

        diffusion_coeff = sqrt(2.0)

        # time discretization is 'flipped' because this is the reverse process
        # we want it to correspond to starting (forward time) at min_time and stopping at max_time
        # discretization = lambda: LinearDiscretization(0.0, max_time - min_time, num_points)
        discretization = lambda: ExpoLinearDiscretization(num_points, min_time)
        max_time = discretization().max_time

        # construct drift from score_estimator
        def drift(x: torch.Tensor, t: float) -> torch.Tensor:
            assert max_time - t > 0
            return x + 2 * score_estimator(x, max_time - t)

        default_dict = {}
        if device is not None:
            default_dict["device"] = device
        if seed is not None:
            default_dict["seed"] = seed
        if generator is not None:
            default_dict["generator"] = generator

        # Super
        super().__init__(
            dimension=dimension,
            drift=drift,
            diffusion_coeff=diffusion_coeff,
            max_time=max_time,
            min_time=min_time,
            discretization=discretization,
            integrator=euler_maruyama,
            **default_dict,
        )

    def sample(self, num_samples: int, debias: bool = False) -> torch.Tensor:
        samples = super().sample(num_samples)
        if debias == True:
            samples *= exp(self.min_time)
        return samples
