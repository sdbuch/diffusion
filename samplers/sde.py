#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from collections.abc import Callable, Iterator
from math import exp, sqrt
from time import sleep
from typing import Iterable, Tuple

import torch
from torch import Tensor
from util.types_custom import FloatTensor
from util.visualization import plot_histogram

from samplers.discretizations import ExpoLinearDiscretization, LinearDiscretization
from samplers.integrators import euler_maruyama

""" This module contains classes for sampling from stochastic differential equations (SDEs).
We employ the following time conventions:
    1. times are torch.Tensors (see util.types_custom)
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
    dimension: Tuple[int, ...] | torch.Size  # dimension of ambient space
    drift: Callable[[Tensor, FloatTensor], Tensor]  # (x, t) -> drift
    diffusion_coeff: FloatTensor  # only support scalars!
    max_time: FloatTensor
    min_time: FloatTensor
    discretization: Callable[[], Iterator[FloatTensor]]
    integrator: Callable[
        [Tensor, FloatTensor, FloatTensor], tuple[Tensor, FloatTensor]
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
        self, current_x: Tensor, current_t: FloatTensor, next_t: FloatTensor
    ) -> Tensor:
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

    def sample(self, num_samples: int) -> Tensor:
        # full iteration of the sampler
        # TODO: Expose initial distribution
        initialization = torch.randn(
            (num_samples,) + self.dimension,
            generator=self.generator,
            device=self.device,
        )
        current_x = initialization
        discretization = self.discretization()
        current_t = discretization.__next__()
        for next_t in discretization:
            next_x = self.step(current_x, current_t, next_t)
            current_x = next_x
            current_t = next_t
        return current_x

    def sample_trajectory(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        # full iteration + log every intermediate
        initialization = torch.randn(
            (num_samples,) + self.dimension,
            generator=self.generator,
            device=self.device,
        )
        log = []
        current_x = initialization
        discretization = self.discretization()
        current_t = discretization.__next__()
        log.append((current_x, current_t))
        for next_t in discretization:
            next_x = self.step(current_x, current_t, next_t)
            current_x = next_x
            current_t = next_t
            log.append((current_x, current_t))
        snapshot, times = zip(*log)
        return torch.stack(snapshot), torch.stack(times)


class BasicOUSampler(SDESampler):
    """A simple Ornstein-Uhlenbeck process sampler. dx_t = -x_t dt + sqrt(2) dw_t."""

    def __init__(
        self,
        dimension: Tuple[int, ...] | torch.Size,
        score_estimator: Callable[[Tensor, FloatTensor], Tensor],  # same as drift
        min_time: FloatTensor,
        num_points: int,
        device: torch.device | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        assert all([dim > 0 for dim in dimension])

        diffusion_coeff = torch.tensor(sqrt(2.0))

        # time discretization is 'flipped' because this is the reverse process
        # we want it to correspond to starting (forward time) at min_time and stopping at max_time
        # discretization = lambda: LinearDiscretization(0.0, max_time - min_time, num_points)
        discretization = lambda: ExpoLinearDiscretization(num_points, min_time)
        max_time = discretization().max_time

        # construct drift from score_estimator
        def drift(x: Tensor, t: FloatTensor) -> Tensor:
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

    def sample(self, num_samples: int, debias: bool = False) -> Tensor:
        samples = super().sample(num_samples)
        if debias == True:
            samples *= torch.exp(self.min_time)
        return samples

    def sample_trajectory(
        self, num_samples: int, debias: bool = False
    ) -> Tuple[Tensor, Tensor]:
        samples, times = super().sample_trajectory(num_samples)
        if debias == True:
            samples *= torch.exp(self.min_time)
        return samples, self.max_time - times
