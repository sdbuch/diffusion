#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from collections.abc import Callable

import torch


def init_generator(seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def tmp():
        return generator

    return tmp


def init_device():
    return torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class SDESampler:
    """Base class for SDE-based samplers."""

    # Process configuration
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x, t) -> drift
    diffusion: float  # only support scalars!
    max_time: float
    min_time: float
    discretization: Callable[..., torch.Tensor]

    # Randomness configuration
    device: torch.device = dataclasses.field(default_factory=init_device)
    seed: int = 0
    generator: torch.Generator = dataclasses.field(
        default_factory=init_generator(seed, device)
    )

    # Seed
    # Drift function
    # diffusion coefficient
    # Time interval
    # Time reparameterization (can support this...)
    # discretization

    # for inheriter:
    # Score estimator (just call with this?)
    #

    def step():
        # one round of the sampler
        pass

    def sample():
        # full iteration of the sampler
        pass

    def sample_trajectory():
        # full iteration + log every intermediate 
        pass


