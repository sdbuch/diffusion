#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

# Affine denoisers.


class AffineSelfAdjointDenoiser(nn.Module):
    def __init__(
        self,
        input_hw: int,
        hidden_dimension: int,
        initialization_std: float,
    ):
        super().__init__()
        self.input_dim = 3 * input_hw**2
        self.hidden_dim = hidden_dimension
        self.flatten = nn.Flatten()
        self.projection = nn.Parameter(
            initialization_std * torch.randn(self.hidden_dim, self.input_dim),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.zeros(self.input_dim),
            requires_grad=True,
        )
        self.unflatten = nn.Unflatten(-1, (3, input_hw, input_hw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = self.flatten(x)
        # Project the input
        x = F.linear(x, self.projection, bias=-self.projection @ self.bias)
        # Lift input
        x = F.linear(x, self.projection.T, bias=self.bias)
        x = self.unflatten(x)
        return x

    def get_mapping(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the affine transformation corresponding to the denoiser
        (flattened)

        I.e.,
            y |-> A @ y + b
        and we return (A, b)

        :return: 2 element tuple of
            (input_dim x input_dim denoiser matrix (symmetric),
            input_dim bias vector)
        """
        gram = self.projection.T @ self.projection
        return (gram, torch.eye(self.input_dim, device=self.projection.device) - gram)


# PERF: can we ensure the __init__ computations are done on gpu...
class OptimalAffineDenoiser(nn.Module):
    def __init__(
        self,
        dataset: torch.Tensor,  # format as n x ... (will be flattened)
        noise_std: float,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_shape = dataset.shape[1:]
        self.unflatten = nn.Unflatten(-1, self.input_shape)
        self.noise_variance = noise_std**2
        self.mean = torch.mean(self.flatten(dataset), dim=0)
        self.covariance = (
            (self.flatten(dataset) - self.mean).T
            @ (self.flatten(dataset) - self.mean)
            / dataset.shape[0]
        )
        self.induced_linear = self.covariance @ torch.linalg.inv(
            self.covariance
            + self.noise_variance
            * torch.eye(math.prod(self.input_shape), device=dataset.device)
        )
        self.induced_bias = (
            torch.eye(math.prod(self.input_shape), device=dataset.device)
            - self.induced_linear
        ) @ self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.flatten(x)
            x = F.linear(x, self.induced_linear, bias=self.induced_bias)
            x = self.unflatten(x)
        return x

    def get_mapping(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the affine transformation corresponding to the denoiser
        (flattened)

        I.e.,
            y |-> A @ y + b
        and we return (A, b)

        :return: 2 element tuple of
            (input_dim x input_dim denoiser matrix (symmetric),
            input_dim bias vector)
        """
        return (self.induced_linear, self.induced_bias)
