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
import torch.nn.functional as F
from torch import nn

# Denoisers based on finite sets.


class OptimalEmpiricalDenoiserConstantEnergy(nn.Module):
    """Denoiser for a finite set, uniform weights, all elements equal ell^2 norm."""

    def __init__(
        self,
        dataset: torch.Tensor, # format as n x ... (will be flattened)
        noise_std: float,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)
        self.input_shape = dataset.shape[1:]
        self.unflatten = nn.Unflatten(-1, self.input_shape)
        self.dataset = self.flatten(dataset)
        self.noise_variance = noise_std**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Flatten the input: output is b x dim
            x = self.flatten(x)
            # Compute correlations: output is b x n
            x = F.linear(x, self.dataset)
            # Calculate softmax weights with corrs: output is b x n
            x = self.softmax(x / self.noise_variance)
            # Combine dataset with softmax weights to generate output
            # output is b x dim
            x = F.linear(self.dataset.T, x).T
            # Reshape output to original shape: output is b x ...
            x = self.unflatten(x)
        return x
