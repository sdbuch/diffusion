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

# Linear denoisers.


class LinearSelfAdjointDenoiser(nn.Module):
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
        self.unflatten = nn.Unflatten(-1, (3, input_hw, input_hw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input
        x = self.flatten(x)
        # Project the input
        x = F.linear(x, self.projection)
        # Lift input
        x = F.linear(x, self.projection.T)
        x = self.unflatten(x)
        return x

    def get_mapping(self) -> torch.Tensor:
        """
        Return the matrix corresponding to the denoiser (flattened)

        :return: input_dim x input_dim denoiser matrix (symmetric)
        """
        return self.projection.T @ self.projection
