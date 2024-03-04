#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
from typing import Literal

from types_custom import OptimizerType


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    algorithm: OptimizerType = OptimizerType.ADAM
    """Gradient-based optimizer to use."""

    learning_rate: float = 3e-4
    """Learning rate to use."""

    weight_decay: float = 1e-2
    """Coefficient for squared L2 regularization."""


@dataclasses.dataclass(frozen=True)
class AffineSelfAdjointDenoiserConfig:
    hidden_dimension: int = 256
    """Size of hidden layer for denoiser. (model is UU^T-type)"""

    initialization_std: float = 1e-3
    """Standard deviation of Gaussian for initialization of denoiser matrix. (one U factor)"""


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    device_str: Literal["cpu", "cuda"] = "cuda"
    """Where to train the model."""

    batch_size: int | None = 32
    """Batch size to use for training. (None means full batch)"""

    num_epochs: int = 100
    """Number of epochs to use for training."""

    noise_level: float = 1.0
    """Gaussian noise level (standard deviation) to train the denoiser at."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """Configuration for optimizer to use for training."""

    # TODO: abstract this better (enum)
    model: AffineSelfAdjointDenoiserConfig = AffineSelfAdjointDenoiserConfig()
    """Configuration for the denoiser to train."""

    seed: int | None = None
    """Random seed to use for reproducibility. If None, a random seed is used."""
