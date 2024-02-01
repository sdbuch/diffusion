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
class ExperimentConfig:
    device_str: Literal["cpu", "cuda"] = "cuda"
    """Where to train the model."""

    batch_size: int = 32
    """Batch size to use for training."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """Configuration for optimizer to use for training."""

    seed: int | None = None
    """Random seed to use for reproducibility. If None, a random seed is used."""
