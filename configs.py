#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
from types import OptimizerType

@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    algorithm: OptimizerType = OptimizerType.ADAM
    """Gradient-based optimizer to use."""

    learning_rate: float = 3e-4
    """Learning rate to use."""

    weight_decay: float = 1e-2
    """Coefficient for L2 regularization."""
