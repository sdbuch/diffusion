#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import enum
import torch

# This implementation relies on the optimizers having a common interface.
# Might not always hold?
class OptimizerType(enum.Enum):
    ADAM = torch.optim.Adam # enum.auto()
    SGD = torch.optim.SGD # enum.auto()
