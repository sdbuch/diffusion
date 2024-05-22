#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import enum
from typing import Protocol, Sized

import torch
from torch.utils.data import Dataset


# This implementation relies on the optimizers having a common interface.
# Might not always hold?
class OptimizerType(enum.Enum):
    ADAM = torch.optim.Adam  # enum.auto()
    SGD = torch.optim.SGD  # enum.auto()


# Pytorch Datasets (as of 2.3) require __getitem__ but __len__ is optional.
# We will implement __len__, so make a custom type for this.
class SizedDatasetType(Dataset, Sized):
    pass
