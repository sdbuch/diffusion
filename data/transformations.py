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
from torchvision.transforms.v2 import Lambda


# Add gaussian noise to the input
# Wrap this in a function to allow setting seed, etc
def make_gaussian_noise_transform(seed: int | None = None):
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise_lambda_func = Lambda(lambda tensor: tensor + torch.randn(tensor.shape, generator=generator))
    else:
        noise_lambda_func = Lambda(lambda tensor: tensor + torch.randn(tensor.shape))
    return noise_lambda_func
