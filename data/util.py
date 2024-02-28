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
from einops import rearrange, reduce
from torchvision.transforms.v2.functional import to_pil_image


def data_gram(
    data: torch.Tensor,  # first dimension: batch
) -> torch.Tensor:
    """
    Compute gram matrix for a (b x c x h x w) image data tensor. Return unflattened.

    :param data: Image data tensor (b x c x h x w)
    :return: Unflattened Gram tensor (c x h x w x c x h x w)
    """
    return reduce(
        rearrange(data, "b c h w -> b c h w 1 1 1")
        * rearrange(data, "b c h w -> b 1 1 1 c h w"),
        "b c h w cc hh ww -> c h w cc hh ww",
        "mean",
    )

def convert_rgb_tensor_to_pil(image_tensor):
    # expect image_tensor to be 3 x H x W
    # rescale image content
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    rescaled = (image_tensor - min_val) / (max_val - min_val)
    to_pil = to_pil_image(rescaled)

    return to_pil
