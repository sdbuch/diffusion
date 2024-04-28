#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Dict, Sized, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from util.types_custom import SizedDatasetType

# Wrapper datasets. For 2D image manifolds.


class TranslationalImageManifoldify(Dataset, Sized):
    def __init__(
        self,
        base_dataset: SizedDatasetType,
        device: torch.device,
        downsample_factor: int = 1,
        data_index: int = 0,  # data[i] is a tuple; which idx is the sample?
    ):
        super().__init__()
        self.data = base_dataset
        self.data_index = data_index
        self.data_shape = self.data[0][data_index].shape
        self.device = device
        # TODO: allow this to be a tuple (for different dims)
        self.downsample_factor = downsample_factor
        self._load()

    def __getitem__(self, index: int):
        shifts = []
        for dim in self.data_shape[1:]:
            shifts.append(
                self.downsample_factor * (index % (dim // self.downsample_factor))
            )
            index //= dim // self.downsample_factor
        data_point = self.data[self.data_index][index]
        shifted_data_point = torch.roll(
            data_point,
            shifts=(0,) + tuple(shifts),
            dims=tuple(range(len(self.data_shape))),
        )
        output = [self.data[i][index] for i in range(len(self.data))]
        output[self.data_index] = shifted_data_point

        return tuple(output)

    def __len__(self) -> int:
        copy_factor = 1
        for dim in self.data_shape[1:]:
            copy_factor *= dim // self.downsample_factor
        return len(self.data) * copy_factor

    def _load(self) -> None:
        # Check the dataset for consistent size
        for data_point in self.data:
            assert data_point[self.data_index].shape == self.data_shape
        # Check that the downsample_factor is valid
        # It needs to divide each dimension of the data
        # We skip the first dimension (channels)
        for dim in self.data_shape[1:]:
            assert dim % self.downsample_factor == 0
