#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from collections.abc import Callable
from typing import Dict, Tuple

import lightning as L
import torch
from numpy import unravel_index
from torch.utils.data import DataLoader, Dataset


class SquareDataset(Dataset):
    def __init__(
        self,
        # root: Path,
        dimension: int,  # side length of images, squares are half-width
        dataset_size: int,  # dataset contains a bunch of identical squares
        device: torch.device,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data: Dict[str, torch.Tensor] = {}
        self.device = device
        self._load(dimension, dataset_size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        data = self.data[str(index)]
        if self.transform is not None:
            data = self.transform(data)
        return (data,)

    def __len__(self) -> int:
        return len(self.data)

    def _load(self, dimension, dataset_size) -> None:
        # Make the red square
        step = torch.zeros((dimension, 1), device=self.device)
        step[: dimension // 2, 0] = 1
        square = (step @ step.T)[None, ...]
        red_square = torch.cat(
            (square, torch.zeros((2, dimension, dimension), device=self.device))
        )
        for i in range(dataset_size):
            self.data[str(i)] = red_square.clone()


# TODO: maybe can write this as a generic wrapper for a set of images...
class TranslatedSquareDataset(Dataset):
    def __init__(
        self,
        # root: Path,
        dimension: int,  # side length of images, squares are half-width
        dataset_size: int,  # copy each element of the dataset this many times
        device: torch.device,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.device = device
        self.dimension = dimension
        self.copy_factor = dataset_size
        self._load(dimension, dataset_size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        i, j, k = unravel_index(
            index, (self.dimension, self.dimension, self.copy_factor)
        )
        data = self.data[(int(i), int(j), int(k))]
        if self.transform is not None:
            data = self.transform(data)
        return (data,)

    def __len__(self) -> int:
        return len(self.data)

    def _load(self, dimension, dataset_size) -> None:
        # Make the red square
        # it's 3 x dimension x dimension
        step = torch.zeros((dimension, 1), device=self.device)
        step[: dimension // 2, 0] = 1
        square = (step @ step.T)[None, ...]
        red_square = torch.cat(
            (square, torch.zeros((2, dimension, dimension), device=self.device))
        )
        # Loop over translates and clone
        for i in range(dimension):
            for j in range(dimension):
                for k in range(dataset_size):
                    translated_square = torch.roll(
                        red_square, shifts=(0, i, j), dims=(0, 1, 2)
                    )
                    self.data[(i, j, k)] = translated_square.clone()


# class SquareDataModule(L.LightningDataModule):
#     def __init__(
#         self,
#         # root: Path,
#         transform: Callable | None = None,
#         dimension: int = 64,
#         dataset_size: int = 1,
#         batch_size: int = 32,
#         num_workers: int = 0,
#         pin_memory: bool = True,
#     ) -> None:
#         super().__init__()
#         self.transform = transform
#         self.dimension = dimension
#         self.dataset_size = dataset_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#
#     def prepare_data(self) -> None:
#         pass
#
#     def setup(self, stage: str = "") -> None:
#         self.train_dataset = SquareDataset(
#             transform=self.transform,
#             dimension=self.dimension,
#             dataset_size=self.dataset_size,
#         )
#         self.val_dataset = SquareDataset(
#             transform=self.transform,
#             dimension=self.dimension,
#             dataset_size=self.dataset_size,
#         )
#         self.test_dataset = SquareDataset(
#             transform=self.transform,
#             dimension=self.dimension,
#             dataset_size=self.dataset_size,
#         )
#
#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=True,
#         )
#
#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#         )
#
#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#         )
