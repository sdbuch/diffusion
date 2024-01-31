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
import torchvision.transforms.v2.functional as F
import tyro
from torch.utils.data import DataLoader
from torch import nn

from data import SquareDataModule
from data.square import SquareDataset

from tqdm import tqdm
from time import sleep


def show_square(dimension: int, dataset_size: int, batch_size: int):
    dm = SquareDataModule(
        dimension=dimension, dataset_size=dataset_size, batch_size=batch_size
    )
    dm.setup()

    for batch in dm.train_dataloader():
        # Batch is a list of length 1 (no labels)
        # Entry is a tensor of shape B x 3 x dimension x dimension
        img = batch[0][0]
        plt.imshow(F.to_pil_image(img))
        plt.show()
        breakpoint()


def denoise_square(
    dimension: int = 24,
    dataset_size: int = 128,
    batch_size: int = 32,
    seed: int | None = None,
    device_str: Literal["cpu", "cuda"] = "cuda",
):
    device = torch.device(device_str)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    train_dataset = SquareDataset(
        dimension=dimension, dataset_size=dataset_size, device=device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True
    )

    # Training loop involves 'online sgd': adding indep. random noise to each minibatch
    loss = nn.MSELoss()
    for batch, (X,) in enumerate(tqdm(train_dataloader)):
        sleep(0.5)



if __name__ == "__main__":
    tyro.cli(denoise_square)
