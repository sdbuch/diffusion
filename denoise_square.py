#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
from collections.abc import Callable
from pathlib import Path
from time import sleep
from typing import Dict, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
import torchvision.transforms.v2.functional as F
import tyro
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from configs import ExperimentConfig, LinearSelfAdjointDenoiserConfig
from data.square import SquareDataset, TranslatedSquareDataset
from data.util import convert_rgb_tensor_to_pil, data_gram
from denoisers.linear import LinearSelfAdjointDenoiser
from types_custom import OptimizerType

# def show_square(dimension: int, dataset_size: int, batch_size: int):
#     dm = SquareDataModule(
#         dimension=dimension, dataset_size=dataset_size, batch_size=batch_size
#     )
#     dm.setup()
#
#     for batch in dm.train_dataloader():
#         # Batch is a list of length 1 (no labels)
#         # Entry is a tensor of shape B x 3 x dimension x dimension
#         img = batch[0][0]
#         plt.imshow(F.to_pil_image(img))
#         plt.show()
#         breakpoint()


def denoise_square(
    dimension: int = 12,
    dataset_size: int = 128,
    config: ExperimentConfig = ExperimentConfig(),
):
    """
    Train a denoiser for the fixed-position-square dataset.

    :param dimension: Side length of images
    :param dataset_size: A blowup factor for the dataset. Used for vectorization.
    :param config: Configuration for the experiment.
    """
    device = torch.device(config.device_str)
    generator = torch.Generator(device=device)
    if config.seed is not None:
        generator.manual_seed(config.seed)
    else:
        generator.seed()

    # TODO: support different datasets at config level
    if config.batch_size is None:
        # full-batch operation
        batch_size = dimension**2 * dataset_size
    else:
        batch_size = config.batch_size
    train_dataset = TranslatedSquareDataset(
        dimension=dimension, dataset_size=dataset_size, device=device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True
    )
    test_dataset = TranslatedSquareDataset(
        dimension=dimension, dataset_size=1, device=device
    )
    test_dataloader = DataLoader(test_dataset, batch_size=dimension**2)

    # Configure the model!
    # TODO: parameter init with our generator
    denoiser = LinearSelfAdjointDenoiser(
        dimension,
        config.model.hidden_dimension,
        # config.model.initialization_std
        1 / config.model.hidden_dimension**0.5 / dimension,
    )
    # Move to gpu
    denoiser = denoiser.to(device)

    # # Configure the optimizer!
    optimizer = config.optimizer.algorithm.value(
        denoiser.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    # Training loop involves 'online sgd': adding indep. random noise to each minibatch
    loss_fn = nn.MSELoss()
    denoiser.train()
    for epoch in tqdm(range(config.num_epochs), desc="Epoch", position=0):
        for batch, (X,) in enumerate(tqdm(train_dataloader, desc="Batch", position=1, leave=False)):
            noisy_X = X + config.noise_level * torch.randn(
                X.shape, device=device, generator=generator
            )
            denoised = denoiser(noisy_X)
            loss = loss_fn(noisy_X, denoised)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(loss.to("cpu"))

    # TODO: log the losses
    # TODO: calculate bayes error

    # examine the test performance
    denoiser.eval()
    with torch.no_grad():
        for (X,) in test_dataloader:
            noisy_X = X + config.noise_level * torch.randn(
                X.shape, device=device, generator=generator
            )
            denoised = denoiser(noisy_X)
    bottom = denoised.min()
    denoised_pad = torch.nn.functional.pad(denoised, (1, 1, 1, 1), value=bottom)
    denoised_tiled = rearrange(
        denoised_pad, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=dimension
    )
    denoised_pil = convert_rgb_tensor_to_pil(denoised_tiled)
    plt.imshow(denoised_pil)
    plt.show()


if __name__ == "__main__":
    tyro.cli(denoise_square)
