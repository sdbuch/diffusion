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
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from configs import ExperimentConfig, LinearSelfAdjointDenoiserConfig
from data.square import SquareDataset, TranslatedSquareDataset
from data.util import data_gram, convert_rgb_tensor_to_pil
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
    config_experiment: ExperimentConfig = ExperimentConfig(),
    config_model: LinearSelfAdjointDenoiserConfig = LinearSelfAdjointDenoiserConfig(),
):
    """
    Train a denoiser for the fixed-position-square dataset.

    :param dimension: Side length of images
    :param dataset_size: A blowup factor for the dataset. Used for vectorization.
    :param config: Configuration for the experiment.
    """
    device = torch.device(config_experiment.device_str)
    generator = torch.Generator(device=device)
    if config_experiment.seed is not None:
        generator.manual_seed(config_experiment.seed)
    else:
        generator.seed()

    # TODO: support different datasets at config level
    train_dataset = TranslatedSquareDataset(
        dimension=dimension, dataset_size=dataset_size, device=device
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config_experiment.batch_size, drop_last=True
    )

    # Configure the model!
    # TODO: parameter init with our generator
    denoiser = LinearSelfAdjointDenoiser(
        dimension,
        config_model.hidden_dimension,
        # config_model.initialization_std
        1 / config_model.hidden_dimension**0.5 / dimension,
    )
    # Move to gpu
    denoiser = denoiser.to(device)

    # Configure the optimizer!
    if config_experiment.optimizer.algorithm == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(
            denoiser.parameters(),
            lr=config_experiment.optimizer.learning_rate,
            weight_decay=config_experiment.optimizer.weight_decay,
        )
    elif config_experiment.optimizer.algorithm == OptimizerType.SGD:
        optimizer = torch.optim.SGD(
            denoiser.parameters(),
            lr=config_experiment.optimizer.learning_rate,
            weight_decay=config_experiment.optimizer.weight_decay,
        )

    # Training loop involves 'online sgd': adding indep. random noise to each minibatch
    loss_fn = nn.MSELoss()
    denoiser.train()
    for epoch in tqdm(range(config_experiment.num_epochs), desc="Epoch"):
        for batch, (X,) in enumerate(tqdm(train_dataloader, desc="Batch")):
            noisy_X = X + config_experiment.noise_level * torch.randn(
                X.shape, device=device, generator=generator
            )
            denoised = denoiser(noisy_X)
            loss = loss_fn(noisy_X, denoised)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.to('cpu'))

    # TODO: log the losses
    # TODO: calculate bayes error
    # TODO: examine the test performance
    batch, (X,) = next(enumerate(train_dataloader))
    noisy_X = X + config_experiment.noise_level * torch.randn(
        X.shape, device=device, generator=generator
    )
    denoised = denoiser(noisy_X)
    # convert_rgb_tensor_to_pil useful here ...


    breakpoint()


if __name__ == "__main__":
    tyro.cli(denoise_square)
