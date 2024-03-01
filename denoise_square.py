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

import wandb
from configs import ExperimentConfig, LinearSelfAdjointDenoiserConfig
from data.square import SquareDataset, TranslatedSquareDataset
from data.util import convert_rgb_tensor_to_pil, data_gram
from denoisers.empirical import OptimalEmpiricalDenoiserConstantEnergy
from denoisers.linear import LinearSelfAdjointDenoiser, OptimalLinearDenoiser
from types_custom import OptimizerType


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
    run = wandb.init(
        project="denoise-squares-linear",
        config={"dimension": dimension, "dataset_size": dataset_size}
        | dataclasses.asdict(config),
    )

    # TODO: support different datasets at config level
    if config.batch_size is None:
        # full-batch operation
        batch_size = dimension**2 * dataset_size
    else:
        batch_size = config.batch_size
    train_dataset = TranslatedSquareDataset(
        dimension=dimension, dataset_size=dataset_size, device=device
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_dataset = TranslatedSquareDataset(
        dimension=dimension, dataset_size=1, device=device
    )
    test_dataloader = DataLoader(test_dataset, batch_size=dimension**2)
    val_dataset = TranslatedSquareDataset(
        dimension=dimension, dataset_size=1, device=device
    )
    val_dataloader = DataLoader(test_dataset, batch_size=dimension**2)

    # Configure the models!
    # TODO: parameter init with our generator
    denoiser = LinearSelfAdjointDenoiser(
        dimension,
        config.model.hidden_dimension,
        # config.model.initialization_std
        1 / config.model.hidden_dimension**0.5 / dimension,
    )
    # For val, get the optimal denoiser for this dataset
    val_data = next(iter(val_dataloader))[0].clone().detach()
    bayes = OptimalEmpiricalDenoiserConstantEnergy(val_data, config.noise_level)
    opt_lin = OptimalLinearDenoiser(val_data, config.noise_level)
    # Move to gpu
    denoiser = denoiser.to(device)
    wandb.watch(denoiser, log_freq=config.num_epochs // 10)
    bayes = bayes.to(device)
    opt_lin = opt_lin.to(device)

    # # Configure the optimizer!
    optimizer = config.optimizer.algorithm.value(
        denoiser.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    # Training loop involves 'online sgd': adding indep. random noise to each minibatch
    # loss_fn = nn.MSELoss(reduction = 'sum') # it takes a mean on all dims by default
    loss_fn = nn.MSELoss()
    denoiser.train()
    for epoch in tqdm(range(config.num_epochs), desc="Epoch", position=0):
        for batch, (X,) in enumerate(
            tqdm(train_dataloader, desc="Batch", position=1, leave=False)
        ):
            noisy_X = X + config.noise_level * torch.randn(
                X.shape, device=device, generator=generator
            )
            denoised = denoiser(noisy_X)
            loss = loss_fn(X, denoised)  # / batch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"batch MSE": loss.item()})
            with torch.no_grad():
                opt_loss = loss_fn(X, opt_lin(noisy_X))
            wandb.log(
                {"batch excess MSE (MSE - MSE_OPT)": loss.item() - opt_loss.item()}
            )
            # print(loss.to("cpu"))
    # with torch.no_grad():
    #     for (X_val,) in val_dataloader:
    #         # Cheap evaluation of optimal denoiser's performance
    #         noisy_X_val = X_val + config.noise_level * torch.randn(
    #                 X_val.shape, device=device, generator=generator
    #                 )
    #         denoised_val = bayes(noisy_X_val)
    #         loss_val = loss_fn(noisy_X_val, denoised_val)
    #         wandb.log({"bayes error (est.)": loss_val.item()})

    # "examine" the test performance
    denoiser.eval()
    with torch.no_grad():
        for (X,) in test_dataloader:
            noisy_X = X + config.noise_level * torch.randn(
                X.shape, device=device, generator=generator
            )
            denoised = denoiser(noisy_X)
            opt_denoised = bayes(noisy_X)
        denoised_pil = convert_denoised_tensor(denoised, dimension)
        opt_denoised_pil = convert_denoised_tensor(opt_denoised, dimension)
    wandb.log({"opt denoising error": torch.mean((opt_denoised - X) ** 2)})
    wandb.log({"noisy test data (tiled)": [wandb.Image(noisy_X)]})
    wandb.log({"denoised test data (tiled)": [wandb.Image(denoised_pil)]})
    wandb.log({"optimal denoised test data (tiled)": [wandb.Image(opt_denoised_pil)]})
    # wandb.log({"denoised test data": [wandb.Image(denoised)]}) # wandb tiles them for you, conversion is correct (rescale not clamp)


def convert_denoised_tensor(denoised: torch.Tensor, dimension: int):
    bottom = denoised.min()
    top = denoised.max()
    denoised_pad = torch.nn.functional.pad(
        denoised, (1, 1, 1, 1), value=float(top - bottom)
    )
    denoised_tiled = rearrange(
        denoised_pad, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=dimension
    )
    return convert_rgb_tensor_to_pil(denoised_tiled)


if __name__ == "__main__":
    tyro.cli(denoise_square)