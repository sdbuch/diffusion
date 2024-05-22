#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import einsum, rearrange, reduce, repeat
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.configs import OptimizerConfig

from .util import PreNorm

# Compression-based denoisers (MCR2-type)


class DeltaRLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_length: int,
        num_subspaces: int,
        step_size: float,
        quantization_error: float = 1e-3,
    ):
        super().__init__()
        assert dim % num_subspaces == 0
        self.subspaces = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.orthogonal_(self.subspaces)
        self.subspace_dim = dim // num_subspaces
        self.seq_length = seq_length
        self.dim = dim
        self.alpha = dim / seq_length / quantization_error**2
        self.beta = self.subspace_dim / seq_length / quantization_error**2
        self.step_size = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: test

        ## EXPANSION GRADIENT
        cov = einsum(x, x, "b n d, b m d -> b n m")
        shrinkage_matrix = torch.linalg.inv(
            torch.eye(self.seq_length)[None, ...] + self.alpha * cov
        )
        expansion_grad = (
            self.alpha
            * (self.dim + self.seq_length)
            * einsum(x, shrinkage_matrix, "b n d, b n m -> b m d")
        )

        ## COMPRESSION GRADIENT
        # compute projections into subspaces
        projectors = rearrange(self.subspaces, "d (k p) -> k p d", p=self.subspace_dim)
        projections = einsum(
            projectors,
            x,
            "k p d, b n d -> b k p n",
        )
        # compute covariances (optimizing for n being larger than p...)
        projected_covs = einsum(
            projections,
            projections,
            "b k p n, b k q n -> b k p q",
        )
        # invert to get shrinkage tensor
        shrinkage_tensor = torch.linalg.inv(
            torch.eye(self.subspace_dim)[None, None, ...] + self.beta * projected_covs
        )
        # get the compression gradient
        shrunk_projections = einsum(
            projections, shrinkage_tensor, "b k p n, b k p q -> b k q n"
        )
        lifted_projections = einsum(
            shrunk_projections, projectors, "b k p n, k p d -> b n d"
        )
        compresion_grad = (
            self.beta * (self.subspace_dim + self.seq_length) * lifted_projections
        )

        grad = self.step_size * (expansion_grad - compresion_grad)
        return x + grad

    def get_subspaces(self) -> torch.Tensor:
        projectors = rearrange(self.subspaces, "d (k p) -> k p d", p=self.subspace_dim)
        return projectors


class ImageTokenizer(nn.Module):
    """Tokenize an image into non-overlapping patches."""

    # TODO: support overlapping patches
    def __init__(
        self,
        embedding_dim: int,
        num_registers: int,
        patch_h: int,
        patch_w: int,
        # stride_h: int,
        # stride_w: int,
        input_h: int,
        input_w: int,
        input_c: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.stride_h = patch_h
        self.stride_w = patch_w
        self.input_h = input_h
        self.input_w = input_w
        self.input_c = input_c
        self.num_registers = num_registers

        # input checks
        assert input_h % patch_h == 0
        assert input_w % patch_w == 0

        # Tokenizer parameters
        self.seq_length = (input_h // patch_h) * (input_w // patch_w) + num_registers
        # TODO: how to initialize these? default?
        # TODO: See CRATE implementation (do it with einops and LN instead)
        self.embedding = nn.Conv2d(
            in_channels=self.input_c,
            out_channels=self.embedding_dim,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.stride_h, self.stride_w),
        )
        # TODO: how to initialize these? default? (CRATE: normal, and lots of layernorms...)
        self.registers = nn.Parameter(
            torch.randn((1, num_registers, embedding_dim), requires_grad=True)
            / math.sqrt(embedding_dim)
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tokenize the image
        x = self.embedding(x)
        x = self.flatten(x)
        x = x.moveaxis(-1, -2)  # batch x seq x dim
        # Concatenate registers
        regs = repeat(self.registers, "1 n d -> b n d", b=x.shape[0])
        x = torch.cat((regs, x), dim=1)
        return x


class MCRNet(nn.Module):
    """Learn a representation for images based on gradient ascent on rate reduction over tokens."""

    def __init__(
        self,
        # input and tokenizer params
        input_h: int,
        input_w: int,
        input_c: int,
        patch_h: int,
        patch_w: int,
        num_classes: int,
        # model params
        num_subspaces: int,
        num_registers: int,
        num_layers: int,
        step_size: float,
        quantization_error: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_h = input_h
        self.input_w = input_w
        self.input_c = input_c
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.num_subspaces = num_subspaces
        self.num_layers = num_layers
        self.step_size = step_size
        self.quantization_error = quantization_error

        self.tokenizer = ImageTokenizer(
            embedding_dim=self.embedding_dim,
            num_registers=self.num_registers,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            input_h=self.input_h,
            input_w=self.input_w,
            input_c=self.input_c,
        )
        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.layers.append(
                # nn.ModuleList(
                #     [
                PreNorm(
                    self.embedding_dim,
                    DeltaRLayer(
                        dim=embedding_dim,
                        seq_length=self.tokenizer.seq_length,
                        num_subspaces=self.num_subspaces,
                        step_size=self.step_size,
                        quantization_error=self.quantization_error,
                    ),
                ),
                # ]
                # )
            )
        self.postprocessor = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tokenize input
        x = self.tokenizer(x)
        # feedforward
        for layer in self.layers:
            x = layer(x)
        # Convert to CLS token
        x = x[:, 0, :]
        # Postprocess for output
        x = self.postprocessor(x)
        return x


if __name__ == "__main__":
    # test things
    img_h = 32
    img_w = 40
    img_c = 3
    batch = 4098
    embedding_dim = 64
    patch_h = 8
    patch_w = 8
    num_regs = 1
    I = torch.rand(batch, img_c, img_h, img_w)

    # Mnist stuff
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
        # [transforms.ToTensor()]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=10000, shuffle=True, num_workers=2)

    # setup arch
    num_subspaces = 4
    step_size = 1e-1
    quantization_error = 1e0
    model = MCRNet(
        input_h=28,
        input_w=28,
        input_c=1,
        patch_h=4,
        patch_w=4,
        num_classes=10,
        num_subspaces=num_subspaces,
        num_registers=num_regs,
        num_layers=4,
        step_size=step_size,
        quantization_error=quantization_error,
    )

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim_config = OptimizerConfig()
    num_epochs = 100

    model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim_config.algorithm.value(
        model.parameters(),
        lr=optim_config.learning_rate,
        weight_decay=optim_config.weight_decay,
    )

    losses = []
    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0):
        for batch, (data, labels) in enumerate(
            tqdm(trainloader, desc="Batch", position=1, leave=False)
        ):
            optimizer.zero_grad()
            out = model(data)
            l = loss(out, labels)
            l.backward()
            optimizer.step()

        # Evaluate accuracy
        with torch.no_grad():
            data, labels = next(iter(testloader))
            logits = model(data)
            preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            print(f'Accuracy: {acc}')

    # # Test tokenizer
    # t = ImageTokenizer(embedding_dim, num_regs, patch_h, patch_w, img_h, img_w, img_c)
    # tokens = t(I)
    #
    # # Test DeltaR layer
    # l = DeltaRLayer(
    #     dim=embedding_dim,
    #     seq_length=tokens.shape[1],
    #     num_subspaces=num_subspaces,
    #     step_size=step_size,
    #     quantization_error=quantization_error,
    # )
    # x = l(tokens)

    # check!
    breakpoint()
