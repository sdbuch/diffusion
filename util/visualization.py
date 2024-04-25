#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch


def plot_histogram(X: torch.Tensor, title: str = '', show: bool = True):
    X = torch.flatten(X)
    plt.hist(X.to("cpu"), density=True)
    plt.title(title)
    if show == True:
        plt.show()
    # TODO: wandb integration
