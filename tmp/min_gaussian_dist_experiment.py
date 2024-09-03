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
from scipy import special

# Quick experiment to look at a certain random variable


def experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 10000
    E_min_dist_g_results = []
    E_min_dist_zero_results = []
    N = torch.arange(10, 10000, 100)
    d = 3
    for n in N:
        G = torch.randn((T, int(n), d), device=device)
        g = torch.randn((T, 1, d), device=device)
        E_min_dist_g = (torch.cdist(G, g) ** 2).squeeze(-1).min(-1).values.mean()
        E_min_dist_zero = (
            (torch.cdist(G, torch.zeros_like(g)) ** 2).squeeze(-1).min(-1).values.mean()
        )
        E_min_dist_g_results.append(E_min_dist_g)
        E_min_dist_zero_results.append(E_min_dist_zero)
    plt.plot(
        N.numpy(),
        N.numpy() ** (2 / d)
        * torch.tensor(E_min_dist_g_results).numpy()
        / 2
        / special.gamma(1 + 2 / d)
        / special.gamma(d / 2 + 1) ** (2 / d)
        / (d / (d - 2)) ** (d / 2),
        label="E[min dist to random] * n**(2/d)",
    )
    # plt.plot(
    #     N.numpy(),
    #     N.numpy() ** (2 / d)
    #     * torch.tensor(E_min_dist_zero_results).numpy()
    #     / 2
    #     / special.gamma(d / 2 + 1) ** (2 / d)
    #     / special.gamma(1 + 2 / d),
    #     label="E[min dist to 0] * n**(2/d)",
    # )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment()
