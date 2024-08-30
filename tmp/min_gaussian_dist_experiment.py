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

# Quick experiment to look at a certain random variable


def experiment():
    T = 1000
    E_min_dist_g_results = []
    E_min_dist_zero_results = []
    N = torch.arange(10, 4000, 10)
    d = 5
    for n in N:
        G = torch.randn((T, int(n), d))
        g = torch.randn((T, 1, d))
        E_min_dist_g = torch.cdist(G, g).squeeze(-1).min(-1).values.mean() ** 2
        E_min_dist_zero = (
            torch.cdist(G, torch.zeros_like(g)).squeeze(-1).min(-1).values.mean() ** 2
        )
        E_min_dist_g_results.append(E_min_dist_g)
        E_min_dist_zero_results.append(E_min_dist_zero)
    plt.semilogy(
        N.numpy(),
        N.numpy() ** (2 / d) * torch.tensor(E_min_dist_g_results).numpy(),
        label="E[min dist to random] * n**(2/d)",
    )
    plt.semilogy(
        N.numpy(),
        N.numpy() ** (2 / d) * torch.tensor(E_min_dist_zero_results).numpy(),
        label="E[min dist to 0] * n**(2/d)",
    )
    # plt.semilogy(N.numpy(), 1/N.numpy(), label='1/n')
    # plt.semilogy(N.numpy(), 1/N.numpy()**(2/d), label='1/n**(2/d)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment()
