#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy import special
from scipy.special import gamma, gammainc, gammaincc

# Quick experiment to look at a certain random variable


def experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 1000
    E_min_dist_g_results = []
    E_min_dist_zero_results = []
    N = torch.arange(10, 4000, 100)
    d = 300
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
        # / special.gamma(1 + 2 / d)
        # / (d / (d - 2)) ** (d / 2)
        / special.gamma(d / 2 + 1) ** (2 / d),
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


def tail_bound_experiment():
    n = 30
    D = np.linspace(10, 100)
    T = 100

    tail = np.zeros((len(D), T))
    t = np.zeros_like(tail)
    for idx_d in range(len(D)):
        d = D[idx_d]
        # t[idx_d, :] = np.logspace(-5, 0, T) * np.sqrt(d)
        t[idx_d, :] = np.logspace(-4, 1, T, base=np.sqrt(d))

        tail[idx_d, :] = gammainc(
            d / 2, (np.sqrt(d) - t[idx_d, :]) ** 2 / 2
        ) + gammaincc(d / 2, (np.sqrt(d) + t[idx_d, :]) ** 2 / 2)
    # tail = gammainc(d / 2, (np.sqrt(d)) ** 2 / 2) + gammaincc(
    #     d / 2, (np.sqrt(d) + t) ** 2 / 2
    # )

    # print(tail[-1])
    # plt.plot(t.T, tail.T)
    scan = np.log(t[0, :]) / np.log(np.sqrt(D[0]))
    idx = np.abs(scan + 1).argmin()
    plt.plot(t[:, idx], tail[:, idx])
    plt.show()


if __name__ == "__main__":
    experiment()
    # tail_bound_experiment()
