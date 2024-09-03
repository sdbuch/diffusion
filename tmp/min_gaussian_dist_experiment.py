#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy.special import gamma, gammainc, gammaincc

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
    # experiment()
    tail_bound_experiment()
