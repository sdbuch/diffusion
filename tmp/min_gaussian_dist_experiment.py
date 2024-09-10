#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy import special
from scipy.special import gamma, gammainc, gammaincc, gammaln, iv

# Quick experiment to look at a certain random variable


def experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    E_min_dist_g_results = []
    E_min_dist_zero_results = []
    T = 100
    N = torch.arange(10, 10000, 100)
    # d = 10
    # D = N
    D = 3 * torch.ones_like(N)
    for idx_n in range(len(N)):
        n = int(N[idx_n])
        d = int(D[idx_n])
        accumulator = 0.0
        counter = 2 * int(n)
        ticks = 0
        while counter > 0:
            G = torch.randn((T, int(n), d), device=device)
            g = torch.randn((T, 1, d), device=device)
            E_min_dist_g = (torch.cdist(G, g) ** 2).squeeze(-1).min(-1).values.sum()
            E_min_dist_zero = (
                (torch.cdist(G, torch.zeros_like(g)) ** 2)
                .squeeze(-1)
                .min(-1)
                .values.sum()
            )
            accumulator += E_min_dist_g.detach()
            # accumulator += E_min_dist_zero.detach()
            counter -= T
            ticks += T
        # print(ticks - total)
        E_min_dist_g_results.append(accumulator / ticks)
        E_min_dist_zero_results.append(accumulator / ticks)
    # D = torch.tensor((d,))
    plt.plot(
        N.numpy(),
        N.numpy() ** (2 / D.numpy())
        * torch.tensor(E_min_dist_g_results).numpy()
        / 2
        / special.gamma(1 + 2 / D)
        / (D / (D - 2)) ** (D / 2)
        / torch.exp(2/D * gammaln(D/2 + 1)),
        label="E[min dist to random] * n**(2/d)",
    )
    # plt.plot(
    #     N.numpy(),
    #     N.numpy() ** (2 / D.numpy())
    #     * torch.tensor(E_min_dist_zero_results).numpy()
    #     / 2
    #     / torch.exp(2 / D * gammaln(D / 2 + 1))
    #     / special.gamma(1 + 2 / D.numpy()),
    #     label="E[min dist to 0] * n**(2/d)",
    #
    # plt.plot(
    #     N.numpy(),
    #     torch.tensor(E_min_dist_zero_results).numpy() / (D),
    #     label="E[min dist to 0] * different",
    # )
    plt.legend()
    plt.show()


def experiment_tail():
    # TODO: see how fast we converge to the limit tail bound...
    pass


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


def bessel_experiment():
    D = np.linspace(10, 100)
    t = np.sqrt(D)
    a = iv(D / 2 - 1, t)
    b = np.exp((D / 2 - 1) * np.log(t / 2) - gammaln(D / 2))
    plt.semilogy(D, a)
    plt.semilogy(D, b)
    plt.show()


if __name__ == "__main__":
    experiment()
    # tail_bound_experiment()
    # bessel_experiment()
