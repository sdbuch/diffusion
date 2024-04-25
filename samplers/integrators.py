#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt

import torch


def euler_maruyama(x: torch.Tensor, t: float, next_t: float) -> tuple[torch.Tensor, float]:
    """
    Euler-Maruyama integration scheme for SDEs.

    :param x: Value of drift at time t
    :param t: Left endpoint of time interval
    :param next_t: Right endpoint of time interval
    :return: tuple(Integrated drift for process (x * dt), integrated diffusion coefficient for process (sqrt(dt)))
    """
    dt = next_t - t
    return (x * dt, sqrt(dt))
