#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from util.types_custom import FloatTensor


def euler_maruyama(
    x: torch.Tensor, t: FloatTensor, next_t: FloatTensor
) -> tuple[torch.Tensor, FloatTensor]:
    """
    Euler-Maruyama integration scheme for SDEs.

    :param x: Value of drift at time t
    :param t: Left endpoint of time interval
    :param next_t: Right endpoint of time interval
    :return: tuple(Integrated drift for process (x * dt), integrated diffusion coefficient for process (sqrt(dt)))
    """
    dt = next_t - t
    return (x * dt, torch.sqrt(dt))
