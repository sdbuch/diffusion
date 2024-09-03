#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from scipy import optimize
from scipy.special import gamma


def root_pmem(d: int, x: float, var: float, N: int, K: int) -> float:
    f = (
        lambda M: 2
        * gamma(d / 2 + 1) ** (d / 2)
        * gamma(2 / d + 1)
        * (d / (d - 2)) ** (d / 2)
        * var
        * (1 - M / N)
        * (K / M) ** (2 / d)
        - x
    )
    return optimize.root_scalar(f)
