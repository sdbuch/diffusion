#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from einops import reduce

def chamfer(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # ell^2 distances between each point in X (batch_sz x ...) and the closest
    # point in Y (batch_sz x ...)
    dists_all = torch.cdist(X[None, ...], Y[None, ...])[0, ...]
    dists_min, arg_min = dists_all.min(dim=0)
    return dists_min
