#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from einops import reduce


def chamfer(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # ell^2 distances between each point in X (batch_sz x ...) and the closest
    # point in Y (batch_sz x ...)
    X_flattened = X.flatten(start_dim=1)
    Y_flattened = Y.flatten(start_dim=1)
    dists_all = torch.cdist(X_flattened[None, ...], Y_flattened[None, ...])[0, ...]
    dists_min, arg_min = dists_all.min(dim=0)
    return dists_min
