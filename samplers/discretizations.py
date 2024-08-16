#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from util.types_custom import FloatTensor


class LinearDiscretization:
    def __init__(self, min_time: FloatTensor, max_time: FloatTensor, num_points: int):
        self.min_time = min_time
        self.max_time = max_time
        self.num_points = num_points
        self.dt = (max_time - min_time) / (num_points - 1)
        self.idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx < self.num_points:
            return self.min_time + self.idx * self.dt
        raise StopIteration

    def __len__(self):
        return self.num_points


class ExpoLinearDiscretization:
    def __init__(self, num_points: int, early_stop_time: FloatTensor):
        assert num_points % 2 == 0
        self.num_points = num_points
        self.early_stop_time = early_stop_time
        self.max_time = 1 + (self.num_points / 2) * (
            torch.exp(-2.0 / self.num_points * torch.log(self.early_stop_time)) - 1
        )  # This scheme forces num_points, early_stop_time, max_time to be related
        # TODO: Can allow to select one...? Or modify it (from T-1 breakpoint)
        assert self.max_time - early_stop_time > 0
        assert self.max_time > 1
        self.dt = (self.max_time - 1) / (self.num_points // 2)
        self.time = 0.0 - self.dt
        self.idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx <= self.num_points // 2:
            self.time = self.idx * self.dt
            return self.time
        elif self.idx <= self.num_points:
            self.time = self.max_time - torch.exp(
                -(self.idx - self.num_points // 2) * torch.log(1 + self.dt)
            )
            return self.time
        raise StopIteration

    def __len__(self):
        return self.num_points
