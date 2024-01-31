#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
import enum

class OptimizerType(enum.Enum):
    ADAM = enum.auto()
    SGD = enum.auto()
