#!/bin/sh

python denoise_square.py --dimension=4 --config.batch_size=None --dataset_size=1 --config.noise-level=0.1 --config.num-epochs 1000
