#!/bin/sh

python denoise_square.py --dimension=4 --config.batch_size=None \
--dataset_size=128 --config.noise-level=1.0 --config.num-epochs 1000 \
--config.optimizer.weight-decay 0.0 --config.optimizer.algorithm=ADAM
