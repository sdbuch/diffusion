#!/bin/sh

python denoise_square.py --dimension=4 --config.batch_size=None \
--dataset_size=128 --config.noise-level=0.5 --config.num-epochs 3000 \
--config.optimizer.weight-decay 0.0 --config.optimizer.algorithm=ADAM
