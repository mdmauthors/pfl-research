#!/bin/bash

set -oeux

export PYTHONPATH=.

python3 training/train_pfl_simulation.py --dataset femnist --model_name simple_cnn --algorithm_name fedavg --central_privacy_mechanism none --cohort_size 50 --val_cohort_size 0 --central_num_iterations 10 --local_num_epochs 10
