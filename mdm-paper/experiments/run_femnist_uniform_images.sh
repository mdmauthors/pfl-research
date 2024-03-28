#!/bin/bash

set -oeux

export PYTHONPATH=.

num_components=2

python3 training/train_femnist.py --num_mixture_components "$num_components" --cohort_size_init_algorithm 1000 --central_num_iterations_init_algorithm 1 --cohort_size_algorithm 1000 --central_num_iterations_algorithm 50 --data_dir data/femnist --cohort_size_histogram_algorithm 1000 --dataset_type original_labels_uniform_datapoints 
