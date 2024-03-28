#!/bin/bash

set -oeux

export PYTHONPATH=.

num_components=2

if [ "$num_components" -eq 1 ]; then

    #alphas=0.3
    alphas='0.3 0.2 0.1 1 1.2 1.3 0.8 0.5 1.2 0.1'
    phi='1.0'

    python3 training/train.py --num_mixture_components "$num_components" --component_mean_user_dataset_length 20 --component_alphas "$alphas" --cohort_size_init_algorithm 1000 --central_num_iterations_init_algorithm 1 --cohort_size_algorithm 1000 --central_num_iterations_algorithm 30 --cohort_size_histogram_algorithm 1000 --component_phi "$phi" --data_dir data/cifar10
fi

if [ "$num_components" -eq 2 ]; then

    alphas='0.3 0.2 0.1 1 1.2 1.3 0.8 0.5 1.2 0.1 0.3 0.2 1.8 1 0.2 0.1 0.1 0.1 3.2 1.1'
    phi='0.1 0.9'
    user_dataset_length='20 60'

    python3 training/train.py --num_mixture_components "$num_components" --component_mean_user_dataset_length "$user_dataset_length" --component_alphas "$alphas" --cohort_size_init_algorithm 1000 --central_num_iterations_init_algorithm 1 --cohort_size_algorithm 1000 --central_num_iterations_algorithm 30 --cohort_size_histogram_algorithm 1000 --component_phi "$phi" --data_dir data/cifar10 
fi
