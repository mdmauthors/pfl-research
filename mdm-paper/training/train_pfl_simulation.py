#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import numpy as np
import torch  # type: ignore

from pfl.aggregate.simulate import SimulatedBackend
from pfl.aggregate.weighting import WeightByDatapoints, WeightByUser
from pfl.callback import (AggregateMetricsToDisk, CentralEvaluationCallback,
                             StopwatchCallback, ModelCheckpointingCallback)
from pfl.model.pytorch import PyTorchModel
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams
from pfl.privacy import CentrallyAppliedPrivacyMechanism
from pfl.internal.ops.pytorch_ops import to_tensor
from polya_mixture.datasets import make_femnist_datasets
from polya_mixture.utils.argument_parsing import add_experiment_args
from model.argument_parsing import add_model_arguments, get_model_pytorch
from utils.argument_parsing import (add_algorithm_arguments,
                                    add_common_arguments,
                                    add_dnn_training_arguments,
                                    add_mechanism_arguments, get_algorithm,
                                    parse_mechanism)
from utils.logging import init_logging

init_logging(logging.DEBUG)
logger = logging.getLogger(name=__name__)

argument_parser = argparse.ArgumentParser(
    description='Train a model using private federated learning in simulation.'
)

argument_parser = add_dnn_training_arguments(argument_parser)
argument_parser = add_experiment_args(argument_parser)
argument_parser = add_mechanism_arguments(argument_parser)
argument_parser = add_common_arguments(argument_parser)
argument_parser = add_algorithm_arguments(argument_parser)
argument_parser = add_dataset_arguments(argument_parser)
argument_parser = add_model_arguments(argument_parser)
arguments = argument_parser.parse_args()

torch.random.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

local_privacy = parse_mechanism(
    mechanism_name=arguments.local_privacy_mechanism,
    clipping_bound=arguments.local_privacy_clipping_bound,
    epsilon=arguments.local_epsilon,
    delta=arguments.local_delta,
    order=arguments.local_order)

central_privacy = parse_mechanism(
    mechanism_name=arguments.central_privacy_mechanism,
    clipping_bound=arguments.central_privacy_clipping_bound,
    epsilon=arguments.central_epsilon,
    delta=arguments.central_delta,
    order=arguments.central_order,
    cohort_size=arguments.cohort_size,
    noise_cohort_size=arguments.noise_cohort_size,
    num_epochs=arguments.central_num_iterations,
    min_separation=arguments.min_separation,
    population=1e6)

training_federated_dataset, val_federated_dataset, central_data = make_femnist_datasets(
    arguments.data_dir,
    digits_only=False,
    numpy_to_tensor=to_tensor,
    dataset_type=arguments.dataset_type)

pytorch_model = get_model_pytorch(arguments)

params = [p for p in pytorch_model.parameters() if p.requires_grad]

model = PyTorchModel(model=pytorch_model,
                     local_optimizer_create=torch.optim.SGD,
                     central_optimizer=torch.optim.SGD(
                         params, arguments.learning_rate))

weighting_strategy = WeightByDatapoints(
) if arguments.weight_by_samples else WeightByUser()

postprocessors = [
    weighting_strategy, local_privacy,
    CentrallyAppliedPrivacyMechanism(central_privacy)
]

backend = SimulatedBackend(training_data=training_federated_dataset,
                           val_data=val_federated_dataset,
                           postprocessors=postprocessors)

algorithm, algorithm_params = get_algorithm(arguments)

model_train_params = NNTrainHyperParams(
    local_learning_rate=arguments.local_learning_rate,
    local_num_epochs=arguments.local_num_epochs,
    local_batch_size=arguments.local_batch_size
    if arguments.local_batch_size > 0 else None)

model_eval_params = NNEvalHyperParams(
    local_batch_size=arguments.central_eval_batch_size)

# Central evaluation on val data.
central_evaluation_cb = CentralEvaluationCallback(
    central_data,
    model_eval_params=model_eval_params,
    frequency=arguments.evaluation_frequency)

if arguments.restore_model_path is not None:
    model.load(arguments.restore_model_path)
    logger.info('Restored model from {}'.format(arguments.restore_model_path))

model = algorithm.run(algorithm_params=algorithm_params,
                      backend=backend,
                      model=model,
                      model_train_params=model_train_params,
                      model_eval_params=model_eval_params,
                      callbacks=[
                          StopwatchCallback(),
                          central_evaluation_cb,
                          ModelCheckpointingCallback('./checkpoints'),
                          AggregateMetricsToDisk('./metrics.csv'),
                      ])
