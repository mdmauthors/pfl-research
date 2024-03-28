import numpy as np

from pfl.statistics.polya_mixture import (
    PolyaMixtureModel, PolyaMixtureModelHyperParams, PolyaMixtureAlgorithm,
    PolyaMixtureAlgorithmParams, PolyaMixtureInitializationAlgorithm,
    PolyaMixtureInitializationAlgorithmParams)
from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import ModelCheckpointingCallback
from pfl.privacy import CentrallyAppliedPrivacyMechanism, MomentsAccountantGaussianMechanism

from .utils.tools import ModelCheckpointingIterationCallback


def solve_polya_mixture_mle(
        arguments,
        training_federated_dataset,
        val_federated_dataset,
        num_components,
        num_categories,
        save_path,
        save_path_histogram,
        add_DP=False,
        extract_labels_fn=lambda user_dataset: user_dataset.raw_data[1]):
    """
    Solve polya-mixture MLE
    """

    # model
    model = PolyaMixtureModel()
    model_params = PolyaMixtureModelHyperParams(num_components, num_categories)
    print(f'before init algo - model alphas: {model.alphas}, phi: {model.phi}')

    if add_DP:
        num_iterations = arguments.central_num_iterations_init_algorithm + arguments.central_num_iterations_algorithm
        max_cohort_size = arguments.cohort_size_algorithm  # TODO make this the max of init cohort size too.

        sampling_rate = 0.001
        mechanism = MomentsAccountantGaussianMechanism(
            clipping_bound=0.5,
            epsilon=2,
            delta=1e-6,
            num_iterations=num_iterations,
            max_cohort_size=max_cohort_size,
            population_size=max_cohort_size / sampling_rate,
            noise_scale=1.0)

        postprocessors = [CentrallyAppliedPrivacyMechanism(mechanism)]
    else:
        postprocessors = []
    backend = SimulatedBackend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=postprocessors)

    bin_edges = np.linspace(
        0, arguments.max_num_samples_mixture_component_init_algorithm, 11)[1:]
    num_samples_mixture_bins = np.vstack([bin_edges] * 6)
    print('\nnum_samples_mixture_bins', num_samples_mixture_bins)

    # init algorithm
    init_algorithm = PolyaMixtureInitializationAlgorithm()
    init_algorithm_params = PolyaMixtureInitializationAlgorithmParams(
        cohort_size=arguments.cohort_size_init_algorithm,
        num_samples_mixture_bins=num_samples_mixture_bins,
        strategy=arguments.strategy,
        central_num_iterations=arguments.central_num_iterations_init_algorithm,
        extract_categories_fn=extract_labels_fn)
    init_algorithm.run(
        algorithm_params=init_algorithm_params,
        backend=backend,
        model=model,
        model_train_params=model_params,
        model_eval_params=None,
        callbacks=[ModelCheckpointingCallback(model_checkpoint_dir=save_path)])
    print(f'after init algo - model alphas: {model.alphas}, phi: {model.phi}')

    # TODO do I need to reset phi and alpha in model?
    # Require model phi = (1 / num_mixture_components) *
    # np.ones(num_mixture_components)

    # algorithm
    algorithm = PolyaMixtureAlgorithm()
    algorithm_params = PolyaMixtureAlgorithmParams(
        cohort_size=arguments.cohort_size_algorithm,
        num_samples_mixture_bins=num_samples_mixture_bins,
        central_num_iterations=arguments.central_num_iterations_algorithm,
        extract_categories_fn=extract_labels_fn)
    algorithm.run(
        algorithm_params=algorithm_params,
        backend=backend,
        model=model,
        model_train_params=model_params,
        model_eval_params=None,
        callbacks=[
            ModelCheckpointingCallback(model_checkpoint_dir=save_path),
            ModelCheckpointingIterationCallback(
                model_checkpoint_dir=save_path + '_iteration_models',
                checkpoint_frequency=1)
        ])
    print(
        f'after algo - model alphas: {model.alphas}, phi: {model.phi}, num_samples_distribution: {model.num_samples_distribution}'
    )

    return model.phi, model.alphas, model.num_samples_distribution
