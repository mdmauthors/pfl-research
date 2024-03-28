# Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomials (MDMs)

This software project accompanies the research paper, "Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomials".

Mixture-of-Dirichlet-Multinomials (MDMs) allow one to model heterogeneous federated datasets, and such MDM models can be trained privacy preserving federated learning.

## Documentation

This repo contains the code to run all experiments in the paper "Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomials", and to process the results to produce the plots shown in the paper are available in the `mdm-paper` directory on this fork of the `pfl-research` framework, for running simulations using Private Federated Learning.

The structure of the `mdm-paper` repo is:
- `experiments/`: This folder contains bash scripts to run local training to infer parameters of Mixture-of-Dirichlet-Multinomial (MDM) models for the CIFAR-10 and FEMNIST datasets. These bash scripts invoke scripts in the `training/` folder.
- `training/`: This folder contains code to run local training to infer parameters of MDM models.
- `notebooks/`: This folder contains iPython notebooks to process results and produce the plots which are included in the paper "Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomialsu".
- `mdm/`: This folder contains the code to implement the MDM algorithm in the pfl-research framework.
