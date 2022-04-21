# Graph Pruning at the CMS HGCAL
This repository contains code to train and evaluate a graph pruning model on CMS HGCAL event data.
The data used here is preprocessed event data, where CLUE3D reconstructed trackster data is available.
From the CLUE3D graphs we infer what trackster was being estimated and construct a graph in the training data.

## Features
- Multi-GPU training
- Multithread data fetching
- GRU and LSTM aggregator and node memory
- Tensorboard logging
- CLI for hyperparameter tuning

## Requirements:
- python
- numpy
- pytorch
- dgl
- pytorch-lighting
- (optional) Tensorboard

Please make sure to install pytorch and dgl by hand (pip), since these depend on CUDA versions.

## Project Overview
This repository contains code to build and train a topology invariant graph vertex classification module.
By parameterizing a Gibbs Distribution that factorizes over a Markov Network we are able to estimate the joint probability distribution over vertex labels. The project is structured as follows: `train.py` contains 
the training logic, which invokes `model/graph_pruning.py`, which contains the actual module. Elementary submodules
are hidden in `model/modules.py` such that `model/graph_pruning.py` only contains the novel graph pruning
logic. `dataloader.py` contains the data fetching scripts and is invoked by multiple threads with pytorch's built-in DataLoader.
The entire training loop is invoked by `main.py` which parses the CLI arguments and trains the model accordingly.
## Usage
The model training is invoked by running `main.py` with command line parameters. For a complete overview of what command line parameters are implemented, run:
```shell
python3 main.py -h
```
An example command would be the following:
```shell
python3 main.py --hidden_dim 64 --num_gnn_steps 6 --num_layers 3 --epochs 100 --workers 14
```
This trains a model using 14 data fetching threads with 64 neurons, 6 message passing steps with a 3 layered MLP as transfer function for 100 epochs.
### Viewing logs and performance
Since we use [Pytorch Lightning](https://www.pytorchlightning.ai/), the metrics we specify are automatically logged and we can upload these to Tensorboard. To do this, run 
```shell
tensorboard dev upload --logdir tb_logs/GraphPruner --name "GraphPruning"
```
to upload an experiment to TensorBooard. Here we also have hyperparameter logging under the hyperparameter tab to track the individual settings of the different experiments.