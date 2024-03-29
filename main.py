import os
import sys

sys.path.append(os.getcwd())

from argparser import parse_trainer
import argparse
from train import *
import torch
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Train models from the NeuralMarkovTree paper on SST or HGCAL data")
    parser = parse_trainer(parser)
    parser = NeuralMarkovTree.add_model_specific_args(parser)
    parser = GMNN.add_model_specific_args(parser)
    args = parser.parse_args()
    train_gmnn_sst(args.batch_size, args.epochs, args.workers, args.prefetch_factor, args.sampling_fraction, args.seed, args.hidden_dim, args.num_gnn_steps, args.gpus)
    train_nmt_sst(args.batch_size, args.epochs, args.workers, args.prefetch_factor,
                      args.sampling_fraction, args.seed, args.hidden_dim, args.num_gnn_steps, args.learning_rate,
                      args.gpus, args.dropout, args.backbone, args.num_layers)
