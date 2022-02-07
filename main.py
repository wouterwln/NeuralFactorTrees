from argparser import parse_trainer
import argparse
from model.graph_pruning import GMNN
from train import *
import torch
import numpy as np

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Train the GMNN model on HGCAL data")
    parser = parse_trainer(parser)
    parser = GMNN.add_model_specific_args(parser)
    args = parser.parse_args()
    gridsearch(args.batch_size, args.training_fraction, args.epochs, args.workers, args.prefetch_factor, args.sampling_fraction, args.seed,args.learning_rate, args.gpus,args.dropout)

    #torch.save(model.state_dict(), args.output_file)
