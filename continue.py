import os
import sys
sys.path.append(os.getcwd())

from argparser import parse_continue
import argparse
from train import *
import torch
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Continue training the TIGMN with extra edges model on HGCAL data")
    parser = parse_continue(parser)
    parser = TIGMN.add_model_specific_args(parser)
    parser = GMNN.add_model_specific_args(parser)
    args = parser.parse_args()
    continue_training(args.batch_size, args.training_fraction, args.epochs, args.workers, args.prefetch_factor, args.sampling_fraction, args.seed, args.gpus, 10, args.num_layers, args.ckpt)