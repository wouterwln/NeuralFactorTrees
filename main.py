from argparser import parse_trainer
import argparse
from train import *
import torch


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Train the GMNN model on HGCAL data")
    parser = parse_trainer(parser)
    parser = TIGMN.add_model_specific_args(parser)
    parser = GMNN.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.gmnn:
        model = train(args.batch_size, args.training_fraction, args.epochs, args.workers, args.prefetch_factor,
                      args.sampling_fraction, args.seed, args.hidden_dim, 2,
                      args.learning_rate, args.gpus, args.dropout)
    else:
        model = train_tigmn(args.batch_size, args.training_fraction, args.epochs, args.workers, args.prefetch_factor, args.sampling_fraction, args.seed, args.hidden_dim, args.num_gnn_steps,
          args.learning_rate, args.gpus, args.dropout, args.backbone)

