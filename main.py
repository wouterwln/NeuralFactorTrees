from argparser import parse_trainer
import argparse
from model.graph_pruning import GMNN
from train import train, train_synthetic_data
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GMNN model on HGCAL data")
    parser = parse_trainer(parser)
    parser = GMNN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = train(args.filename, args.trackster_root_name, args.edge_root_name, args.batch_size, args.training_fraction,
                      args.pretrain_epochs, args.iterations, args.epochs_q, args.epochs_p, args.workers,
                      args.prefetch_factor, args.sampling_fraction, args.seed, args.hidden_dim, args.num_gnn_steps,
                      args.learning_rate, args.gpus, args.dropout, args.teacher_forcing, args.stratified_tf, args.include_features)
    torch.save(model.state_dict(), args.output_file)
