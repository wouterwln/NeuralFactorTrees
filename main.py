from argparser import parse_trainer
import argparse
from model.graph_pruning import GraphPruner
from train import train, gridsearch, train_synthetic_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AutoRegressive Graph Pruning model on HGCAL data")
    parser = parse_trainer(parser)
    parser = GraphPruner.add_model_specific_args(parser)
    args = parser.parse_args()
    model = train(args.filename, args.trackster_root_name, args.edge_root_name, args.batch_size, args.training_fraction,
                  args.epochs, args.workers, args.prefetch_factor, args.sampling_fraction, args.seed, args.hidden_dim, args.num_gnn_steps, args.num_iterations, args.aggregator,
                   args.learning_rate, args.autoregressive, args.memory, args.gpus)