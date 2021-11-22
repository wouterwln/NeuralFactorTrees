from dataloader import TracksterDataset, SyntheticData
from torch.utils.data import DataLoader, Subset, random_split
from model.graph_pruning import GraphPruner, GNNPruner
import torch
import pytorch_lightning as pl
import math
from pytorch_lightning.loggers import TensorBoardLogger


def train(filename, trackster_root_name, edge_root_name, batch_size, training_fraction, epochs, workers,
          prefetch_factor, sampling_fraction, seed, hidden_dim, num_gnn_steps, num_iterations, aggregator,
          learning_rate, autoregressive, memory, gpus):
    train_loader, val_loader, test_loader = prepare_data(filename, trackster_root_name, edge_root_name,
                                                         sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor)
    model = GNNPruner(9, hidden_dim, num_gnn_steps, num_iterations, aggregator, learning_rate, autoregressive, memory)

    logger = TensorBoardLogger(save_dir="tb_logs", name="GraphPruner")
    if gpus == 1:
        trainer = pl.Trainer(gpus=1, precision=32, accelerator="dp", max_epochs=epochs, logger=logger)
    else:
        trainer = pl.Trainer(gpus=gpus, precision=32, accelerator="ddp", max_epochs=epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    return model


def gridsearch(filename, trackster_root_name, edge_root_name, batch_size, training_fraction, epochs,
               workers, prefetch_factor, sampling_fraction, seed, args):
    train_loader, val_loader, test_loader = prepare_data(filename, trackster_root_name, edge_root_name,
                                                         sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor)
    for aggregator in ['gru', 'lstm']:
        for memory in ['gru', 'lstm', 'none']:
            args.aggregator = aggregator
            args.memory = memory
            model, logger = GraphPruner.parse_arguments(args)
            trainer = pl.Trainer(gpus=1, precision=32, accelerator="dp", max_epochs=epochs,
                                 logger=logger)
            trainer.fit(model, train_loader, val_loader)


def prepare_data(filename, trackster_root_name, edge_root_name, sampling_fraction, training_fraction, seed, batch_size,
                 workers, prefetch_factor):
    dataset = TracksterDataset(filename, trackster_root_name, edge_root_name)
    dataset = Subset(dataset, [i for i in range(math.floor(sampling_fraction * len(dataset)))])
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers,
                              collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=workers,
                            collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers,
                             collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    return train_loader, val_loader, test_loader


def train_synthetic_data(batch_size, workers, args, epochs, training_fraction=0.75):
    dataset = SyntheticData(200)
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator())
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              collate_fn=TracksterDataset.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            collate_fn=TracksterDataset.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=TracksterDataset.collate_fn)

    model, logger = GraphPruner.parse_arguments(args)
    trainer = pl.Trainer(gpus=1, precision=32, accelerator="dp", max_epochs=epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    return model
