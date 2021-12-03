from dataloader import TracksterDataset, SyntheticData
from torch.utils.data import DataLoader, Subset, random_split
from model.graph_pruning import *
import torch
import pytorch_lightning as pl
import math
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


def train(filename, trackster_root_name, edge_root_name, batch_size, training_fraction, pretrain_epochs, iterations,
          epochs_q, epochs_p, workers, prefetch_factor, sampling_fraction, seed, hidden_dim, num_gnn_steps,
          learning_rate, gpus, dropout, teacher_forcing, stratified_tf, include_feats):
    train_loader, val_loader, test_loader = prepare_data(filename, trackster_root_name, edge_root_name,
                                            sampling_fraction, training_fraction, seed, batch_size,
                                            workers, prefetch_factor)
    pl.seed_everything(seed)
    model = GMNN(hidden_dim, pretrain_epochs, epochs_q, epochs_p, dropout=dropout, lr=learning_rate,
                 teacher_forcing=teacher_forcing, num_steps=num_gnn_steps, stratified_tf=stratified_tf, include_features=include_feats)

    trainer = get_trainer(pretrain_epochs + iterations * (epochs_q + epochs_p), gpus, "gmnn")
    trainer.fit(model, train_loader, val_loader)
    return model


def prepare_data(filename, trackster_root_name, edge_root_name, sampling_fraction, training_fraction, seed, batch_size,
                 workers, prefetch_factor):
    dataset = TracksterDataset(filename, trackster_root_name, edge_root_name)
    dataset = Subset(dataset, [i for i in range(math.floor(sampling_fraction * len(dataset)))])
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers, persistent_workers=True,
                              collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=math.ceil(workers / 2),
                            persistent_workers=True,
                            collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=3, persistent_workers=True,
                             collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    return train_loader, val_loader, test_loader


def get_trainer(epochs, gpus, tag):
    logger = TensorBoardLogger(save_dir="tb_logs", name=tag)
    if gpus == 1:
        trainer = pl.Trainer(gpus=1, precision=32, strategy="dp", max_epochs=epochs, logger=logger)
    else:
        trainer = pl.Trainer(gpus=gpus, precision=32, strategy=DDPPlugin(find_unused_parameters=False),
                             max_epochs=epochs, logger=logger, num_sanity_val_steps=0)
    return trainer


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

    model, logger = GMNN.parse_arguments(args)
    trainer = pl.Trainer(gpus=1, precision=32, accelerator="dp", max_epochs=epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    return model
