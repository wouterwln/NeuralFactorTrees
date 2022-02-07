from dgl.data import SSTDataset

from dataloader import TracksterDataset, SyntheticData, InMemoryDataset
from dgl.dataloading.pytorch import GraphDataLoader
from torch.utils.data import DataLoader, Subset, random_split
from model.graph_pruning import *
import torch
import pytorch_lightning as pl
import math
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

def gridsearch(batch_size, training_fraction,epochs, workers, prefetch_factor, sampling_fraction, seed,
          learning_rate, gpus, dropout):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor)
    for h_dim in [64, 128, 256]:
        for num_gnn_steps in [2, 4, 6, 10]:
            #model = GMNN(num_h_feats=h_dim, in_feats=9, epochs=100, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate)
            #trainer = get_trainer(epochs, gpus, "gmnn")
            #trainer.fit(model, train_loader, val_loader)
            for backbone in ['gmm', 'gat', 'tag', 'ggsnn', 'gmm']:
                model = TIGMN(num_h_feats=h_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate,
                          backbone=backbone)
                trainer = get_trainer(epochs, gpus, "mn")
                trainer.fit(model, train_loader, val_loader)


def train( batch_size, training_fraction,epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim, num_gnn_steps,
          learning_rate, gpus, dropout, teacher_forcing, stratified_tf, include_feats):
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                            workers, prefetch_factor)
    pl.seed_everything(seed)
    #model = TIGMN(num_h_feats=hidden_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate)
    model = GMNN(num_h_feats=hidden_dim, in_feats=9, epochs=100, num_steps=num_gnn_steps, dropout=dropout)
    trainer = get_trainer(epochs, gpus, "gmnn")
    trainer.fit(model, train_loader, val_loader)
    return model

def train_tigmn( batch_size, training_fraction,epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim, num_gnn_steps,
          learning_rate, gpus, dropout, teacher_forcing, stratified_tf, include_feats):
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                            workers, prefetch_factor)
    pl.seed_everything(seed)
    model = TIGMN(num_h_feats=hidden_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate, backbone='gmm')
    #model = GMNN(num_h_feats=hidden_dim, in_feats=9, epochs=100, num_steps=num_gnn_steps, dropout=dropout)
    trainer = get_trainer(epochs, gpus, "gmnn")
    trainer.fit(model, train_loader, val_loader)
    return model


def prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                 workers, prefetch_factor):
    dataset = TracksterDataset("tracksters_preprocessed.root", "Tracksters;1", "Edges;1")
    dataset = Subset(dataset, [i for i in range(math.floor(sampling_fraction * len(dataset)))])
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor, persistent_workers=True,collate_fn=TracksterDataset.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor, persistent_workers=True,collate_fn=TracksterDataset.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=TracksterDataset.collate_fn)
    return train_loader, val_loader, test_loader


def get_trainer(epochs, gpus, tag):
    logger = TensorBoardLogger(save_dir="tb_logs", name=tag)
    if gpus == 1:
        trainer = pl.Trainer(gpus=1, precision=32, max_epochs=epochs, logger=logger)
    else:
        trainer = pl.Trainer(gpus=gpus, precision=32, strategy=DDPPlugin(find_unused_parameters=False),
                             max_epochs=epochs, logger=logger, num_sanity_val_steps=0)
    return trainer


def train_synthetic_data(batch_size, workers, args, epochs, training_fraction=0.75):
    dataset = SyntheticData(1000)
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
    pl.seed_everything(42)
    model = GMNN(16, 10, 0, 10, dropout=0.2, num_steps=2, teacher_forcing=0.5, include_features=False)
    trainer = get_trainer(20, 1, "gmnn_synthetic")
    trainer.fit(model, train_loader, val_loader)
    return model
