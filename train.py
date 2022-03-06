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
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import json


def gridsearch(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed,
               learning_rate, gpus, dropout):
    pl.seed_everything(seed)
    train_loader, val_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                            workers, prefetch_factor, InMemoryDataset)
    for h_dim in [32, 64, 128, 256]:
        for num_gnn_steps in [2, 4, 6]:
            # model = GMNN(num_h_feats=h_dim, in_feats=9, epochs=100, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate)
            # trainer = get_trainer(epochs, gpus, "gmnn")
            # trainer.fit(model, train_loader, val_loader)
            for backbone in ['gmm', 'gat', 'tag', 'ggsnn']:
                model = TIGMN(num_h_feats=h_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate,
                              backbone=backbone)
                trainer = get_trainer(epochs, gpus, "mn")
                trainer.fit(model, train_loader, val_loader)


def train(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim,
          num_gnn_steps,
          learning_rate, gpus, dropout):
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                            workers, prefetch_factor)
    pl.seed_everything(seed)
    model = GMNN(num_h_feats=hidden_dim, in_feats=9, epochs=epochs, num_steps=num_gnn_steps, dropout=dropout)
    trainer = get_trainer(epochs, gpus, "gmnn")
    trainer.fit(model, train_loader, val_loader)
    test_metrics = trainer.test(dataloaders=test_loader, ckpt_path="best")[0]
    with open(f'gmnn-{sampling_fraction}-{hidden_dim}-{num_gnn_steps}-{epochs}.json', 'w') as f:
        json.dump(test_metrics, f)
    return model


def train_tigmn(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim,
                num_gnn_steps,
                learning_rate, gpus, dropout, backbone):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor)
    model = TIGMN(num_h_feats=hidden_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate, backbone=backbone)
    trainer = get_trainer(epochs, gpus, "tigmn")
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    test_metrics = trainer.test(model, test_loader, ckpt_path="best")[0]
    with open(f'{backbone}-{sampling_fraction}-{hidden_dim}-{num_gnn_steps}-{epochs}.json', 'w') as f:
        json.dump(test_metrics, f)
    return model



def prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                 workers, prefetch_factor, memset=TracksterDataset, test=False):
    dataset = memset("tracksters_preprocessed.root", "Tracksters;1", "Edges;1")
    dataset = Subset(dataset, [i for i in range(math.floor(sampling_fraction * len(dataset)))])
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_data, batch_size=16*batch_size, num_workers=workers, prefetch_factor=prefetch_factor,
                             persistent_workers=False, collate_fn=TracksterDataset.collate_fn)
    if not test:
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers,
                                  prefetch_factor=prefetch_factor, persistent_workers=False,
                                  collate_fn=TracksterDataset.collate_fn)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor,
                                persistent_workers=False, collate_fn=TracksterDataset.collate_fn)
        return train_loader, val_loader, test_loader
    else:
        return test_loader


def get_trainer(epochs, gpus, tag):
    logger = TensorBoardLogger(save_dir="tb_logs", name=tag)
    if tag == "gmnn":
        trainer = pl.Trainer(gpus=1, precision=32, max_epochs=epochs, logger=logger)
        return trainer
    if gpus == 1:
        trainer = pl.Trainer(gpus=1, precision=32, max_epochs=epochs, logger=logger, auto_lr_find=True,
                             gradient_clip_val=1.,callbacks=[StochasticWeightAveraging(0.5)])
    else:
        trainer = pl.Trainer(gpus=gpus, precision=32, strategy=DDPPlugin(find_unused_parameters=False),
                             max_epochs=epochs, logger=logger, num_sanity_val_steps=0, gradient_clip_val=1.,
                             callbacks=[StochasticWeightAveraging(0.5)])
    return trainer
