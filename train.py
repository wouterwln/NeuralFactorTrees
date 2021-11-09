from dataloader import TracksterDataset
from torch.utils.data import DataLoader, Subset, random_split
from modules import ARGraphPruningModule, ARGGSNN
import torch
import pytorch_lightning as pl
import math

def train(filename, trackster_root_name, edge_root_name, batch_size, training_fraction, epochs,
              workers, prefetch_factor, sampling_fraction, seed):
    dataset = TracksterDataset(filename, trackster_root_name, edge_root_name)
    dataset = Subset(dataset, [i for i in range(math.floor(sampling_fraction * len(dataset)))])
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers, collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=workers, collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers, collate_fn=TracksterDataset.collate_fn, prefetch_factor=prefetch_factor)
    # model
    model = ARGraphPruningModule(9, 16, 2)

    # training
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), precision=32, accelerator="dp", auto_lr_find=True, max_epochs=epochs)
    #lr_find = trainer.tuner.lr_find(model, train_loader)
    #fig = lr_find.plot(suggest=True)
    #fig.show()
    #model.hparams.learning_rate = lr_find.suggestion()
    trainer.fit(model, train_loader, val_loader)
    return model

