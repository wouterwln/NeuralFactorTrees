from dataloader import *
from torch.utils.data import DataLoader, Subset, random_split
from model.graph_pruning import *
import torch
import pytorch_lightning as pl
import math
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import json
from dgl.data import SSTDataset
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout


def train(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim,
          num_gnn_steps,
          learning_rate, gpus, dropout, fetching=False):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_sst_data(batch_size, workers, prefetch_factor)
    model = GMNN(in_feats=300, num_h_feats=hidden_dim, epochs=epochs, num_steps=num_gnn_steps, dropout=dropout)
    trainer = get_trainer(epochs, gpus, "gmnn")
    trainer.fit(model, train_loader, val_loader)
    metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    for i, m in enumerate(metrics):
        with open(f'gmnn-{sampling_fraction}-{hidden_dim}-{num_gnn_steps}-{epochs}-{i}.json', 'w') as f:
            json.dump(m, f)
    return model


def train_tigmn(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim,
                num_gnn_steps, learning_rate, gpus, dropout, backbone, num_layers):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_sst_data(sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor, extra_edges=False)
    model = TIGMN(num_h_feats=hidden_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate, backbone=backbone, num_layers=num_layers)
    trainer = get_trainer(epochs, gpus, "tigmn")
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    for i, m in enumerate(metrics):
        with open(f'{backbone}-{sampling_fraction}-{hidden_dim}-{num_gnn_steps}-{epochs}-{i}.json', 'w') as f:
            json.dump(m, f)
    return model


def prepare_sst_data(batch_size, workers, prefetch_factor, test=False):
    train_data = SSTDataset(glove_embed_file="glove.6B.300d.txt")
    val_data = SSTDataset(glove_embed_file="glove.6B.300d.txt", mode='dev')
    test_data = SSTDataset(glove_embed_file="glove.6B.300d.txt", mode='test')
    train_data.process()
    val_data.process()
    test_data.process()
    for dataset in [train_data, val_data, test_data]:
        for i in trange(len(dataset)):
            dataset[i].ndata["words"] = dataset[i].ndata["x"]
            dataset[i].ndata["x"] = train_data.pretrained_emb[dataset[i].ndata["x"]].float()
            dataset[i].ndata["x"][dataset[i].ndata["words"] == -1] = torch.zeros(300)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor,
                             persistent_workers=False, collate_fn=dgl.batch)
    if not test:
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers,
                                  prefetch_factor=prefetch_factor, persistent_workers=False, collate_fn=dgl.batch)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor,
                                persistent_workers=False, collate_fn=dgl.batch)
        return train_loader, val_loader, test_loader
    else:
        return test_loader

def prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                 workers, prefetch_factor, memset=TracksterDataset, test=False, k=0.75, extra_edges=True):
    dataset = SSTDataset(glove_embed_file="glove.6B.300d.txt")
    dataset.process()
    for i in trange(len(dataset)):
        dataset[i].ndata["words"] = dataset[i].ndata["x"]
        dataset[i].ndata["x"] = dataset.pretrained_emb[dataset[i].ndata["x"]].float()
        dataset[i].ndata["x"][dataset[i].ndata["words"] == -1] = torch.zeros(300)
    dataset = Subset(dataset, [i for i in range(math.floor(sampling_fraction * len(dataset)))])
    splits = [math.ceil(i * len(dataset)) for i in
              [training_fraction, (1. - training_fraction) / 2., (1. - training_fraction) / 2.]]
    while sum(splits) > len(dataset):
        splits[-1] -= 1
    train_data, val_data, test_data = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor,
                             persistent_workers=False, collate_fn=dgl.batch)
    if not test:
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=workers,
                                  prefetch_factor=prefetch_factor, persistent_workers=False, collate_fn=dgl.batch)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=workers, prefetch_factor=prefetch_factor,
                                persistent_workers=False, collate_fn=dgl.batch)
        return train_loader, val_loader, test_loader
    else:
        return test_loader


def get_trainer(epochs, gpus, tag):
    logger = TensorBoardLogger(save_dir="tb_logs", name=tag)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    progress_bar = TQDMProgressBar(refresh_rate=2)
    if tag == "gmnn":
        trainer = pl.Trainer(gpus=gpus, precision=32, max_epochs=epochs, logger=logger)
        return trainer
    if gpus == 1:
        trainer = pl.Trainer(gpus=1, precision=32, max_epochs=epochs, logger=logger, auto_lr_find=True,
                             gradient_clip_val=1.,callbacks=[checkpoint_callback, progress_bar, StochasticWeightAveraging(0.5)])
    else:
        trainer = pl.Trainer(gpus=gpus, precision=32, strategy=DDPPlugin(find_unused_parameters=False),
                             max_epochs=epochs, logger=logger, num_sanity_val_steps=0, gradient_clip_val=1.,
                             callbacks=[StochasticWeightAveraging(0.5), checkpoint_callback, progress_bar])
    return trainer


def train_intratrackster_model(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed, hidden_dim,
                num_gnn_steps, learning_rate, gpus, dropout, backbone, k, num_layers, hgt):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_sst_data(batch_size, workers, prefetch_factor)
    if not hgt:
        model = SSTTIGMN(num_h_feats=hidden_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate, backbone=backbone, num_layers=num_layers,
                               num_classes=5)
    else:
        model = HGTMultiTST_TIGMN(num_h_feats=hidden_dim, num_steps=num_gnn_steps, dropout=dropout, lr=learning_rate)
    trainer = get_trainer(epochs, gpus, "itt")
    #if gpus == 1:
    #    trainer.tune(model, train_loader, val_loader)

    trainer.fit(model, train_loader, val_loader)
    metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    for i, m in enumerate(metrics):
        with open(f'{backbone}-{sampling_fraction}-{hidden_dim}-{num_gnn_steps}-{epochs}-{i}.json', 'w') as f:
            json.dump(m, f)
    return model


def continue_training(batch_size, training_fraction, epochs, workers, prefetch_factor, sampling_fraction, seed, gpus, k, num_layers, ckpt):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor, k=k)

    trainer = get_trainer(epochs, gpus, "itt")
    try:
        model = MultiTST_TIGMN.load_from_checkpoint(ckpt, num_layers=num_layers)
    except:
        model = HGTMultiTST_TIGMN.load_from_checkpoint(ckpt)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)
    metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    for i, m in enumerate(metrics):
       with open(f'results.json', 'w') as f:
           json.dump(m, f)
    return model

def test(batch_size, training_fraction, workers, prefetch_factor, sampling_fraction, seed, gpus, k, num_layers, ckpt):
    pl.seed_everything(seed)
    train_loader, val_loader, test_loader = prepare_data(sampling_fraction, training_fraction, seed, batch_size,
                                                         workers, prefetch_factor, k=k)

    trainer = get_trainer(1, gpus, "itt")
    model = MultiTST_TIGMN.load_from_checkpoint(ckpt, num_layers=num_layers)
    metrics = trainer.test(model, dataloaders=test_loader)
    for i, m in enumerate(metrics):
        with open(f'{ckpt}.json', 'w') as f:
            json.dump(m, f)
    return model