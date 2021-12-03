import dgl.nn.pytorch
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from dgl.nn.pytorch import SAGEConv, GraphConv, GATConv, Sequential
import numpy as np


class GCNMNN(nn.Module):
    def __init__(self, num_h_feats, in_feats=9, dropout=0.5, num_classes=2):
        super(GCNMNN, self).__init__()
        self.conv1 = GraphConv(in_feats, num_h_feats)
        self.conv2 = GraphConv(num_h_feats, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, g, feats):
        h = self.dropout2(self.conv1(g, self.dropout1(feats)))
        o = self.conv2(g, h)
        return o


class GatMNN(nn.Module):
    def __init__(self, num_h_feats, in_feats=9, in_dropout=0.5, dropout=0., num_classes=2, num_heads=3, num_steps=2):
        super(GatMNN, self).__init__()
        assert num_steps >= 2
        self.conv1 = GATConv(in_feats, num_h_feats, num_heads, feat_drop=dropout)
        self.convBlock = Sequential(
            *[GATConv(num_h_feats, num_h_feats, num_heads, feat_drop=dropout) for _ in range(num_steps - 2)])
        self.conv2 = GATConv(num_h_feats, num_classes, num_heads, feat_drop=dropout)
        self.dropout1 = nn.Dropout(in_dropout)

    def forward(self, g, feats):
        h = self.conv1(g, self.dropout1(feats))
        h = torch.sum(h, dim=1)
        for layer in self.convBlock:
            h = layer(g, h)
            h = torch.sum(h, dim=1)
        o = self.conv2(g, h)
        o = torch.sum(o, dim=1)
        return o


class GMNN(pl.LightningModule):
    # TODO lots of calls to softmax after pushing stuff through q, makes it really ugly, try to fix this somewhere

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GMNN")
        parser.add_argument("--hidden_dim", type=int, default=128,
                            help="number of hidden features to use")
        parser.add_argument("--num_gnn_steps", type=int, default=3,
                            help="number of message passing steps to use for p and q")
        parser.add_argument("--pretrain_epochs", type=int, default=25, help="Number of epochs to do for pretraining q")
        parser.add_argument("--iterations", type=int, default=1,
                            help="Number of iterations to train the EM procedure")
        parser.add_argument("--epochs_q", type=int, default=50, help="Number of epochs to train q in every iteration")
        parser.add_argument("--epochs_p", type=int, default=50, help="Number of epochs to train p in every iteration")
        parser.add_argument("--teacher_forcing", type=float, default=0.8, help="Amount of teacher forcing to use")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate to train with")
        parser.add_argument("--in_dropout", type=float, default=0.5, help="Dropout factor on the input")
        parser.add_argument("--dropout", type=float, default=0., help="Dropout after every layer")
        parser.add_argument("--stratified_tf", action='store_true')
        parser.add_argument("--include_features", action='store_true')
        return parent_parser

    def __init__(self, num_h_feats, pretrain_epochs, epochs_q, epochs_p, in_feats=9, dropout=0.5, num_classes=2,
                 lr=0.001, teacher_forcing=0.5, num_steps=2, stratified_tf=False, include_features=True):
        super(GMNN, self).__init__()
        self.save_hyperparameters()
        self.q = GatMNN(num_h_feats, in_feats, dropout=dropout, num_classes=num_classes, num_steps=num_steps)
        self.p = GatMNN(num_h_feats, num_classes + (include_features * in_feats), dropout=dropout,
                        num_classes=num_classes, num_steps=num_steps)
        self.automatic_optimization = False
        if include_features:
            self.aggregation = lambda x, y: torch.cat([x, y], dim=1)
        else:
            self.aggregation = lambda x, y: x

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_q = torch.optim.RMSprop(self.q.parameters(), lr=lr)
        opt_p = torch.optim.RMSprop(self.p.parameters(), lr=lr)
        return opt_q, opt_p

    def forward(self, g, feats):
        out_q = F.softmax(self.q(g, feats), dim=1)
        self.aggregation(out_q, feats)
        out_p = self.p(g, out_q)
        return out_p

    def pretrain_step(self, batch):
        for g, labels in batch:
            opt, _ = self.optimizers()
            features = g.ndata["features"]
            out_q = self.q(g, features)
            weights = self._calculate_weights(labels)
            loss = F.cross_entropy(out_q, labels.long(), weight=weights)
            pred = torch.argmax(out_q, dim=1)
            self.log_results(loss, pred, labels, "q_train")
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

    def iterative_train_step(self, batch):
        for g, labels in batch:
            training_q = ((self.current_epoch - self.hparams.pretrain_epochs) % (
                        self.hparams.epochs_q + self.hparams.epochs_p)) >= self.hparams.epochs_p
            q_opt, p_opt = self.optimizers()
            tf = self.hparams.teacher_forcing
            features = g.ndata["features"]
            weights = self._calculate_weights(labels)
            mask = self.get_teacher_forcing_mask(labels, tf, self.hparams.stratified_tf)
            # Train Q
            if training_q:
                out = self.q(g, features)
                out_q = F.softmax(out, dim=1)
                out_q = self.aggregation(out_q, features)
                out_p = self.p(g, out_q)
                target = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
                target[mask] = F.softmax(out_p[mask], dim=1)
                tag = "q_train"
                #loss = -torch.mean(torch.sum(F.log_softmax(out, dim=-1) * target, dim=-1))
                loss = F.cross_entropy(out, target, weight=weights)
                q_opt.zero_grad()
                self.manual_backward(loss)
                q_opt.step()
            # Train P
            else:
                out_q = F.softmax(self.q(g, features), dim=1).detach()
                target = F.one_hot(labels.long(), num_classes=self.hparams.num_classes).float()
                out_q[mask] = target[mask]
                out_q = self.aggregation(out_q, features)
                out = self.p(g, out_q)
                tag = "p_train"
                loss = F.cross_entropy(out, labels.long(), weight=weights)
                p_opt.zero_grad()
                self.manual_backward(loss)
                p_opt.step()

            pred = torch.argmax(out, dim=1)
            self.log_results(loss, pred, labels, tag)

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.hparams.pretrain_epochs:
            self.pretrain_step(batch)
        else:
            self.iterative_train_step(batch)

    def validation_step(self, batch, batch_idx):
        for g, labels in batch:
            features = g.ndata["features"]
            weights = self._calculate_weights(labels)
            out_q = self.q(g, features)
            pred = torch.argmax(out_q, dim=1)
            q_loss = F.cross_entropy(out_q, labels.long(), weight=weights)
            out_q = F.softmax(out_q, dim=1)
            out_q = self.aggregation(out_q, features)
            out_p = self.p(g, out_q)
            p_loss = F.cross_entropy(out_p, labels.long(), weight=weights)
            if not self.current_epoch < self.hparams.pretrain_epochs:
                pred = torch.argmax(out_p, dim=1)
                self.log_results(p_loss, pred, labels, "p_val")
            else:
                self.log_results(q_loss, pred, labels, "q_val")

    def log_results(self, loss, pred, labels, tag):
        self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False,
                 sync_dist=True, batch_size=1)
        if sum(labels) > 0:
            self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                     on_epoch=True, on_step=False, sync_dist=True, batch_size=1)
        if sum(pred) > 0:
            self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                     on_epoch=True, on_step=False, sync_dist=True, batch_size=1)
        self.log("hp_metric", loss, sync_dist=True, batch_size=1)
        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True, batch_size=1)

    @staticmethod
    def _calculate_weights(labels):
        positives = max(1, torch.sum(labels == 1).detach())
        negatives = max(1, torch.sum(labels == 0).detach())
        frac = negatives / positives
        weight = torch.tensor([1., frac], device=labels.device)
        return weight

    @staticmethod
    def get_teacher_forcing_mask(labels, tf, stratified=True):
        if not stratified:
            mask = np.random.choice([0, 1], size=len(labels), p=[1 - tf, tf])
            mask = torch.from_numpy(mask) == 1
        else:
            mask = torch.zeros(len(labels), dtype=torch.int32)
            for i in range(len(set(labels))):
                mask[labels == i] = torch.from_numpy(
                    np.random.choice([0, 1], size=len(labels[labels == i]), p=[1 - tf, tf]))
            mask = mask == 1
        return mask