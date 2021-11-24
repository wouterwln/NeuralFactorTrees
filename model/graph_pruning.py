import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.modules import *
import torch
import abc
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score


class GraphPruner(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphPruningModel")
        parser.add_argument("--hidden_dim", type=int, default=128, help="number of hidden features to use (for node embeddings and gating mechanisms)")
        parser.add_argument("--num_gnn_steps", type=int, default=3, help="number of GNN layers used in every iteration")
        parser.add_argument("--num_iterations", type=int, default=5, help="number of iterations used for equilibrium finding modules")
        parser.add_argument("--aggregator", type=str, default='lstm', help="aggregator gating mechanism to use (either lstm or gru)")
        parser.add_argument("--memory", type=str, default='none', help="memory type to use (gru, lstm or none) to add memory of node embeddings")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate to train with")
        parser.add_argument("--autoregressive", action="store_true", help="use this flag to train autoregressive model instead of equilibrium finding")
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        self.step(val_batch, "val")

    def forward(self, g, in_feat):
        if self.autoregressive:
            return self._step_autoregressive(g, in_feat)
        else:
            return self._step_eqfinding(g, in_feat)

    def step(self, batch, tag):
        if self.autoregressive:
            return self._step_autoregressive(batch, tag)
        else:
            return self._step_eqfinding(batch, tag)

    @abc.abstractmethod
    def _step_autoregressive(self, batch, tag):
        pass

    @abc.abstractmethod
    def _step_eqfinding(self, batch, tag):
        pass

    @abc.abstractmethod
    def _forward_autoregressive(self, g, in_feat):
        pass

    @abc.abstractmethod
    def _forward_eqfinding(self, g, in_feat):
        pass

    def log_results(self, loss, pred, labels, tag):
        self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False, sync_dist=True, batch_size=1)
        if sum(labels) > 0:
            self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                     on_epoch=True, on_step=False, sync_dist=True, batch_size=1)
            self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                     on_epoch=True, on_step=False, sync_dist=True, batch_size=1)
        self.log("hp_metric", loss, sync_dist=True, batch_size=1)
        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=False, on_step=False, sync_dist=True, batch_size=1)


class GNNPruner(GraphPruner):
    def __init__(self, in_feats, h_feats, num_gnn_steps, num_iterations, aggregator='lstm', lr=1e-3,
                 autoregressive=False, memory='none'):
        super(GraphPruner, self).__init__()
        self.save_hyperparameters()
        assert aggregator in ['gru', 'lstm']
        self.agg = aggregator
        self.memory_type = memory
        self.in_read = Initializer(in_feats, h_feats)

        if self.agg == 'gru':
            self.aggregator = GRUCell(2, h_feats)
        else:
            self.aggregator = LSTMCell(2, h_feats)

        if self.memory_type == 'gru':
            self.mem = GRUCell(h_feats, h_feats)
        elif self.memory_type == 'lstm':
            self.mem = LSTMCell(h_feats, h_feats)
        self.gnn = GNNCell(h_feats, num_gnn_steps, 2)

        self.num_iterations = num_iterations
        self.h_feats = h_feats

        self.learning_rate = lr
        self.autoregressive = autoregressive


    def _step_autoregressive(self, batch, tag):
        loss = 0
        for g, labels in batch:
            output = torch.zeros(g.number_of_nodes(), 2).to(g.device)
            state = torch.zeros(g.number_of_nodes()).to(g.device)
            seq = torch.zeros_like(output)
            h = self.in_read(g.ndata["features"])
            if self.agg == 'lstm':
                c_a = torch.zeros_like(h)
            if self.memory_type != 'none':
                mh = torch.zeros_like(h)
                if self.memory_type == "lstm":
                    c_m = torch.zeros_like(h)
            while sum(state) < g.number_of_nodes():
                unseen = state == 0
                if self.agg == 'lstm':
                    h, c_a = self.aggregator(seq, (h, c_a))
                else:
                    h = self.aggregator(seq, h)
                if self.memory_type != 'none':
                    if self.memory_type == 'lstm':
                        h, c_m = self.mem(h, (mh, c_m))
                    elif self.memory_type == 'gru':
                        h = self.mem(h, mh)
                    mh = h
                o, h, l = self.gnn(g, h)
                scores = F.softmax(l[unseen].flatten(), dim=0)
                chosen_node = torch.where(unseen)[0][torch.argmax(scores)]
                state[chosen_node] = 1.
                seq = torch.zeros_like(output)
                seq[chosen_node, 0] = 1.
                seq[chosen_node, 1] = labels[chosen_node]
                weight = self._calculate_weights(labels[unseen])
                loss += torch.sum(F.cross_entropy(o[unseen], labels[unseen].long(), reduction='none',
                                         weight=weight) * scores)
                output[chosen_node] = o[chosen_node]
            loss = loss / g.number_of_nodes()
            pred = torch.argmax(output, dim=1)
            self.log_results(loss, pred, labels, tag)
        return loss

    def _step_eqfinding(self, batch, tag):
        loss = 0
        for g, labels in batch:
            logits = self._forward_eqfinding(g, g.ndata["features"])
            weight = self._calculate_weights(labels)
            loss += F.cross_entropy(logits, labels.long(), weight=weight)
            pred = torch.argmax(logits, dim=1)
            self.log_results(loss, pred, labels, tag)
        return loss

    def _forward_autoregressive(self, g, in_feat):
        pass

    def _forward_eqfinding(self, g, in_feat):
        h = self.in_read(in_feat)
        x = torch.zeros(g.number_of_nodes(), 2).to(g.device)
        if self.agg == 'lstm':
            c_a = torch.zeros_like(h)
        if self.memory_type != 'none':
            mh = torch.zeros_like(h)
            if self.memory_type == "lstm":
                c_m = torch.zeros_like(h)
        for i in range(self.num_iterations):
            if self.agg == 'lstm':
                h, c_a = self.aggregator(x, (h, c_a))
            else:
                h = self.aggregator(x, h)
            if self.memory_type != 'none':
                if self.memory_type == 'lstm':
                    h, c_m = self.mem(h, (mh, c_m))
                else:
                    h = self.mem(h, mh)
                mh = h
            o, h, _ = self.gnn(g, h)
            x = F.softmax(o, dim=1)
        return o

    @staticmethod
    def _calculate_weights(labels):
        positives = max(1, torch.sum(labels == 1).detach())
        negatives = max(1, torch.sum(labels == 0).detach())
        frac = negatives / positives
        weight = torch.tensor([1., frac], device=labels.device)
        return weight