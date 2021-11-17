import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.modules import Initializer, GRUCell, GNNCell, LSTMCell
import torch
import abc
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score


class GraphPruner(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ARGNNModel")
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--num_gnn_steps", type=int, default=10)
        parser.add_argument("--num_iterations", type=int, default=10)
        parser.add_argument("--aggregator", type=str, default='lstm')
        parser.add_argument("--memory", type=str, default='none')
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--autoregressive", action="store_true")
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

    @staticmethod
    def parse_arguments(args):
        vnum = f"{args.hidden_dim}_{args.num_gnn_steps}_{args.aggregator}_{args.learning_rate}_{args.sampling_fraction}"
        if args.memory == 'none':
            model = MemorylessPruner(9, args.hidden_dim, args.num_gnn_steps, args.num_iterations, args.aggregator,
                                     args.learning_rate, args.autoregressive)

        else:
            model = MemoryPruner(9, args.hidden_dim, args.num_gnn_steps, args.num_iterations, args.aggregator, args.memory,
                                         args.learning_rate, args.autoregressive)
            vnum += f"_{args.memory}"
        if args.autoregressive:
            vnum += "_AR"
        else:
            vnum += f"_EF_{args.num_iterations}"
        logger = TensorBoardLogger(save_dir="tb_logs", name="GraphPruner")

        return model, logger

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
        self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False)
        if sum(labels) > 0:
            self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                     on_epoch=True, on_step=False)
            self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                     on_epoch=True, on_step=False)

        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)


class MemorylessPruner(GraphPruner):

    def __init__(self, in_feats, h_feats, num_gnn_steps, num_iterations, aggregator='lstm', lr=1e-3,
                 autoregressive=False, memory='none'):
        super(GraphPruner, self).__init__()
        self.save_hyperparameters()
        assert aggregator in ['gru', 'lstm']
        self.agg = aggregator
        self.in_read = Initializer(in_feats, h_feats)
        if self.agg == 'gru':
            self.aggregator = GRUCell(2, h_feats)
        else:
            self.aggregator = LSTMCell(2, h_feats)
        self.gnn = GNNCell(h_feats, num_gnn_steps, 2)

        self.num_iterations = num_iterations
        self.h_feats = h_feats
        self.memory = memory

        self.learning_rate = lr
        self.weight = torch.tensor([1., 70.]).cuda()
        self.autoregressive = autoregressive

    def _step_autoregressive(self, batch, tag):
        weight = self.weight.to(batch[0][0].device)
        loss = 0
        for g, labels in batch:
            output = torch.zeros(g.number_of_nodes(), 2).to(g.device)
            state = torch.zeros(g.number_of_nodes()).to(g.device)
            seq = torch.zeros_like(output)
            h = self.in_read(g.ndata["features"])
            if self.agg == 'lstm':
                c_a = torch.zeros_like(h)
            while sum(state) < g.number_of_nodes():
                unseen = state == 0
                if self.agg == 'lstm':
                    h, c_a = self.aggregator(seq, (h, c_a))
                else:
                    h = self.aggregator(seq, h)
                o, h, l = self.gnn(g, h)
                scores = F.softmax(l[unseen].flatten(), dim=0)
                chosen_node = torch.where(unseen)[0][torch.argmax(scores)]
                state[chosen_node] = 1.
                seq = torch.zeros_like(output)
                seq[chosen_node, 0] = 1.
                seq[chosen_node, 1] = labels[chosen_node]
                loss += (F.cross_entropy(o[unseen], labels[unseen].long(), reduction='none',
                                         weight=weight) * scores).mean()
                output[chosen_node] = o[chosen_node]
            loss = loss / g.number_of_nodes()
            pred = torch.argmax(output, dim=1)
            self.log_results(loss, pred, labels, tag)
        return loss

    def _step_eqfinding(self, batch, tag):
        loss = 0
        self.weight = self.weight.to(batch[0][0].device)
        for g, labels in batch:
            logits = self._forward_eqfinding(g, g.ndata["features"])
            loss += F.cross_entropy(logits, labels.long(), weight=self.weight)
            pred = torch.argmax(logits, dim=1)
            self.log_results(loss, pred, labels, tag)
        return loss

    def _forward_autoregressive(self, g, in_feat):
        pass

    def _forward_eqfinding(self, g, in_feat):
        h = self.in_read(in_feat)
        x = torch.zeros(g.number_of_nodes(), 2).to(g.device)
        if self.agg == 'lstm':
            c = torch.zeros_like(h)
        for i in range(self.num_iterations):
            if self.agg == 'lstm':
                h, c = self.aggregator(x, (h, c))
            else:
                h = self.aggregator(x, h)
            o, h, _ = self.gnn(g, h)
            x = F.softmax(o, dim=1)
        return o


class MemoryPruner(GraphPruner):
    def __init__(self, in_feats, h_feats, num_gnn_steps, num_iterations, aggregator, memory='lstm', lr=1e-3, autoregressive=False):
        super(GraphPruner, self).__init__()
        self.save_hyperparameters()
        assert memory in ['gru', 'lstm']

        self.agg = aggregator
        self.memory_type = memory

        self.in_read = Initializer(in_feats, h_feats)

        if self.agg == 'gru':
            self.aggregator = GRUCell(2, h_feats)
        elif self.agg == 'lstm':
            self.aggregator = LSTMCell(2, h_feats)

        if self.memory_type == 'gru':
            self.mem = GRUCell(h_feats, h_feats)
        elif self.memory_type == 'lstm':
            self.mem = LSTMCell(h_feats, h_feats)

        self.gnn = GNNCell(h_feats, num_gnn_steps, 2)

        self.num_iterations = num_iterations
        self.h_feats = h_feats

        self.learning_rate = lr
        self.weight = torch.tensor([1., 70.]).cuda()
        self.autoregressive = autoregressive

    def _step_autoregressive(self, batch, tag):
        weight = self.weight.to(batch[0][0].device)
        loss = 0
        for g, labels in batch:
            output = torch.zeros(g.number_of_nodes(), 2).to(g.device)
            state = torch.zeros(g.number_of_nodes()).to(g.device)
            seq = torch.zeros_like(output)
            h = self.in_read(g.ndata["features"])
            if self.agg == 'lstm':
                c_a = torch.zeros_like(h)
            mh = torch.zeros_like(h)
            if self.memory_type == "lstm":
                c_m = torch.zeros_like(h)
            while sum(state) < g.number_of_nodes():
                unseen = state == 0
                if self.agg == 'lstm':
                    h, c_a = self.aggregator(seq, (h, c_a))
                else:
                    h = self.aggregator(seq, h)
                if self.memory_type == 'lstm':
                    h, c_m = self.mem(h, (mh, c_m))
                else:
                    h = self.mem(h, mh)
                mh = h
                o, h, l = self.gnn(g, h)
                scores = F.softmax(l[unseen].flatten(), dim=0)
                chosen_node = torch.where(unseen)[0][torch.argmax(scores)]
                state[chosen_node] = 1.
                seq = torch.zeros_like(output)
                seq[chosen_node, 0] = 1.
                seq[chosen_node, 1] = labels[chosen_node]
                loss += (F.cross_entropy(o[unseen], labels[unseen].long(), reduction='none',
                                         weight=weight) * scores).mean()
                output[chosen_node] = o[chosen_node]
            loss = loss / g.number_of_nodes()
            pred = torch.argmax(output, dim=1)
            self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False)
            if sum(labels) > 0:
                self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                         on_epoch=True, on_step=False)
                self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                         on_epoch=True, on_step=False)

        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)
        return loss

    def _step_eqfinding(self, batch, tag):
        loss = 0
        self.weight = self.weight.to(batch[0][0].device)
        for g, labels in batch:
            logits = self._forward_eqfinding(g, g.ndata["features"])
            loss += F.cross_entropy(logits, labels.long(), weight=self.weight)
            pred = torch.argmax(logits, dim=1)
            self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False)
            if sum(labels) > 0:
                self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                         on_epoch=True, on_step=False)
                self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                         on_epoch=True, on_step=False)
        self.log("hp_metric", loss)
        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)
        return loss

    def _forward_autoregressive(self, g, in_feat):
        pass

    def _forward_eqfinding(self, g, in_feat):
        h = self.in_read(in_feat)
        x = torch.zeros(g.number_of_nodes(), 2).to(g.device)
        if self.agg == 'lstm':
            c_a = torch.zeros_like(h)
        mh = torch.zeros_like(h)
        if self.memory_type == "lstm":
            c_m = torch.zeros_like(h)
        for i in range(self.num_iterations):
            if self.agg == 'lstm':
                h, c_a = self.aggregator(x, (h, c_a))
            else:
                h = self.aggregator(x, h)
            if self.memory_type == 'lstm':
                h, c_m = self.mem(h, (mh, c_m))
            else:
                h = self.mem(h, mh)
            mh = h
            o, h, _ = self.gnn(g, h)
            x = F.softmax(o, dim=1)
        return o


