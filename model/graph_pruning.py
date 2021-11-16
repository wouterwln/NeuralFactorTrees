import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.modules import Initializer, GRUCell, GNNCell, LSTMCell
import torch
import abc
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score


class GraphPruner(pl.LightningModule):
    def __init__(self, in_feats, h_feats, num_gnn_steps, num_iterations, memory='lstm', lr=1e-3):
        super(GraphPruner, self).__init__()
        self.save_hyperparameters()
        assert memory in ['gru', 'lstm', 'none']
        self.memory_type = memory
        self.in_read = Initializer(in_feats, h_feats)
        self.gru = GRUCell(2, h_feats)
        if self.memory_type == 'gru':
            self.mem = GRUCell(h_feats, h_feats)
        elif self.memory_type == 'lstm':
            self.mem = LSTMCell(h_feats, h_feats)
        else:
            self.mem = lambda x, y: x
        self.gnn = GNNCell(h_feats, num_gnn_steps, 2)

        self.num_iterations = num_iterations
        self.h_feats = h_feats

        self.learning_rate = lr
        self.weight = torch.tensor([1., 70.]).cuda()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ARGNNModel")
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--num_gnn_steps", type=int, default=8)
        parser.add_argument("--num_iterations", type=int, default=10)
        parser.add_argument("--memory", type=str, default='gru')
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--autoregressive", action="store_true")
        return parent_parser

    @abc.abstractmethod
    def forward(self, g, in_feat):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        self.step(val_batch, "val")

    @abc.abstractmethod
    def step(self, batch, tag):
        pass

    @staticmethod
    def parse_arguments(args):
        if args.autoregressive:
            model = ARGraphPruningModule(9, args.hidden_dim, args.num_gnn_steps, args.num_iterations, args.memory, args.learning_rate)
            logger = TensorBoardLogger(save_dir="tb_logs",
                                   version=f"AR_{args.hidden_dim}_{args.num_gnn_steps}_{args.memory}_{args.learning_rate}_{args.sampling_fraction}")
        else:
            model = EFGraphPruningModule(9, args.hidden_dim, args.num_gnn_steps, args.num_iterations, args.memory, args.learning_rate)
            logger = TensorBoardLogger(save_dir="tb_logs",
                                       version=f"EF_{args.hidden_dim}_{args.num_gnn_steps}_{args.num_iterations}_{args.memory}_{args.learning_rate}_{args.sampling_fraction}")
        return model, logger

class ARGraphPruningModule(GraphPruner):

    def forward(self, g, in_feat):
        pass

    def step(self, batch, tag):
        weight = self.weight.to(batch[0][0].device)
        loss = 0
        for g, labels in batch:
            output = torch.zeros(g.number_of_nodes(), 2).to(g.device)
            state = torch.zeros(g.number_of_nodes()).to(g.device)
            seq = torch.zeros_like(output)
            h = self.in_read(g.ndata["features"])
            mh = torch.zeros_like(h)
            while sum(state) < g.number_of_nodes():
                unseen = state == 0
                h = self.gru(seq, h)
                h = self.mem(h, mh)
                mh = h
                if self.memory_type == 'lstm':
                    h = h[0]
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


class EFGraphPruningModule(GraphPruner):
    def forward(self, g, in_feat):
        h = self.in_read(in_feat)
        x = torch.zeros(g.number_of_nodes(), 2).to(g.device)
        mh = torch.zeros_like(h)
        if self.memory_type == 'lstm':
            mh = (mh, mh)
        for i in range(self.num_iterations):
            h = self.gru(x, h)
            h = self.mem(h, mh)
            mh = h
            if self.memory_type == 'lstm':
                h = h[0]
            o, h, _ = self.gnn(g, h)
            x = F.softmax(o, dim=1)
        return o


    def step(self, batch, tag):
        loss = 0
        self.weight = self.weight.to(batch[0][0].device)
        for g, labels in batch:
            logits = self(g, g.ndata["features"])
            loss += F.cross_entropy(logits, labels.long(), weight=self.weight)
            pred = torch.argmax(logits, dim=1)
            self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False)
            if sum(labels) > 0:
                self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                         on_epoch=True, on_step=False)
                self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False,
                         on_epoch=True, on_step=False)

        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)
        return loss

