import torch

from torch.nn import functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout, Module
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score
from dgl.nn import SAGEConv, GatedGraphConv
from dgl.nn.pytorch.utils import Sequential
from dataloader import TracksterDataset

import pytorch_lightning as pl


class Initializer(Module):
    def __init__(self, in_feats, h_feats):
        super(Initializer, self).__init__()
        self.lin1 = Linear(in_feats, h_feats // 2)
        self.lin2 = Linear(h_feats // 2, h_feats)
        self.lin3 = Linear(h_feats, h_feats)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x

class GRUCell(Module):
    def __init__(self, input_size, h_feats):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.h_feats = h_feats


        self.x2h = Linear(input_size, 3 * h_feats)
        self.h2h = Linear(h_feats, 3 * h_feats)


    def forward(self, input, hx=None):
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy

class GNNCell(Module):
    def __init__(self, h_feats, num_iterations, num_outputs):
        super(GNNCell, self).__init__()
        self.convBlock = Sequential(*[SAGEConv(h_feats, h_feats, 'lstm', activation=ReLU()) for _ in range(num_iterations)])
        self.lin = Linear(h_feats, num_outputs)

    def forward(self, g, h):
        h = self.convBlock(g, h)
        o = self.lin(h)
        return o, h

class ARGraphPruningModule(pl.LightningModule):
    def __init__(self, in_feats, h_feats, num_outputs, num_gnn_steps, num_iterations):
        super(ARGraphPruningModule, self).__init__()
        self.in_read = Initializer(in_feats, h_feats)
        self.gru = GRUCell(num_outputs, h_feats)
        self.gnn = GNNCell(h_feats, num_gnn_steps, num_outputs)

        self.num_iterations = num_iterations
        self.num_outputs = num_outputs
        self.h_feats = h_feats

        self.learning_rate = 0.001
        self.weight = torch.tensor([1., 50.]).cuda()

    def forward(self, g, in_feat):
        h = self.in_read(in_feat)
        x = torch.zeros(g.number_of_nodes(), 2).to(g.device)
        for i in range(self.num_iterations):
            h = self.gru(x, h)
            o, h = self.gnn(g, h)
            x = F.softmax(o, dim=1)
        return o

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        self.step(val_batch, "val")

    def step(self, batch, tag):
        loss = 0
        for g, labels in batch:
            logits = self(g, g.ndata["features"])
            loss += F.cross_entropy(logits, labels.long(), weight=self.weight)
            pred = torch.argmax(logits, dim=1)
            self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False)
            if sum(labels) > 0:
                self.log(f"{tag}_precision", precision_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False, on_epoch=True, on_step=False)
                self.log(f"{tag}_recall", recall_score(labels.cpu(), pred.cpu(), zero_division=0), prog_bar=False, on_epoch=True, on_step=False)

        self.log(f"{tag}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)
        return loss