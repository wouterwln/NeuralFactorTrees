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


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.xh = Linear(input_size, hidden_size * 4)
        self.hh = Linear(hidden_size, hidden_size * 4)

    def forward(self, input, hx=None):
        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size)
        #       cy: of shape (batch_size, hidden_size)

        hx, cx = hx

        gates = self.xh(input) + self.hh(hx)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)

        return (hy, cy)


class GNNCell(Module):
    def __init__(self, h_feats, num_iterations, num_outputs):
        super(GNNCell, self).__init__()
        self.convBlock = Sequential(
            *[SAGEConv(h_feats, h_feats, 'lstm', activation=ReLU()) for _ in range(num_iterations)])
        self.lin1 = Linear(h_feats, num_outputs)
        self.lin2 = Linear(h_feats, 1)

    def forward(self, g, h):
        h = self.convBlock(g, h)
        o = self.lin1(h)
        l = self.lin2(h)
        return o, h, l

