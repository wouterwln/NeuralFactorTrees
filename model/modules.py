import torch

from torch.nn import functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout, Module
from dgl.nn import SAGEConv, GatedGraphConv
from dgl.nn.pytorch.utils import Sequential



class Initializer(Module):
    def __init__(self, in_feats, h_feats):
        super(Initializer, self).__init__()
        self.lin1 = Linear(in_feats, h_feats // 2)
        self.batchnorm1 = BatchNorm1d(h_feats // 2)
        self.lin2 = Linear(h_feats // 2, h_feats)
        self.batchnorm2 = BatchNorm1d(h_feats)
        self.lin3 = Linear(h_feats, h_feats)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.lin2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.lin3(x))
        return x


class GNNInitializer(Module):
    def __init__(self, in_feats, h_feats, num_iterations):
        super(GNNInitializer, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'lstm',activation=ReLU())
        self.convBlock = Sequential(
            *[SAGEConv(h_feats, h_feats, 'lstm', activation=ReLU()) for _ in range(num_iterations)])

    def forward(self, g, in_feats):
        return self.convBlock(g, self.conv1(g, in_feats))

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
        self.cls = [torch.nn.Sequential(Linear(h_feats, h_feats), Dropout(0.2), ReLU()) for _ in range(5)]
        self.cls += [Linear(h_feats, num_outputs)]
        self.cls = torch.nn.Sequential(*self.cls)
        self.cnf = [torch.nn.Sequential(Linear(h_feats, h_feats), Dropout(0.2), ReLU()) for _ in range(5)]
        self.cnf += [Linear(h_feats, 1)]
        self.cnf = torch.nn.Sequential(*self.cnf)

    def forward(self, g, h):
        h = self.convBlock(g, h)
        o = self.cls(h)
        l = self.cnf(h)
        return o, h, l

