import torch
import dgl

from torch.nn import functional as F
from torch import nn
from dgl.nn.pytorch import GMMConv, GatedGraphConv, GATConv, TAGConv
from dgl.nn.pytorch.utils import Sequential

class MultiLayeredGatedGraphConv(GatedGraphConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 n_layers,
                 n_etypes=1,
                 bias=True,
                 dropout=0.):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(out_feats, out_feats))
            layers.append(nn.Dropout(dropout))
        self.linears = nn.ModuleList(
            [nn.Sequential(*layers) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = nn.init.calculate_gain('relu')
        self.gru.reset_parameters()
        for l in self.linears:
            for linear in l:
                if isinstance(linear, nn.Linear):
                    nn.init.xavier_normal_(linear.weight, gain=gain)
                    nn.init.zeros_(linear.bias)

class GATBackbone(nn.Module):
    def __init__(self, num_h_feats, in_feats=9, in_dropout=0., dropout=0., num_classes=2, num_heads=8, num_steps=2):
        super(GATBackbone, self).__init__()
        assert num_steps >= 2
        self.conv1 = GATConv(in_feats, num_h_feats, num_heads, feat_drop=dropout)
        self.convBlock = Sequential(
            *[GATConv(num_h_feats, num_h_feats, num_heads, feat_drop=dropout) for _ in range(num_steps - 2)])
        self.conv2 = GATConv(num_h_feats, num_classes, num_heads)
        self.dropout1 = nn.Dropout(in_dropout)
        self.batchnorm = nn.BatchNorm1d(num_h_feats)

    def forward(self, g, feats):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, self.dropout1(feats))
        h = torch.sum(h, dim=1)
        h = self.batchnorm(h)
        for layer in self.convBlock:
            h = layer(g, h)
            h = torch.sum(h, dim=1)
            h = self.batchnorm(h)
        o = self.conv2(g, h)
        o = torch.sum(o, dim=1)
        return o

class GeneralBackbone(nn.Module):
    def __init__(self, num_h_feats, in_feats=9, in_dropout=0., dropout=0., num_steps=2, convoperator='tag', num_layers=3):
        super(GeneralBackbone, self).__init__()
        if convoperator == 'tag':
            self.convBlock = TAGConv(in_feats, num_h_feats, num_steps)
        elif convoperator == "ggsnn":
            self.convBlock = MultiLayeredGatedGraphConv(in_feats, num_h_feats, num_steps, n_etypes=1, n_layers=3)

    def forward(self, g, feats):
        h = self.convBlock(g, feats)
        return h

class GMMBackbone(nn.Module):
    def __init__(self, num_h_feats, in_feats=9, in_dropout=0., dropout=0., num_steps=2):
        super(GMMBackbone, self).__init__()
        in_feats = in_feats - 3
        self.convBlock = [GMMConv(in_feats, num_h_feats, 3, 2, 'mean')]
        self.convBlock += [GMMConv(num_h_feats, num_h_feats, 3, 2, 'mean') for i in range(num_steps - 1)]
        self.convBlock = Sequential(*self.convBlock)
        pass

    def forward(self, g, feats):
        g = dgl.add_self_loop(g)
        g.apply_edges(self.edge_func)
        feats = feats[:, 3:]
        for layer in self.convBlock:
            feats = layer(g, feats, g.edata["coords"])
        return feats

    @staticmethod
    def edge_func(edges):
        coords = edges.dst["x"][:, :3] - edges.src["x"][:, :3]
        return {"coords" : coords}
