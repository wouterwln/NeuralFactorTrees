import torch
import dgl

from torch.nn import functional as F
from torch import nn
from dgl.nn.pytorch import GMMConv, GatedGraphConv, GATConv, TAGConv
from dgl.nn.pytorch.utils import Sequential

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
    def __init__(self, num_h_feats, in_feats=9, in_dropout=0., dropout=0., num_steps=2, convoperator='tag'):
        super(GeneralBackbone, self).__init__()
        if convoperator == 'tag':
            self.convBlock = TAGConv(in_feats, num_h_feats, num_steps)
        elif convoperator == "ggsnn":
            self.convBlock = GatedGraphConv(in_feats, num_h_feats, num_steps, 1)

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
