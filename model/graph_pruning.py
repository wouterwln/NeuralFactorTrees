from copy import deepcopy

import dgl.nn.pytorch
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from dgl.nn.pytorch import SAGEConv, GraphConv, GATConv, Sequential
import numpy as np
import dgl.function as fn
from model.modules import GATBackbone, GMMBackbone, GeneralBackbone
import collections


class TIGMN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TIGMN")
        parser.add_argument("--hidden_dim", type=int, default=128,
                            help="number of hidden features to use")
        parser.add_argument("--num_gnn_steps", type=int, default=3,
                            help="number of message passing steps to use for p and q")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train q in every iteration")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate to train with")
        parser.add_argument("--dropout", type=float, default=0., help="Dropout after every layer")
        parser.add_argument("--backbone", type=str, default="ggsnn", help="Backbone convolution to use")
        return parent_parser

    def __init__(self, in_feats=9, num_h_feats=128, dropout=0.1, num_classes=2, lr=0.001, num_steps=2,
                 backbone='ggsnn'):
        super(TIGMN, self).__init__()
        self.save_hyperparameters()
        assert backbone in ['gat', 'ggsnn', 'tag', 'gmm']
        if backbone == 'gat':
            self.e = GATBackbone(num_h_feats, in_feats, dropout, num_classes=num_h_feats, num_steps=num_steps)
        elif backbone == 'gmm':
            self.e = GMMBackbone(num_h_feats, in_feats, num_steps=num_steps)
        else:
            self.e = GeneralBackbone(num_h_feats, in_feats, num_steps=num_steps, convoperator=backbone)
        self.dropout = nn.Dropout(dropout)
        self.edge_generator = nn.Linear(num_h_feats * 2, 2 * num_classes)
        self.node_generator = nn.Linear(num_h_feats, num_classes)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return opt

    def forward(self, g):
        feats = g.ndata["x"]
        prop_graph = dgl.add_reverse_edges(g)
        h = self.e(prop_graph, feats)
        h = self.dropout(h)
        g.ndata["feat"] = h
        g = self.generate_edge_factors(g)
        g.ndata["out"] = self.node_generator(h)
        return g

    def step(self, batch, tag):
        labels = batch.ndata["y"]
        pgm = self(batch)
        likelihood = self.sum_likelihood(pgm, labels) - self.partition_function(pgm)
        likelihood = likelihood / len(labels)
        loss = -likelihood

        self.log(f"{tag}_log_likelihood", likelihood, sync_dist=True)
        self.log(f"{tag}_loss", loss, sync_dist=True)
        if tag == "val":
            self.log("hp_metric", likelihood, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        labels = batch.ndata["y"]
        pgm = self(batch)
        likelihood = self.sum_likelihood(pgm, labels) - self.partition_function(pgm)
        likelihood = likelihood / len(labels)
        loss = -likelihood
        self.log("test_loss", loss)
        self.log("test_likelihood", likelihood)
        samples = self.sample(pgm, 1000)
        pgm.ndata["sample"] = samples.T
        for g, trackster in zip(dgl.unbatch(batch), dgl.unbatch(pgm)):
            labels = g.ndata["y"]
            samples = trackster.ndata["sample"].T
            samples_strings = [tuple(row) for row in samples.tolist()]
            counter = collections.Counter(samples_strings)
            top_20 = counter.most_common(20)
            label_str = ''.join(str(item) for item in labels.tolist())
            occurrence = 21
            for n in range(min(20, len(top_20))):
                sample_str = ''.join(str(item) for item in top_20[n][0])
                if sample_str == label_str:
                    occurrence = n
                    break
            for n in range(20):
                self.log(f"top-{n + 1}-accuracy", int(occurrence <= n))
                if torch.sum(labels) > 0:
                    self.log(f"prunable-top-{n + 1}-accuracy", int(occurrence <= n))
            if torch.sum(labels) > 0.05 * len(labels):
                subgraphs = g.subgraph(labels.bool())
                if len(dgl.topological_nodes_generator(subgraphs)) > 2:
                    for n in range(20):
                        self.log(f"clustered-top-{n + 1}-accuracy", int(occurrence <= n))




    def generate_edge_factors(self, g):
        g.apply_edges(self.concat_message_function)
        g.edata['out'] = self.edge_generator(g.edata['cat_feat'])
        return g

    @staticmethod
    def partition_function(pgm):
        order = dgl.topological_nodes_generator(pgm)
        for nodes in order:
            nodes = nodes.to(pgm.device)
            pgm.pull(nodes, TIGMN.send_message, fn.sum('factor', 'fct'))
            pgm.apply_nodes(lambda x: {'out': x.data['out'] + x.data['fct']}, nodes)
        roots = dgl.topological_nodes_generator(pgm, reverse=True)[0].to(pgm.device)
        return torch.sum(torch.logsumexp(pgm.ndata["out"][roots], dim=-1))

    @staticmethod
    def sum_likelihood(pgm, labels):
        pgm.apply_nodes(lambda nodes: {"c": nodes.data["out"][F.one_hot(labels, num_classes=2).bool()]})
        pgm.apply_edges(TIGMN.edge_output)
        return torch.sum(pgm.ndata["c"]) + torch.sum(pgm.edata["c"])

    @staticmethod
    def sample(pgm, steps):
        o, t = pgm.edges()
        order = dgl.topological_nodes_generator(pgm)
        pgm = dgl.add_edges(pgm, t, o,
                            data={"out": torch.transpose(pgm.edata["out"].reshape((-1, 2, 2)), 1, 2).reshape((-1, 4))})
        pgm.pull(pgm.nodes(), TIGMN.receive_sample_init_msg, fn.sum('factor', 'fct'))
        pgm.apply_nodes(lambda x: {'factor': x.data['out'] + x.data['fct']}, pgm.nodes())
        pgm.apply_nodes(lambda x: {"label": F.gumbel_softmax(x.data["factor"], dim=-1, hard=True)})
        samples = torch.zeros((steps - 10, pgm.number_of_nodes()), device=pgm.device, dtype=torch.uint8)
        for i in range(steps):
            for nodes in order:
                nodes = nodes.to(pgm.device)
                pgm.pull(nodes, TIGMN.receive_sample_loopy_msg, fn.sum('factor', 'fct'))
                pgm.apply_nodes(lambda x: {'factor': x.data['out'] + x.data['fct']}, pgm.nodes())
                pgm.apply_nodes(lambda x: {"label": F.gumbel_softmax(x.data["factor"], dim=-1, hard=True)})
            if i >= 10:
                samples[i - 10] = torch.argmax(pgm.ndata["label"], dim=-1)
        return samples

    @staticmethod
    def edge_output(edges):
        labels = F.one_hot(edges.src["y"] + 2 * edges.dst["y"], num_classes=4).bool()
        return {"c": edges.data["out"][labels]}

    @staticmethod
    def send_message(edges):
        messages = edges.data["out"].reshape((-1, 2, 2)) + edges.src["out"].unsqueeze(1)
        messages = torch.logsumexp(messages, dim=-1)
        return {"factor": messages}

    @staticmethod
    def concat_message_function(edges):
        return {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}

    @staticmethod
    def receive_sample_init_msg(edges):
        messages = edges.data["out"].reshape((-1, 2, 2))
        messages = torch.logsumexp(messages, dim=-1)
        return {"factor": messages}

    @staticmethod
    def receive_sample_loopy_msg(edges):
        return {"factor": torch.transpose(edges.data["out"].reshape((-1, 2, 2)), 1, 2)[edges.src["label"].bool()]}


class GMNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GMNN")
        parser.add_argument("--include_features", action='store_true')
        return parent_parser

    def __init__(self, num_h_feats, in_feats=3, dropout=0.5, num_classes=2, lr=0.001, num_steps=2,
                 include_features=True, epochs=100):
        super(GMNN, self).__init__()
        self.save_hyperparameters()
        self.q = GATBackbone(num_h_feats, in_feats, dropout=dropout, num_classes=num_classes, num_steps=num_steps)
        self.p = GATBackbone(num_h_feats, num_classes + (include_features * in_feats), dropout=dropout,
                             num_classes=num_classes, num_steps=num_steps)
        if include_features:
            self.aggregation = lambda x, y: torch.cat([x, y], dim=1)
        else:
            self.aggregation = lambda x, y: x
        self.automatic_optimization = False

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_q = torch.optim.RMSprop(self.q.parameters(), lr=lr)
        opt_p = torch.optim.RMSprop(self.p.parameters(), lr=lr)
        return opt_q, opt_p

    def forward(self, g, feats):
        out_q = self.q(g, feats)
        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, feats)
        out_p = self.p(g, in_p)
        return out_p

    def step(self, g, tag):
        features = g.ndata["x"]
        labels = g.ndata["y"]
        out_q = self.q(g, features)
        # weights = self._calculate_weights(labels)
        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        q_loss = F.cross_entropy(out_q, labels)
        pred = torch.argmax(F.softmax(out_q, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(q_loss.detach()), pred, deepcopy(labels), f"q_{tag}")
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p)
        p_loss = F.cross_entropy(out_p, labels)
        pred = torch.argmax(F.softmax(out_p, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(p_loss.detach()), pred, deepcopy(labels), f"p_{tag}")
        loss = q_loss + p_loss
        return loss

    def q_step(self, g, tag):
        opt, _ = self.optimizers()
        features = g.ndata["x"]
        labels = g.ndata["y"]
        out_q = self.q(g, features)
        # weights = self._calculate_weights(labels)
        loss = F.cross_entropy(out_q, labels)
        if tag == "train":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        pred = torch.argmax(F.softmax(out_q, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(loss.detach()), pred, deepcopy(labels), f"q_{tag}")

    def p_step(self, g, tag):
        _, opt = self.optimizers()
        features = g.ndata["x"]
        labels = g.ndata["y"]
        out_q = self.q(g, features)
        # weights = self._calculate_weights(labels)

        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p)

        loss = F.cross_entropy(out_p, labels)
        if tag == "train":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        pred = torch.argmax(F.softmax(out_p, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(loss.detach()), pred, deepcopy(labels), f"p_{tag}")

    def training_step(self, batch, batch_idx):
        batch = dgl.add_self_loop(dgl.add_reverse_edges(batch))
        if self.current_epoch < self.hparams.epochs // 2:
            self.q_step(batch, "train")
        else:
            self.p_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        batch = dgl.add_self_loop(dgl.add_reverse_edges(batch))
        if self.current_epoch < self.hparams.epochs // 2:
            self.q_step(batch, "val")
        else:
            self.p_step(batch, "val")


    def test_step(self, batch, batch_idx):
        g = dgl.add_self_loop(dgl.add_reverse_edges(batch))
        _, opt = self.optimizers()
        features = g.ndata["x"]
        out_q = self.q(g, features)

        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p)
        samples = torch.zeros((1000, g.number_of_nodes()), dtype=torch.uint8, device=batch.device)
        for i in range(1000):
            samples[i] = torch.argmax(F.gumbel_softmax(out_p, hard=True), dim=-1)
        batch.ndata["sample"] = samples.T
        for graph in dgl.unbatch(batch):
            labels = graph.ndata["y"]
            samples = graph.ndata["sample"].T
            samples_strings = [tuple(row) for row in samples.tolist()]
            counter = collections.Counter(samples_strings)
            top_20 = counter.most_common(20)
            label_str = ''.join(str(item) for item in labels.tolist())
            occurrence = 21
            for n in range(min(20, len(top_20))):
                sample_str = ''.join(str(item) for item in top_20[n][0])
                if sample_str == label_str:
                    occurrence = n
                    break
            for n in range(20):
                self.log(f"top-{n + 1}-accuracy", int(occurrence <= n))
                if torch.sum(labels) > 0:
                    self.log(f"prunable-top-{n + 1}-accuracy", int(occurrence <= n))
            if torch.sum(labels) > 0.05 * len(labels):
                subgraphs = graph.subgraph(labels.bool())
                if len(dgl.topological_nodes_generator(subgraphs)) > 2:
                    for n in range(20):
                        self.log(f"clustered-top-{n + 1}-accuracy", int(occurrence <= n))


    def log_results(self, loss, pred, labels, tag):
        self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False,
                 sync_dist=True, batch_size=1)
        self.log("hp_metric", -loss, sync_dist=True, batch_size=1)
        self.log(f"{tag}_log_likelihood", -loss, sync_dist=True)
        self.log(f"{tag}_loss", loss, prog_bar=True, sync_dist=True, batch_size=1)
