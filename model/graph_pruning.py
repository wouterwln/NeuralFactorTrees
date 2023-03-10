import math
from copy import deepcopy

import dgl.nn.pytorch
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
import dgl.function as fn
from model.modules import GATBackbone, GMMBackbone, GeneralBackbone, MultiLayeredGatedGraphConv
import collections
from dgl.nn.pytorch import GatedGraphConv
from tqdm import trange


class NeuralMarkovTree(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NeuralMarkovTree")
        parser.add_argument("--hidden_dim", type=int, default=128,
                            help="number of hidden features to use")
        parser.add_argument("--num_gnn_steps", type=int, default=8,
                            help="number of message passing steps to use for p and q")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
        parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate to train with")
        parser.add_argument("--dropout", type=float, default=0.1, help="Dropout after every layer")
        parser.add_argument("--backbone", type=str, default="ggsnn", help="Backbone convolution to use")
        parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in MLP to compute messages")
        return parent_parser

    def __init__(self, in_feats=9, num_h_feats=128, dropout=0.1, num_classes=2, lr=0.001, num_steps=2,
                 backbone='ggsnn', num_layers=3):
        super(NeuralMarkovTree, self).__init__()
        self.save_hyperparameters()
        assert backbone in ['gat', 'ggsnn', 'tag', 'gmm']
        if backbone == 'gat':
            self.e = GATBackbone(num_h_feats, in_feats, dropout, num_classes=num_h_feats, num_steps=num_steps)
        elif backbone == 'gmm':
            self.e = GMMBackbone(num_h_feats, in_feats, num_steps=num_steps)
        else:
            self.e = GeneralBackbone(num_h_feats, in_feats, num_steps=num_steps, convoperator=backbone,
                                     num_layers=num_layers)
        if backbone != 'gat':
            self.dropout = nn.Dropout(dropout)
        self.edge_generator = nn.Linear(num_h_feats * 2, 2 * num_classes)
        self.node_generator = nn.Linear(num_h_feats, num_classes)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        return opt

    def forward(self, batch):
        g, prop_graph, _ = batch
        feats = g.ndata["x"]
        h = self.e(prop_graph, feats)
        if self.hparams.backbone != 'gat':
            h = self.dropout(h)
        g.ndata["feat"] = h
        g = self.generate_edge_factors(g)
        g.ndata["out"] = self.node_generator(h)
        return g

    def step(self, batch, tag):
        labels = batch[0].ndata["y"]
        pgm = self(batch)
        likelihood = self.sum_likelihood(pgm, labels) - self.partition_function(pgm)
        loss = -likelihood
        likelihood = likelihood / len(labels)

        self.log(f"{tag}_log_likelihood", likelihood, sync_dist=True)
        self.log(f"{tag}_loss", -likelihood, sync_dist=True)
        if tag == "val":
            self.log("hp_metric", likelihood, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        labels = batch[0].ndata["y"]
        pgm = self(batch)
        likelihood = self.sum_likelihood(pgm, labels) - self.partition_function(pgm)
        likelihood = likelihood / len(labels)
        loss = -likelihood
        self.log("test_loss", loss)
        self.log("test_likelihood", likelihood)
        samples = self.sample(pgm, 110, 100, 10)
        pgm.ndata["sample"] = samples.T
        batch = batch[0]
        for g, trackster in zip(dgl.unbatch(batch), dgl.unbatch(pgm)):
            labels = g.ndata["y"]
            labels_bool = labels.bool()
            energies = g.ndata["x"][:, 3]
            samples = trackster.ndata["sample"].T
            samples_strings = [tuple(row) for row in samples.tolist()]
            counter = collections.Counter(samples_strings)
            top_20 = counter.most_common(20)
            for i, entry in enumerate(top_20):
                self.log(f"top-{i + 1}-mass", entry[1] / 100)
            label_str = ''.join(str(item) for item in labels.tolist())
            occurrence = 21
            for n in range(min(20, len(top_20))):
                sample_str = ''.join(str(item) for item in top_20[n][0])
                if sample_str == label_str:
                    occurrence = n
                    break
            for n in range(20):
                self.log(f"top-{n + 1}-accuracy", int(occurrence <= n))
                self.log(f"top-{n + 1}-{len(labels)}-accuracy", int(occurrence <= n))
                if torch.sum(labels) > 0:
                    self.log(f"prunable-top-{n + 1}-accuracy", int(occurrence <= n))
                    self.log(f"prunable-top-{n + 1}-{len(labels)}-accuracy", int(occurrence <= n))
            for n in range(20):
                try:
                    accuracy = torch.sum(torch.tensor(top_20[n][0], device=labels.device) == labels) / len(labels)
                    self.log(f"node_wise_accuracy_{n + 1}", accuracy)
                    self.log(f"node_wise_accuracy_{n + 1}_{len(labels)}", accuracy)
                    frac_pruned = torch.ceil(20 * torch.mean(labels.float())) / 20
                    self.log(f"node_wise_accuracy_{n + 1}_fraction_{frac_pruned}", accuracy)
                except:
                    continue
                try:
                    prediction = torch.tensor(top_20[n][0], device=labels.device)
                    correctly_retained = torch.logical_and((labels == 0), (prediction == 0))
                    correctly_pruned = torch.logical_and((labels == 1), (prediction == 1))
                    energy_correctly_retained = torch.sum(energies[correctly_retained]) / torch.sum(
                        energies[labels == 0])
                    self.log(f"energy_correctly_retained_{n}", energy_correctly_retained)
                    self.log(f"energy_correctly_retained_{n}_{len(labels)}", energy_correctly_retained)
                    if torch.sum(labels) > 0:
                        energy_correctly_pruned = torch.sum(energies[correctly_pruned]) / torch.sum(
                            energies[labels == 1])
                        self.log(f"energy_correctly_pruned_{n}", energy_correctly_pruned)
                        self.log(f"energy_correctly_pruned_{n}_{len(labels)}", energy_correctly_pruned)
                        self.log(f"sum_energy_pruned_retained_{n}", energy_correctly_pruned + energy_correctly_retained)
                        self.log(f"sum_energy_pruned_retained_{n}_{len(labels)}",
                                 energy_correctly_pruned + energy_correctly_retained)
                    if torch.sum(labels) > 0.1 * len(labels):
                        self.log(f"high_pruning_fr_energy_correctly_retained_{n}", energy_correctly_retained)
                        self.log(f"high_pruning_fr_energy_correctly_retained_{n}_{len(labels)}",
                                 energy_correctly_retained)
                        self.log(f"high_pruning_fr_energy_correctly_pruned_{n}", energy_correctly_pruned)
                        self.log(f"high_pruning_fr_energy_correctly_pruned_{n}_{len(labels)}", energy_correctly_pruned)
                        self.log(f"high_pruning_fr_sum_energy_pruned_retained_{n}",
                                 energy_correctly_pruned + energy_correctly_retained)
                        self.log(f"high_pruning_fr_sum_energy_pruned_retained_{n}_{len(labels)}",
                                 energy_correctly_pruned + energy_correctly_retained)
                except:
                    continue

    def generate_edge_factors(self, g):
        g.apply_edges(self.concat_message_function)
        g.edata['out'] = self.edge_generator(g.edata['cat_feat'])
        return g

    @staticmethod
    def partition_function(pgm):
        order = dgl.topological_nodes_generator(pgm)
        for nodes in order:
            nodes = nodes.to(pgm.device)
            pgm.pull(nodes, NeuralMarkovTree.send_message, fn.sum('factor', 'fct'))
            pgm.apply_nodes(lambda x: {'out': x.data['out'] + x.data['fct']}, nodes)
        roots = dgl.topological_nodes_generator(pgm, reverse=True)[0].to(pgm.device)
        return torch.sum(torch.logsumexp(pgm.ndata["out"][roots], dim=-1))

    @staticmethod
    def sum_likelihood(pgm, labels):
        num_classes = pgm.ndata["out"].shape[1]
        pgm.apply_nodes(lambda nodes: {"c": nodes.data["out"][F.one_hot(labels, num_classes=num_classes).bool()]})
        pgm.apply_edges(NeuralMarkovTree.edge_output)
        return torch.sum(pgm.ndata["c"]) + torch.sum(pgm.edata["c"])

    @staticmethod
    def sample(pgm, steps, initial_rejection, rejection_rate):
        o, t = pgm.edges()
        order = dgl.topological_nodes_generator(pgm)
        num_classes = int(math.sqrt(pgm.edata["out"].shape[1]))
        pgm = dgl.add_edges(pgm, t, o,
                            data={"out": torch.transpose(pgm.edata["out"].reshape((-1, num_classes, num_classes)), 1,
                                                         2).reshape((-1, num_classes ** 2))})
        pgm.pull(pgm.nodes(), NeuralMarkovTree.receive_sample_init_msg, fn.sum('factor', 'fct'))
        pgm.apply_nodes(lambda x: {'factor': x.data['out'] + x.data['fct']}, pgm.nodes())
        pgm.apply_nodes(lambda x: {"label": F.gumbel_softmax(x.data["factor"], dim=-1, hard=True)})
        samples = torch.zeros(
            ((steps // rejection_rate) - (initial_rejection // rejection_rate), pgm.number_of_nodes()),
            device=pgm.device, dtype=torch.uint8)
        for i in range(steps):
            for nodes in order:
                nodes = nodes.to(pgm.device)
                pgm.pull(nodes, NeuralMarkovTree.receive_sample_loopy_msg, fn.sum('factor', 'fct'))
                pgm.apply_nodes(lambda x: {'factor': x.data['out'] + x.data['fct']}, pgm.nodes())
                pgm.apply_nodes(lambda x: {"label": F.gumbel_softmax(x.data["factor"], dim=-1, hard=True)})
            if i >= initial_rejection and i % rejection_rate == 0:
                samples[(i // rejection_rate) - (initial_rejection // rejection_rate)] = torch.argmax(
                    pgm.ndata["label"], dim=-1)
        return samples

    @staticmethod
    def edge_output(edges):
        num_classes = int(math.sqrt(edges.data["out"].shape[1]))
        labels = F.one_hot(edges.src["y"] + num_classes * edges.dst["y"], num_classes=num_classes ** 2).bool()
        return {"c": edges.data["out"][labels]}

    @staticmethod
    def send_message(edges):
        num_classes = int(math.sqrt(edges.data["out"].shape[1]))
        messages = edges.data["out"].reshape((-1, num_classes, num_classes)) + edges.src["out"].unsqueeze(1)
        messages = torch.logsumexp(messages, dim=-1)
        return {"factor": messages}

    @staticmethod
    def send_max_product_message(edges):
        num_classes = int(math.sqrt(edges.data["out"].shape[1]))
        messages = edges.data["out"].reshape((-1, num_classes, num_classes)) + edges.src["out"].unsqueeze(1)
        classifications = torch.argmax(torch.logsumexp(messages, dim=-1), dim=-1)
        messages = messages[range(messages.shape[0]), classifications]
        return {"factor": messages}

    @staticmethod
    def reverse_max_product_message(edges):
        num_classes = int(math.sqrt(edges.data["out"].shape[1]))
        messages = edges.data["out"].reshape((-1, num_classes, num_classes))
        messages = torch.logsumexp(messages, dim=1)
        return {"factor": messages}

    @staticmethod
    def max_product_classify(pgm):
        order = dgl.topological_nodes_generator(pgm)
        for nodes in order:
            nodes = nodes.to(pgm.device)
            pgm.pull(nodes, NeuralMarkovTree.send_max_product_message, fn.sum('factor', 'fct'))
            pgm.apply_nodes(lambda x: {'out': x.data['out'] + x.data['fct']}, nodes)
        reverse_pgm = pgm.reverse(copy_edata=True)
        reverse_pgm.pull(range(reverse_pgm.number_of_nodes()), NeuralMarkovTree.reverse_max_product_message, fn.sum('factor', 'fct'))
        reverse_pgm.apply_nodes(lambda x: {'out': x.data['out'] + x.data['fct']})
        return torch.argmax(reverse_pgm.ndata["out"], dim=-1)

    @staticmethod
    def concat_message_function(edges):
        return {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}

    @staticmethod
    def receive_sample_init_msg(edges):
        num_classes = int(math.sqrt(edges.data["out"].shape[1]))
        messages = edges.data["out"].reshape((-1, num_classes, num_classes))
        messages = torch.logsumexp(messages, dim=-1)
        return {"factor": messages}

    @staticmethod
    def receive_sample_loopy_msg(edges):
        num_classes = int(math.sqrt(edges.data["out"].shape[1]))
        return {"factor": torch.transpose(edges.data["out"].reshape((-1, num_classes, num_classes)), 1, 2)[
            edges.src["label"].bool()]}


class MultiTST_NeuralMarkovTree(NeuralMarkovTree):
    def __init__(self, in_feats=9, num_h_feats=128, dropout=0.1, num_classes=2, lr=0.001, num_steps=2,
                 backbone='ggsnn', num_layers=3):
        super(NeuralMarkovTree, self).__init__()
        self.save_hyperparameters()
        if num_layers > 1:
            self.i = MultiLayeredGatedGraphConv(in_feats, num_h_feats, n_steps=num_steps, n_etypes=2,
                                                n_layers=num_layers, dropout=dropout)
        else:
            self.i = GatedGraphConv(in_feats, num_h_feats, n_steps=num_steps, n_etypes=2)
        if backbone != 'gat':
            self.dropout = nn.Dropout(dropout)
        self.edge_generator = nn.Linear(num_h_feats * 2, 2 * num_classes)
        self.node_generator = nn.Linear(num_h_feats, num_classes)
        self.num_classes = num_classes

    def forward(self, batch):
        g, prop_graph, etypes = batch
        feats = g.ndata["x"]
        h = self.i(prop_graph, feats, etypes)
        if self.hparams.backbone != 'gat':
            h = self.dropout(h)
        g.ndata["feat"] = h
        g = self.generate_edge_factors(g)
        g.ndata["out"] = self.node_generator(h)
        return g


class SST_NeuralMarkovTree(NeuralMarkovTree):
    def __init__(self, in_feats=300, num_h_feats=400, dropout=0.1, num_classes=2, lr=0.001, num_steps=2,
                 backbone='ggsnn', num_layers=3):
        super(NeuralMarkovTree, self).__init__()
        self.save_hyperparameters()
        if num_layers > 1:
            self.i = MultiLayeredGatedGraphConv(in_feats, num_h_feats, n_steps=num_steps, n_layers=num_layers,
                                                dropout=dropout)
        else:
            self.i = GatedGraphConv(in_feats, num_h_feats, n_steps=num_steps, n_etypes=1)
        if backbone != 'gat':
            self.dropout = nn.Dropout(dropout)
        self.edge_generator = nn.Linear(num_h_feats * 2, num_classes * num_classes)
        self.node_generator = nn.Linear(num_h_feats, num_classes)
        self.num_classes = num_classes

    def forward(self, batch):
        g = batch
        feats = g.ndata["x"]
        h = self.i(g, feats)
        if self.hparams.backbone != 'gat':
            h = self.dropout(h)
        g.ndata["feat"] = h
        g = self.generate_edge_factors(g)
        g.ndata["out"] = self.node_generator(h)
        return g

    def test_step(self, batch, batch_idx):
        labels = batch.ndata["y"]
        pgm = self(batch)
        likelihood = self.sum_likelihood(pgm, labels) - self.partition_function(pgm)
        likelihood = likelihood / len(labels)
        loss = -likelihood
        self.log("test_loss", loss, batch_size=len(dgl.unbatch(batch)))
        self.log("test_likelihood", likelihood, batch_size=len(dgl.unbatch(batch)))
        n_steps = 5000
        classifications = self.max_product_classify(pgm).to(labels.device)
        self.log("max-product accuracy", torch.mean(classifications == labels, dtype=torch.float64), batch_size=1)
        samples = self.sample(pgm, n_steps + 100, 100, 10)
        pgm.ndata["sample"] = samples.T
        for g, trackster in zip(dgl.unbatch(batch), dgl.unbatch(pgm)):
            labels = g.ndata["y"]
            samples = trackster.ndata["sample"].T
            samples_strings = [tuple(row) for row in samples.tolist()]
            counter = collections.Counter(samples_strings)
            top_100 = counter.most_common(100)
            self.log("accuracy",
                     torch.mean(torch.Tensor(top_100[0][0]).to(labels.device) == labels, dtype=torch.float64),
                     batch_size=1)
            counter_thousand_samples = collections.Counter(samples_strings[:n_steps // 100])
            for i, entry in enumerate(top_100):
                self.log(f"top-{i + 1}-mass", entry[1] / (n_steps // 10), batch_size=1)
                self.log(f"top-{i + 1}-mass-thousand", counter_thousand_samples[entry[0]] / (n_steps // 100),
                         batch_size=1)

    def step(self, batch, tag):
        batch_size = len(dgl.unbatch(batch))
        labels = batch.ndata["y"]
        pgm = self.forward(batch)
        likelihood = self.sum_likelihood(pgm, labels) - self.partition_function(pgm)
        loss = -likelihood
        likelihood = likelihood / len(labels)

        self.log(f"{tag}_log_likelihood", likelihood, sync_dist=True, batch_size=batch_size)
        self.log(f"{tag}_loss", -likelihood, sync_dist=True, batch_size=batch_size)
        if tag == "val":
            self.log("hp_metric", likelihood, sync_dist=True, batch_size=batch_size)
        return loss


class GMNN(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GMNN")
        parser.add_argument("--include_features", action='store_true')
        return parent_parser

    def __init__(self, num_h_feats, in_feats=3, num_classes=2, lr=0.001, num_steps=2,
                 include_features=True, epochs=100, n_etypes=2, batch_size=32):
        super(GMNN, self).__init__()
        self.save_hyperparameters()
        self.q = GatedGraphConv(in_feats, num_h_feats, n_steps=num_steps, n_etypes=n_etypes)
        self.p = GatedGraphConv(in_feats + num_classes, num_h_feats, n_steps=num_steps, n_etypes=n_etypes)
        self.qlin = nn.Linear(num_h_feats, num_classes)
        self.plin = nn.Linear(num_h_feats, num_classes)
        if include_features:
            self.aggregation = lambda x, y: torch.cat([x, y], dim=1)
        else:
            self.aggregation = lambda x, y: x
        self.automatic_optimization = False
        self.batch_size = batch_size

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_q = torch.optim.RMSprop(self.q.parameters(), lr=lr)
        opt_p = torch.optim.RMSprop(self.p.parameters(), lr=lr)
        return opt_q, opt_p

    def forward(self, g, feats):
        out_q = self.q(g, feats)
        out_q = self.qlin(out_q)
        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, feats)
        out_p = self.p(g, in_p)
        return out_p

    def q_step(self, batch, tag):
        _, g, etypes = batch
        opt, _ = self.optimizers()
        features = g.ndata["x"]
        labels = g.ndata["y"]
        out_q = self.q(g, features.float(), etypes)
        out_q = self.qlin(out_q)
        # weights = self._calculate_weights(labels)
        loss = F.cross_entropy(out_q, labels)
        if tag == "train":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        pred = torch.argmax(F.softmax(out_q, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(loss.detach()), pred, deepcopy(labels), f"q_{tag}")

    def p_step(self, batch, tag):
        _, g, etypes = batch
        _, opt = self.optimizers()
        features = g.ndata["x"].float()
        labels = g.ndata["y"]
        out_q = self.q(g, features, etypes)
        out_q = self.qlin(out_q)
        # weights = self._calculate_weights(labels)

        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p, etypes)
        out_p = self.plin(out_p)

        loss = F.cross_entropy(out_p, labels)
        if tag == "train":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        pred = torch.argmax(F.softmax(out_p, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(loss.detach()), pred, deepcopy(labels), f"p_{tag}")

    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.hparams.epochs // 2:
            self.q_step(batch, "train")
        else:
            self.p_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.hparams.epochs // 2:
            self.q_step(batch, "val")
        else:
            self.p_step(batch, "val")

    def test_step(self, batch, batch_idx):
        _, g, etypes = batch
        features = g.ndata["x"].float()
        out_q = self.q(g, features, etypes)
        out_q = self.qlin(out_q)
        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p, etypes)
        out_p = self.plin(out_p)
        q_loss = F.cross_entropy(out_q, g.ndata["y"])
        p_loss = F.cross_entropy(out_p, g.ndata["y"])
        self.log("test_q_likelihood", -q_loss, batch_size=self.batch_size)
        self.log("test_p_likelihood", -p_loss, batch_size=self.batch_size)
        pred_q = F.gumbel_softmax(out_q, hard=True, dim=-1)
        pred_p = F.gumbel_softmax(out_p, hard=True, dim=-1)
        self.log("q_accuracy", torch.mean(torch.argmax(pred_q, dim=-1) == g.ndata["y"], dtype=torch.float64),
                 batch_size=self.batch_size)
        self.log("p_accuracy", torch.mean(torch.argmax(pred_p, dim=-1) == g.ndata["y"], dtype=torch.float64),
                 batch_size=self.batch_size)
        samples = torch.zeros((1000, g.number_of_nodes()), dtype=torch.uint8, device=g.device)
        for i in range(1000):
            samples[i] = torch.argmax(F.gumbel_softmax(out_p, hard=True), dim=-1)
        batch = batch[0]
        batch.ndata["sample"] = samples.T
        for graph in dgl.unbatch(batch):
            labels = graph.ndata["y"]
            samples = graph.ndata["sample"].T
            samples_strings = [tuple(row) for row in samples.tolist()]
            counter = collections.Counter(samples_strings)
            top_20 = counter.most_common(20)
            for i, entry in enumerate(top_20):
                self.log(f"top-{i + 1}-mass", entry[1] / 1000)
            label_str = ''.join(str(item) for item in labels.tolist())
            occurrence = 21
            for n in range(min(20, len(top_20))):
                sample_str = ''.join(str(item) for item in top_20[n][0])
                if sample_str == label_str:
                    occurrence = n
                    break
            for n in range(20):
                self.log(f"top-{n + 1}-accuracy", int(occurrence <= n))
                self.log(f"top-{n + 1}-{len(labels)}-accuracy", int(occurrence <= n))
                if torch.sum(labels) > 0:
                    self.log(f"prunable-top-{n + 1}-accuracy", int(occurrence <= n))
                    self.log(f"prunable-top-{n + 1}-{len(labels)}-accuracy", int(occurrence <= n))
            if torch.sum(labels) > 0.05 * len(labels):
                subgraphs = graph.subgraph(labels.bool())
                if len(dgl.topological_nodes_generator(subgraphs)) > 2:
                    for n in range(20):
                        self.log(f"clustered-top-{n + 1}-accuracy", int(occurrence <= n))

    def log_results(self, loss, pred, labels, tag):
        self.log(f"{tag}_accuracy", (labels == pred).float().mean(), prog_bar=False, on_epoch=True, on_step=False,
                 sync_dist=True, batch_size=1)
        self.log("hp_metric", -loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f"{tag}_log_likelihood", -loss, sync_dist=True, batch_size=self.batch_size)
        self.log(f"{tag}_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)


class SST_GMNN(GMNN):

    def q_step(self, batch, tag):
        g = batch
        opt, _ = self.optimizers()
        features = g.ndata["x"]
        labels = g.ndata["y"]
        out_q = self.q(g, features)
        out_q = self.qlin(out_q)
        loss = F.cross_entropy(out_q, labels)
        if tag == "train":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        pred = torch.argmax(F.softmax(out_q, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(loss.detach()), pred, deepcopy(labels), f"q_{tag}")

    def p_step(self, batch, tag):
        g = batch
        _, opt = self.optimizers()
        features = g.ndata["x"]
        labels = g.ndata["y"]
        out_q = self.q(g, features)
        out_q = self.qlin(out_q)

        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p)
        out_p = self.plin(out_p)

        loss = F.cross_entropy(out_p, labels)
        if tag == "train":
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        pred = torch.argmax(F.softmax(out_p, dim=-1), dim=-1).detach()
        self.log_results(deepcopy(loss.detach()), pred, deepcopy(labels), f"p_{tag}")

    def test_step(self, batch, batch_idx):
        g = batch
        features = g.ndata["x"]
        out_q = self.q(g, features)
        out_q = self.qlin(out_q)
        in_p = F.gumbel_softmax(out_q, hard=True, dim=-1)
        in_p = self.aggregation(in_p, features).detach()
        out_p = self.p(g, in_p)
        out_p = self.plin(out_p)
        q_loss = F.cross_entropy(out_q, batch.ndata["y"])
        p_loss = F.cross_entropy(out_p, batch.ndata["y"])
        self.log("test_q_likelihood", -q_loss, batch_size=self.batch_size)
        self.log("test_p_likelihood", -p_loss, batch_size=self.batch_size)
        pred_q = F.gumbel_softmax(out_q, hard=True, dim=-1)
        pred_p = F.gumbel_softmax(out_p, hard=True, dim=-1)
        self.log("q_accuracy", torch.mean(torch.argmax(pred_q, dim=-1) == batch.ndata["y"], dtype=torch.float64),
                 batch_size=self.batch_size)
        self.log("p_accuracy", torch.mean(torch.argmax(pred_p, dim=-1) == batch.ndata["y"], dtype=torch.float64),
                 batch_size=self.batch_size)
