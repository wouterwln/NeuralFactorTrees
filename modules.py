import torch

from torch.nn import functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch.utils.data import DataLoader, Subset
from dgl.nn import SAGEConv, GatedGraphConv
from dgl.nn.pytorch.utils import Sequential
from dataloader import TracksterDataset

import pytorch_lightning as pl


class Baseline(pl.LightningModule):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Baseline, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'lstm', activation=Sequential(Dropout(0.2), ReLU(), BatchNorm1d(h_feats)))
        self.conv2 = SAGEConv(h_feats, 2 * h_feats, 'lstm')
        self.conv5 = SAGEConv(2 * h_feats, num_classes, 'lstm')
        self.learning_rate = 0.001

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        return h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        batch_loss = 0
        for g, labels in train_batch:
            logits = self(g, g.ndata["features"])
            pred = logits.argmax(1)
            loss = F.cross_entropy(logits, labels.long())
            if isinstance(batch_loss, int):
                batch_loss = loss
            else:
                batch_loss += loss
            self.log('train_loss', loss)
            self.log('acc', (pred == labels).float().mean())
        return batch_loss


class ARGraphPruningModule(pl.LightningModule):
    def __init__(self, in_feats, h_feats, num_classes):
        super(ARGraphPruningModule, self).__init__()
        self.conv1 = SAGEConv(in_feats + 2, h_feats, 'lstm', activation=ReLU())
        self.convblock = Sequential(*[SAGEConv(h_feats, h_feats, 'lstm', activation=ReLU()) for _ in range(10)])
        self.conv2 = SAGEConv(h_feats, num_classes, 'lstm')

        self.lin1 = Linear(h_feats, 4 * h_feats)
        self.lin2 = Linear(4 * h_feats, 4 * h_feats)
        self.lin3 = Linear(4 * h_feats, num_classes)

        self.single_step_training = True

        self.learning_rate = 0.001

    def toggle_single_step(self):
        self.single_step_training = not self.single_step_training

    def forward(self, g, in_feat, state):
        h = torch.cat([in_feat, state], dim=1)
        h = self.conv1(g, h)
        h = self.convblock(g, h)
        o = self.conv2(g, h)
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)
        h = self.lin3(h)
        return o, h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        return self.single_step_pass(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        self.single_step_pass(val_batch, "val")

    def single_step_pass(self, batch, loss_tag):
        batch_loss = 0
        for g, labels in batch:
            state = torch.zeros((g.number_of_nodes(), 2)).to(g.device)
            state[:, 1] = torch.randint(2, (g.number_of_nodes(),))
            state[-1, 1] = 0
            state[:-1, 0] = F.relu((state[:-1, 1] + labels) - 1)

            o, h = self(g, g.ndata["features"], state)
            logits = F.softmax(o, dim=1)
            h = F.softmax(h, dim=1)
            if sum(F.relu(labels - state[:-1, 1])) == 0:
                labels = torch.cat([labels, torch.tensor([1.]).to(labels.device)])
            else:
                labels = torch.cat([labels, torch.tensor([0.]).to(labels.device)])
            # The targts this step is the labels minus the nodes we have visited
            step_targets = F.relu(labels - state[:, 1])
            #loss = F.cross_entropy(logits, step_targets.long())

            # The prediction should be correct
            chosen_node = torch.argmax(logits[:, 1])
            loss = -torch.log(h[chosen_node, step_targets[chosen_node].long()] + 1e-16)
            if step_targets[chosen_node] == 1:
                loss = 10 * -torch.log(h[chosen_node, step_targets[chosen_node].long()] + 1e-16)
            # Nodes that have their step target 1 should be chosen at some point (This exists as we always have to make a choice)
            loss += -torch.log(logits[step_targets == 1, 1] + 1e-16).mean()
            if torch.sum(state[:, 1]) > 0:
                # If we have visited nodes, we shouldn't visit them again
                loss += -torch.log(logits[state[:, 1] == 1, 0] + 1e-16).mean()
            if chosen_node == g.number_of_nodes() - 1:
                if sum(step_targets[:-1]) != 0:
                    loss += -torch.log(logits[step_targets == 1, 1] + 1e-16).mean()
                    loss += -torch.log(logits[chosen_node, 0] + 1e-16).mean()
                clf = 1
            else:
                clf = torch.argmax(h[chosen_node])
            self.log(f'{loss_tag}_acc', (clf == labels[chosen_node]).float(), prog_bar=True, on_step=True, on_epoch=True,
                     logger=True)
            if isinstance(batch_loss, int):
                batch_loss = loss
            else:
                batch_loss += loss
            self.log(f'{loss_tag}_loss', loss)
        return batch_loss

    def autoregressive_pass(self, batch, loss_tag):
        batch_loss = 0
        for g, labels in batch:
            state = torch.zeros((g.number_of_nodes(), 2)).to(g.device)
            chosen_node = 0
            while chosen_node != g.number_of_nodes() - 1:
                o, h = self(g, g.ndata["features"], state)
                logits = F.softmax(o, dim=1)
                h = F.softmax(h, dim=1)
                chosen_node = torch.argmax(logits[:, 1])
                if chosen_node == g.number_of_nodes() - 1:
                    break
                clf = torch.argmax(h[chosen_node])
                state[chosen_node, 0] = clf
                state[chosen_node, 1] = 1.
            labels = torch.cat([labels, torch.tensor([0.]).to(labels.device)])
            self.log(f'{loss_tag}_acc', (labels == state[:, 0]).float().mean())

class ARGGSNN(pl.LightningModule):

    def __init__(self, in_feats, h_feats):
        super(ARGGSNN, self).__init__()
        self.conv1 = GatedGraphConv(in_feats + 2, h_feats, 15, 1)
        self.node_selector = Linear(h_feats, 1)
        self.classifier = [Sequential(*[Linear(h_feats, h_feats) for _ in range(3)])]
        self.classifier += [Linear(h_feats, 2)]
        self.classifier = Sequential(*self.classifier)

    def forward(self, g, in_feat, state):
        h = torch.cat([in_feat, state], dim=1)
        h = self.conv1(g, h)
        o = self.node_selector(h)
        h = self.classifier(h)
        return o.flatten(), h

    def training_step(self, train_batch, batch_idx):
        return self.single_step_pass(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        self.single_step_pass(val_batch, "val")

    def single_step_pass(self, batch, loss_tag):
        batch_loss = 0
        for g, labels in batch:
            state = torch.zeros((g.number_of_nodes(), 2)).to(g.device)
            state[:, 1] = torch.randint(2, (g.number_of_nodes(),))
            state[-1, 1] = 0
            state[:-1, 0] = F.relu((state[:-1, 1] + labels) - 1)

            o, h = self(g, g.ndata["features"], state)
            logits = F.softmax(o)
            h = F.softmax(h, dim=1)
            if sum(F.relu(labels - state[:-1, 1])) == 0:
                labels = torch.cat([labels, torch.tensor([1.]).to(labels.device)])
            else:
                labels = torch.cat([labels, torch.tensor([0.]).to(labels.device)])
            # The targts this step is the labels minus the nodes we have visited
            step_targets = F.relu(labels - state[:, 1])
            loss = -torch.log(1 - torch.abs(logits - step_targets) + 1e-16).mean()

            # The prediction should be correct
            chosen_node = torch.argmax(logits)
            loss += -torch.log(h[chosen_node, step_targets[chosen_node].long()] + 1e-16)

            # Nodes that have their step target 1 should be chosen at some point (This exists as we always have to make a choice)
            loss += -torch.log(logits[step_targets == 1] + 1e-16).mean()
            if torch.sum(state[:, 1]) > 0:
                # If we have visited nodes, we shouldn't visit them again
                loss += -torch.log(1 - (logits[state[:, 1] == 1]) + 1e-16).mean()
            if chosen_node == g.number_of_nodes() - 1:
                clf = 1
            else:
                clf = torch.argmax(h[chosen_node])
            self.log(f'{loss_tag}_acc', (clf == labels[chosen_node]).float(), prog_bar=True, on_step=True, on_epoch=True,
                     logger=True)
            if isinstance(batch_loss, int):
                batch_loss = loss
            else:
                batch_loss += loss
            self.log(f'{loss_tag}_loss', loss)
        return batch_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer