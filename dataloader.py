import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import dgl
import uproot
import numpy as np
from synthetic_graph_generation import *
from torch.utils.data.dataset import T_co


class JSONTracksterDataset(Dataset):

    def __init__(self, filename):
        self.data = []
        with open(filename, 'r') as f:
            data = json.load(f)
        for point in data:
            if sum(point["label"]) > 0:
                self.data.append(point)
        self.num = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        num = self.num
        if num == len(self):
            raise StopIteration
        self.num += 1
        return self[num]

    def __getitem__(self, item):
        if isinstance(self.data[item], dict):
            self.data[item] = self._generate_graph(self.data[item])
        return self.data[item]

    @staticmethod
    def _generate_graph(trackster_dict):
        num_nodes = len(trackster_dict["id"])
        node_features = torch.zeros((num_nodes, 9))
        for i, key in enumerate(
                ["pos_x", "pos_y", "pos_z", "energy", "time", "eta", "phi", "isSeedCLUE3DHigh", "isSeedCLUE3DLow"]):
            node_features[:, i] = torch.Tensor(trackster_dict[key])
        o, t = [], []
        for i, parents in enumerate(trackster_dict["parents"]):
            for parent in parents:
                o.append(trackster_dict["id"].index(parent))
                t.append(i)
        g = dgl.graph((o, t), num_nodes=num_nodes)
        g.ndata["features"] = node_features
        g = dgl.add_reverse_edges(g)
        g = dgl.add_self_loop(g)
        labels = torch.Tensor(trackster_dict["label"])
        return g, labels



class TracksterDataset(Dataset):

    def __init__(self, filename, trackster_root_name, edge_root_name, step_size = 100):
        read_file = uproot.open(filename)
        self.trackster_root = read_file[trackster_root_name]
        self.edge_root = read_file[edge_root_name]
        self.step_size = step_size

    def __len__(self):
        return len(self.trackster_root["id"].array())

    def __getitem__(self, index):
        trackster_data = self.trackster_root.arrays(entry_start=index, entry_stop=index+1)[0]
        edge_root = self.edge_root.arrays(entry_start=index, entry_stop=index+1)[0]
        return self._generate_graph(trackster_data, edge_root)

    @staticmethod
    def _generate_graph(trackster, edges):
        num_nodes = len(trackster.id)
        ids = list(trackster.id)
        node_features = torch.zeros((num_nodes, 9))
        for i, key in enumerate(
                ["pos_x", "pos_y", "pos_z", "energy", "time", "eta", "phi", "isSeedCLUE3DHigh", "isSeedCLUE3DLow"]):
            node_features[:, i] = torch.Tensor(trackster[key])
        if sum(trackster["isSeedCLUE3DLow"]) > 0:
            seed = np.argsort(trackster["isSeedCLUE3DLow"])[-1]
        elif sum(trackster["isSeedCLUE3DHigh"]) > 0:
            seed = np.argsort(trackster["isSeedCLUE3DHigh"])[-1]
        else:
            seed = 0
        node_features[:, :3] = node_features[:, :3] - node_features[seed, :3]
        node_features[:, 4] = F.relu(node_features[:,4])
        o, t = [ids.index(i) for i in edges.origin], [ids.index(i) for i in edges.target]
        g = dgl.graph((o, t), num_nodes=num_nodes)
        g.ndata["features"] = node_features
        g = dgl.add_reverse_edges(g)
        g = dgl.add_self_loop(g)
        labels = torch.Tensor(trackster["label"])
        return g, labels

    @staticmethod
    def collate_fn(data):
        return data

class SyntheticData(Dataset):
    def __init__(self, num_graphs_per):
        self.graphs = [generate_example_graph() for i in range(num_graphs_per)]

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    @staticmethod
    def collate_fn(data):
        return data