import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import dgl
import uproot
import numpy as np
from tqdm import tqdm

from synthetic_graph_generation import *


class TracksterDataset(Dataset):

    def __init__(self, filename, trackster_root_name, edge_root_name, step_size = 100):
        read_file = uproot.open(filename)
        self.trackster_root = read_file[trackster_root_name]
        self.edge_root = read_file[edge_root_name]
        self.step_size = step_size
        self.trackster_arrays = self.trackster_root.arrays()
        self.edge_arrays = self.edge_root.arrays()

    def __len__(self):
        return len(self.trackster_root["id"].array())

    def __getitem__(self, index):
        trackster_data = self.trackster_arrays[index]
        edge_root = self.edge_arrays[index]
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
        g = dgl.graph((t, o), num_nodes=num_nodes)
        g.ndata["x"] = node_features
        g.ndata["y"] = torch.Tensor(trackster["label"]).long()
        return g

    @staticmethod
    def collate_fn(data):
        return dgl.batch(data)

class InMemoryDataset(Dataset):
    def __init__(self, filename, trackster_root_name, edge_root_name, step_size=100):
        set = TracksterDataset(filename, trackster_root_name, edge_root_name, step_size)
        self.data = []
        loader = DataLoader(set, batch_size=16, num_workers=3, prefetch_factor=3, persistent_workers=False,collate_fn=SyntheticData.collate_fn, shuffle=False)
        for batch in tqdm(loader):
            self.data.extend(batch)
            if len(self.data) > 500:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class SyntheticData(Dataset):
    def __init__(self, num_graphs_per):
        self.graphs = [generate_example_graph() for i in range(num_graphs_per)]
        #self.graphs.extend([generate_example_graph() for i in range(num_graphs_per)])

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    @staticmethod
    def collate_fn(data):
        return data