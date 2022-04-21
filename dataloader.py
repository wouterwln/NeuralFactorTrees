import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import uproot
import numpy as np
import awkward as ak
import collections
from tqdm import tqdm
import os
import shutil

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

class preprocessingEvent(Dataset):
    def __init__(self, filename, event_root):
        file = uproot.open(filename)
        self.events_root = file[event_root]

    def __getitem__(self, item):
        arr = self.events_root.arrays(entry_start=item+ 20582, entry_stop=item+20583)[0]
        return EventDataset.preprocess_event(arr)

    def __len__(self):
        return len(self.events_root["event"].array()) - 20582

class EventDataset(Dataset):
    def __init__(self, filename, event_root, num_samples=0):
        file = uproot.open(filename)
        self.events_root = file[event_root]
        #if num_samples > 0:
        #    self.events = self.events_root.arrays(entry_stop = num_samples)
        self.arrays = []
        self.length = num_samples
        to_select = ['id', 'pos_x', 'pos_y', 'pos_z', 'energy', 'time', 'eta', 'phi', 'layer', 'isSeedCLUE3DHigh', 'isSeedCLUE3DLow', 'simTst_idx', 'recoTst_idx', 'ts_edgestart', 'ts_edgeend']
        for arrays in tqdm(self.events_root.iterate(to_select, step_size=10, entry_stop=num_samples), total=(num_samples // 10)):
            self.arrays.append(arrays)

    def __getitem__(self, item):
        return self.preprocess_event(self.arrays[item // 100][item % 100])

    def __len__(self):
        return self.length

    @staticmethod
    def preprocess_event(event):
        """
        Takes an event and outputs a graph wil labels and node features

        :param event:
        :return:
        """
        reco_tsts = set(event.recoTst_idx[:, 0])
        reco_tsts.remove(-1)
        if len(reco_tsts) == 0:
            return []
        graphs = []
        intratrackster_graph = dgl.DGLGraph()

        for trackster in reco_tsts:
            layer_clusters = event.id[event.recoTst_idx[:, 0] == trackster]
            ids = layer_clusters.tolist()
            counter = collections.Counter(ak.flatten(event.simTst_idx[layer_clusters]))
            del counter[-1]
            most_common = counter.most_common(1)[0][0]
            labels = torch.tensor([most_common not in lc for lc in event.simTst_idx[layer_clusters]], dtype=torch.int64)
            o, t = [ids.index(i) for i in event.ts_edgestart[trackster]], [ids.index(i) for i in event.ts_edgeend[trackster]]
            g = dgl.graph((t, o))
            node_features = torch.zeros((len(layer_clusters), 9))
            intratrackster_graph.add_nodes(len(layer_clusters))
            for i, key in enumerate(
                    ["pos_x", "pos_y", "pos_z", "energy", "time", "eta", "phi", "isSeedCLUE3DHigh", "isSeedCLUE3DLow"]):
                node_features[:, i] = torch.Tensor(event[key][layer_clusters])
            g.ndata["id"] = torch.tensor(layer_clusters)
            g.ndata["y"] = torch.tensor(labels)
            g.ndata["x"] = node_features
            graphs.append(g)
        global_ids = dgl.batch(graphs).ndata["id"].tolist()
        o = []
        t = []
        for layer in range(1, 52):
            mask = ~(~(event.layer == layer) + ~(event.recoTst_idx[:, 0] != -1))
            relevant_layer_clusters = event.id[mask]
            if len(relevant_layer_clusters) > 10:
                coordinates = torch.tensor([event.pos_x[relevant_layer_clusters], event.pos_y[relevant_layer_clusters]]).T
                neighbors = torch.cdist(coordinates, coordinates).topk(4, largest=False).indices[:, 1:]
                for origin, a in enumerate(neighbors):
                    o.extend(global_ids.index(relevant_layer_clusters[origin]) for _ in a)
                    t.extend(global_ids.index(relevant_layer_clusters[target]) for target in a)
        edges = np.array([o,t])
        return graphs, edges

    @staticmethod
    def find(tensor, values):
        return torch.nonzero(tensor[..., None] == values)

    @staticmethod
    def collate_fn(data):
        return data

class PreProcessedEventDataset(Dataset):
    def __init__(self, include_intratrackster_edges=True, use_presaved_edges=True, k=1):
        self.root_dir = "event_graph_dataset"
        self.datapoints = os.listdir(self.root_dir)
        self.length = len(self.datapoints)
        self.include_intratrackster_edges = include_intratrackster_edges
        self.use_presaved_edges = use_presaved_edges
        self.k = k

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        i = self.datapoints[item]
        edges = torch.tensor([[], []], dtype=torch.int64)
        try:
            g = dgl.load_graphs(f"event_graph_dataset/{i}/{i}.dgl")[0]
            if self.include_intratrackster_edges:
                if self.k == 10 and os.path.exists(f"event_graph_dataset/{i}/{i}.npy"):
                    edges = np.load(f"event_graph_dataset/{i}/{i}.npy", allow_pickle=True)
                else:
                    edges = self.prepare_edges(g)
                    edges = edges.numpy().astype(np.int16)
        except IndexError:
            return self[item - 1]

        return g, edges

    def prepare_edges(self, g):
        for i, graph in enumerate(g):
            graph.ndata["i"] = torch.ones(graph.number_of_nodes()) * i
        g = dgl.batch(g)
        g.ndata["y"] = g.ndata["y"].type(torch.int64)
        g.ndata["x"] = g.ndata["x"].type(torch.float32)
        e = []
        layers = torch.round(g.ndata["x"][:, 2])
        ulayers = torch.unique(layers)
        for layer in ulayers:
            mask = torch.abs(layers - layer) <= 5
            relevant_layer_clusters = mask.nonzero().flatten()
            if len(relevant_layer_clusters) > 10:
                coordinates = torch.stack(
                    [g.ndata["x"][:, 0][mask], g.ndata["x"][:, 1][mask]]).T
                neighbors = relevant_layer_clusters[(torch.cdist(coordinates, coordinates) < self.k).nonzero()]
                tsts = g.ndata["i"][neighbors]
                neighbors = neighbors[tsts[:,0] != tsts[:, 1]]
                e.append(neighbors)
        try:
            edges = torch.cat(e).T
        except:
            return torch.zeros(2, 1)
        edges = torch.sort(edges, dim=0).values
        return torch.unique(edges, dim=-1)

    @staticmethod
    def collate_fn(data):
        graphs = []
        prop_graphs = []
        for batch in data:
            batched_graphs, edges = batch
            g = dgl.batch(batched_graphs)
            prop_graph = dgl.add_reverse_edges(g)
            prop_graph.edata["t"] = torch.zeros(prop_graph.number_of_edges())
            try:
                edges = edges.astype(np.int64)
            except:
                edges = edges.type(torch.int64)
            if len(edges) > 1:
                prop_graph.add_edges(edges[0], edges[1], data={"t": torch.ones(len(edges[0]))})
                prop_graph.add_edges(edges[1], edges[0], data={"t": torch.ones(len(edges[0]))})
            else:
                prop_graph.add_edges([0], [0], data={"t": torch.ones(1)})
            graphs.append(g)
            prop_graphs.append(prop_graph)
        graphs = dgl.batch(graphs)
        prop_graphs = dgl.batch(prop_graphs)
        etypes = prop_graphs.edata["t"]
        graphs.ndata["y"] = graphs.ndata["y"].type(torch.int64)
        graphs.ndata["x"] = graphs.ndata["x"].type(torch.float32)
        return graphs, prop_graphs, etypes

    @staticmethod
    def filtered_collate_fn(data):
        graphs = []
        prop_graphs = []
        for batch in data:
            batched_graphs, edges = batch
            new_graphs = []
            for individual_graph in batched_graphs:
                individual_graph.apply_edges(lambda edges: {"lab": edges.dst["y"] > edges.src["y"]})
                if not torch.any(individual_graph.edata["lab"]):
                    new_graphs.append(individual_graph)
            if len(new_graphs) == 0:
                continue
            g = dgl.batch(new_graphs)
            prop_graph = dgl.add_reverse_edges(g)
            graphs.append(g)
            prop_graphs.append(prop_graph)
        graphs = dgl.batch(graphs)
        prop_graphs = dgl.batch(prop_graphs)
        etypes = torch.zeros(graphs.number_of_nodes())
        return graphs, prop_graphs, etypes