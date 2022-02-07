import dgl
import torch
import random


def generate_clique(num_nodes, seem_pos, pos):
    o = [([i] * (num_nodes)) for i in range(num_nodes)]
    o = [item for sublist in o for item in sublist]
    t = [([i for i in range(num_nodes)])] * num_nodes
    t = [item for sublist in t for item in sublist]
    g = dgl.graph((o, t))
    features = torch.randn((g.number_of_nodes(), 9))
    indices = torch.randint(g.number_of_nodes(), (seem_pos,))
    features[indices] = features[indices] * 2
    labels = torch.zeros(g.number_of_nodes())
    labels[indices[:pos + 1]] = 1
    g.ndata["features"] = features
    return g, labels

def generate_split():
    o = [0, 1, 2, 3, 4, 5, 6]
    t = [1, 2, 3, 4, 5, 6, 0]
    g = dgl.graph((o, t))
    g.ndata["x"] = torch.randn((g.number_of_nodes(),9))
    if random.random() > 0.5:
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 1])
    else:
        labels = torch.tensor([1, 0, 1, 0, 1, 0, 1])
    g.ndata["y"] = labels
    return dgl.add_self_loop(dgl.add_reverse_edges(g))


def generate_example_graph():
    o = [0, 1, 1, 2, 3]
    t = [1, 2, 3, 3, 4]
    g = dgl.graph((o, t))
    g.ndata["x"] = torch.ones((g.number_of_nodes(),3))
    g.ndata["x"][[0, 3]] = g.ndata["x"][[0, 3]] * 2

    labels = torch.zeros(g.number_of_nodes())
    r = random.random()
    if r < 0.05:
        pass
    elif r < 0.45:
        labels[0] = 1
    elif r < 0.85:
        labels[3] = 1
    else:
        labels[0] = 1
        labels[3] = 1
    labels[4] = 1
    g.ndata["y"]  = labels.long()

    return dgl.add_self_loop(dgl.add_reverse_edges(g))