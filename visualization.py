import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import uproot
from dataloader import TracksterDataset
from torch.utils.data import DataLoader
import dgl
import torch

def generate_edges(trackster):
    o, t = trackster.edges()
    o = trackster.ndata["features"][o, :3].unsqueeze(-1)
    t = trackster.ndata["features"][t, :3].unsqueeze(-1)
    edges = torch.cat([o, t], dim=-1)
    edges = edges.tolist()
    return edges


def plot_trackster(trackster, labels, cc_seed = True):
    '''
        Helper function to plot the points.
    '''
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(projection='3d')
    xs = np.array(trackster.ndata["features"][:,0])
    ys = np.array(trackster.ndata["features"][:,1])
    zs = np.array(trackster.ndata["features"][:,2])
    sizes = [20 + (e * 10) for e in
             trackster.ndata["features"][:,3]]  # marker size proportional to point energy

    ax.scatter(xs, zs, ys, s=sizes, marker='o', c=labels, alpha=.5)
    edges = generate_edges(trackster)
    for edge in edges:
        ax.plot(edge[0], edge[2], edge[1], c='blue')
    ax.set_xlabel('$X$', rotation=150)
    ax.set_ylabel('$Z$')
    ax.set_zlabel('$Y$', rotation=60)
    ax.set_title("XYZ")
    fig.suptitle(f"Event ", fontsize=16)
    plt.show(block=True)

if __name__ =="__main__":
    file = uproot.open("tracksters_preprocessed.root")
    dataset = TracksterDataset("tracksters_preprocessed.root", "Tracksters;1", "Edges;1")
    data = DataLoader(dataset,num_workers=1, batch_size=1, prefetch_factor=2,
                      collate_fn=TracksterDataset.collate_fn)

    for batch in data:
        trackster, labels = batch[0]
        plot_trackster(trackster, labels)