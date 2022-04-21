from dataloader import TracksterDataset, PreProcessedEventDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    batch_size = 1
    dataset = PreProcessedEventDataset(k=10)
    data = DataLoader(dataset, num_workers=3, prefetch_factor=3, batch_size=batch_size,
                      collate_fn=PreProcessedEventDataset.collate_fn, persistent_workers=True)
    #snsdata = []
    #with tqdm(data) as pbar:
    #    for batch in pbar:
    #        for graph in dgl.unbatch(batch[0]):
    #            labels = graph.ndata["y"]
    #            frac = torch.sum(labels) / len(labels)
    #            snsdata.append(frac.item())
    #            if len(snsdata) > 10000:
    #                break
    #        if len(snsdata) > 10000:
    #            break
    #sns.displot(snsdata, kind='hist')
    #plt.show()
    #
    with tqdm(data) as pbar:
        for batch in pbar:
            for graph in dgl.unbatch(batch[0]):
                g = dgl.reverse(graph)
                g_nx = dgl.to_networkx(g)
                pos = graphviz_layout(g_nx, prog="dot")
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(1,1,1)
                ax.set_facecolor((238./255., 232./255., 232./255.))
                nx.draw_networkx(g_nx, pos, node_color=g.ndata["y"], with_labels=False, ax=ax)
                plt.show()
