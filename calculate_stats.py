from dataloader import TracksterDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

if __name__ == "__main__":
    batch_size = 5
    dataset = TracksterDataset("tracksters_preprocessed.root", "Tracksters;1", "Edges;1")
    data = DataLoader(dataset, num_workers=3, prefetch_factor=3, batch_size=batch_size,
                      collate_fn=TracksterDataset.collate_fn, persistent_workers=True)
    num_zeros = 0
    num_nodes = 0
    num_pruned = 0
    with tqdm(data) as pbar:
        for batch in pbar:
            for trackster, labels in batch:
                if torch.sum(labels) == 0:
                    num_zeros += 1
                num_nodes += len(labels)
                num_pruned += torch.sum(labels).item()
            pbar.set_postfix(tst_require_pruning=f"{(100 * num_zeros) / ((pbar.n * batch_size) + 1.)}%", num_nodes_pruned=f"{(100 * num_pruned) / num_nodes}")

