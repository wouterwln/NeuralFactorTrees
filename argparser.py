import argparse

def parse_trainer(parser):

    parser.add_argument("filename", help="The filename of the root file containing events")
    parser.add_argument("trackster_root_name", help="The root name of the tracksters in the root file")
    parser.add_argument("edge_root_name", help="The root name of the edges in the root file")
    parser.add_argument("--training_fraction", type=restricted_float, default=0.75, help="Train/val split fraction")
    parser.add_argument("--sampling_fraction", type=restricted_float, default=1, help="Subset of data to use")
    parser.add_argument("--prefetch_factor", type=int, default=3, help="Prefetch factor for data fetching threads")
    parser.add_argument("--batch_size", type=int, help="Batch size to use during training", default=3)
    parser.add_argument("--epochs", type=int, help="Specify the amount of epochs", default=100)
    parser.add_argument("--workers", type=int, help="Number of worker threads to use for data fetching", default=8)
    parser.add_argument("--seed", type=int, help="Seed to use for random sampling", default=42)
    parser.add_argument("--output_file", type=str, help="File to write output model to", default="argnn.pt")
    parser.add_argument("--gpus", type=int, help="Number of GPU's to train on (default 1)", default=1)
    return parser

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x