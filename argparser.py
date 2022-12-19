import argparse
import torch


def parse_trainer(parser):
    parser.add_argument("--training_fraction", type=restricted_float, default=0.75, help="Train/val split fraction")
    parser.add_argument("--sampling_fraction", type=restricted_float, default=1, help="Subset of data to use")
    parser.add_argument("--prefetch_factor", type=int, default=3, help="Prefetch factor for data fetching threads")
    parser.add_argument("--batch_size", type=int, help="Batch size to use during training", default=1)
    parser.add_argument("--workers", type=int, help="Number of worker threads to use for data fetching", default=8)
    parser.add_argument("--seed", type=int, help="Seed to use for random sampling", default=42)
    parser.add_argument("--output_file", type=str, help="File to write output model to", default="tigmn.pt")
    parser.add_argument("--gpus", type=int, help="Number of GPU's to train on", default=torch.cuda.device_count())
    parser.add_argument("--gmnn", action='store_true')
    parser.add_argument("--cern", action='store_true')
    return parser

def parse_continue(parser):
    parser.add_argument("--training_fraction", type=restricted_float, default=0.75, help="Train/val split fraction")
    parser.add_argument("--sampling_fraction", type=restricted_float, default=1, help="Subset of data to use")
    parser.add_argument("--prefetch_factor", type=int, default=3, help="Prefetch factor for data fetching threads")
    parser.add_argument("--batch_size", type=int, help="Batch size to use during training", default=1)
    parser.add_argument("--workers", type=int, help="Number of worker threads to use for data fetching", default=8)
    parser.add_argument("--seed", type=int, help="Seed to use for random sampling", default=42)
    parser.add_argument("--gpus", type=int, help="Number of GPU's to train on", default=torch.cuda.device_count())
    parser.add_argument("--ckpt", type=str, help="Checkpoint to continue training from", default="")
    return parser


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x
