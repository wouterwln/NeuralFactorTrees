import argparser
from train import train

if __name__ == "__main__":
    args = argparser.parse_voxelnet()
    model = train(args.filename, args.trackster_root_name, args.edge_root_name, args.batch_size, args.training_fraction, args.epochs,
              args.workers, args.prefetch_factor, args.sampling_fraction, args.seed)
