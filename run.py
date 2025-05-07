# run.py - For generating dataset batches based on VAE paper configs
import argparse
import numpy as np
import os
from datasets import get_dataset  # make sure your datasets.py includes get_dataset
import jax

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['linear_gaussian', 'sigmoid_gaussian', 'sphere'], required=True)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--padding_dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--dd', type=int, default=3)  # underlying data dimension
    parser.add_argument('--num_batches', type=int, default=10000)  # how many samples to generate
    args = parser.parse_args()
    return args

def main(args):
    dataset = get_dataset(args.dataset, args.dataset_seed, args.padding_dim, args.batch_size, args)
    data = dataset.get_batch(args.num_batches)

    os.makedirs("generated_data", exist_ok=True)
    save_path = os.path.join("generated_data", f"{args.name}.npy")
    np.save(save_path, np.array(data))
    print(f"[✓] Saved data to {save_path}")

if __name__ == '__main__':
    with jax.disable_jit():  # ensure deterministic behavior
        args = parse_arguments()
        main(args)
