import argparse
import time
import logging
import jax
import jax.numpy as jnp

from datasets import SphereDataset, LinearGaussianDataset, SigmoidDataset
from diffusion_model import UNet, GaussianDiffusion
from utils import make_output_dir, manifold_error, save_samples


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="Experiment name and output directory.")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for training.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument('--timesteps', type=int, default=1000,
                        help="Number of diffusion timesteps.")
    parser.add_argument('--beta_start', type=float, default=1e-4,
                        help="Start beta for noise schedule.")
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help="End beta for noise schedule.")
    parser.add_argument('--dataset', choices=['sphere', 'linear_gaussian', 'sigmoid'],
                        default='linear_gaussian', help="Dataset type.")
    parser.add_argument('-ds', '--dataset_seed', type=int, default=0,
                        help="Random seed for dataset.")
    parser.add_argument('-dd', '--dataset_dimension', type=int, default=3,
                        help="Ambient data dimension d.")
    parser.add_argument('-did', '--dataset_intrinsic_dimension', type=int, default=3,
                        help="Intrinsic manifold dimension r*.")
    parser.add_argument('-dn', '--dataset_noise', type=float, default=0.,
                        help="Added noise variance for linear dataset.")
    parser.add_argument('--padding_dim', type=int, default=0,
                        help="Padding dimension for dataset.")
    return parser.parse_args()


def get_dataset(args):
    if args.dataset == 'sphere':
        ds = SphereDataset(
            seed=args.dataset_seed,
            dimension=args.dataset_dimension,
            padding_dimension=args.padding_dim
        )
    elif args.dataset == 'linear_gaussian':
        ds = LinearGaussianDataset(
            seed=args.dataset_seed,
            dimension=args.dataset_dimension,
            intrinsic_dimension=args.dataset_intrinsic_dimension,
            padding_dimension=args.padding_dim,
            var_added=args.dataset_noise
        )
    elif args.dataset == 'sigmoid':
        ds = SigmoidDataset(
            seed=args.dataset_seed,
            dimension=args.dataset_dimension,
            padding_dimension=args.padding_dim
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_ds, test_ds, data_shape = ds.get_splits(batch_size=args.batch_size)
    return train_ds, test_ds, data_shape


def train_diffusion(train_ds, test_ds, data_shape, args):
    # Initialize logging
    logging.info(f"Starting diffusion training for {args.num_epochs} epochs...")

    # Build model and diffusion
    channels = [64, 128, 256]
    unet = UNet(channels=channels)
    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        lr=args.learning_rate
    )

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        for batch in train_ds:
            key = jax.random.PRNGKey(args.dataset_seed + epoch)
            diffusion.state = diffusion.train_step(diffusion.state, batch, key)

        # Evaluation
        key = jax.random.PRNGKey(args.dataset_seed + epoch)
        samples = diffusion.sample(key, (args.batch_size, *data_shape))
        err = manifold_error(samples, test_ds)
        logging.info(f"Epoch {epoch}/{args.num_epochs} - manifold_error: {err:.4f} - time: {time.time() - start_time:.1f}s")

    # Final sampling and save
    key = jax.random.PRNGKey(args.dataset_seed + args.num_epochs + 1)
    final_samples = diffusion.sample(key, (args.batch_size, *data_shape))
    save_samples(final_samples, args.name)
    logging.info("Training complete and samples saved.")


def main():
    args = parse_arguments()
    output_dir = make_output_dir(args.name, overwrite=False)

    logging.basicConfig(level=logging.INFO)
    train_ds, test_ds, data_shape = get_dataset(args)
    train_diffusion(train_ds, test_ds, data_shape, args)


if __name__ == '__main__':
    main()
