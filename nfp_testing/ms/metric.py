"""Monosemanticity Metric - weighted pairwise cosine similarity over neuron activations.

Measures how monosemantic each neuron (SAE feature) is by computing, for every pair
of clips, the cosine similarity of their embeddings weighted by the product of the two
clips' activations for that neuron. A neuron that fires only on embedding-similar clips
scores high; one that fires indiscriminately scores low. Embeddings and activations must
describe the SAME clips in the SAME row order, which is asserted before any computation.

Usage:
    python -m nfp_testing.ms.metric --embeddings_path EMB.pth \
        --activations_dir ACT_DIR --output_subdir SUBDIR --device cpu
"""

import torch
import os.path
import argparse
from utils.datasets.activations import ActivationsDataset
import os

from torch.utils.data import DataLoader, Subset
import tqdm
import torch.nn.functional as F


def get_args_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the monosemanticity metric.

    Returns:
        The configured argument parser (with ``add_help=False``).
    """
    parser = argparse.ArgumentParser(
        "Measure monosemanticity via weighted pairwise cosine similarity", add_help=False
    )
    parser.add_argument("--embeddings_path")
    parser.add_argument("--activations_dir")
    parser.add_argument("--output_subdir")
    parser.add_argument("--device", default="cpu")
    return parser


def main(args: argparse.Namespace) -> None:
    """Compute the per-neuron monosemanticity score and write scores plus a summary.

    Loads clip embeddings and neuron activations (asserting a matching row order),
    min-max scales activations per neuron to [0, 1], accumulates the activation-weighted
    pairwise cosine similarity over all clip pairs, and reports the mean/std score, the
    dead-neuron count, and the top and bottom 10 neurons. Results are saved under
    ``activations_dir/output_subdir``.

    Args:
        args: Parsed command-line arguments with ``embeddings_path``,
            ``activations_dir``, ``output_subdir``, and ``device``.

    Returns:
        None. Writes ``all_neurons_scores.pth`` and ``metric_stats_new.txt`` to disk.
    """
    compute_device = torch.device(args.device)

    # Load embeddings
    embeddings = torch.load(args.embeddings_path, map_location=compute_device)
    print(f"Loaded embeddings found at {args.embeddings_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Load activations
    activations_dataset = ActivationsDataset(
        args.activations_dir, device=compute_device, take_every=1
    )
    activations_dataloader = DataLoader(
        activations_dataset, batch_size=len(activations_dataset), shuffle=False
    )
    activations = next(iter(activations_dataloader))
    print(f"Loaded activations found at {args.activations_dir}")
    print(f"Activations shape: {activations.shape}")

    # Embeddings and activations must be the SAME clips in the SAME order; a row
    # mismatch would silently pair unrelated clips (review item 7).
    assert embeddings.shape[0] == activations.shape[0], (
        f"row mismatch: {embeddings.shape[0]} embeddings vs "
        f"{activations.shape[0]} activation rows (must be the same clips, in order)"
    )

    # Scale to 0-1 per neuron
    min_values = activations.min(dim=0, keepdim=True)[0]
    max_values = activations.max(dim=0, keepdim=True)[0]
    activations = (activations - min_values) / (max_values - min_values)

    # embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]

    # Initialize accumulators (kept on the single declared compute device)
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=compute_device)
    weight_sum = torch.zeros(num_neurons, device=compute_device)
    batch_size = 100  # Set batch size

    for i in tqdm.tqdm(range(num_images), desc="Processing image pairs"):
        for j_start in range(i + 1, num_images, batch_size):  # Process in batches
            j_end = min(j_start + batch_size, num_images)

            embeddings_i = embeddings[i].to(compute_device)  # (embedding_dim)
            embeddings_j = embeddings[j_start:j_end].to(
                compute_device
            )  # (batch_size, embedding_dim)
            activations_i = activations[i].to(compute_device)  # (num_neurons)
            activations_j = activations[j_start:j_end].to(
                compute_device
            )  # (batch_size, num_neurons)

            # Compute cosine similarity
            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(
                    j_end - j_start, -1
                ),  # Expanding to (batch_size, embedding_dim)
                embeddings_j,
                dim=1,
            )

            # Compute weights and weighted similarities
            # Expanding activations_i to (1, num_neurons)
            weights = activations_i.unsqueeze(0) * activations_j  # (batch_size, num_neurons)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(
                1
            )  # (batch_size, num_neurons)

            weighted_cosine_similarities = torch.sum(
                weighted_cosine_similarities, dim=0
            )  # (num_neurons)
            weighted_cosine_similarity_sum += weighted_cosine_similarities

            weights = torch.sum(weights, dim=0)  # (num_neurons)
            weight_sum += weights

    monosemanticity = torch.where(
        weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan
    )

    os.makedirs(os.path.join(args.activations_dir, args.output_subdir), exist_ok=True)
    torch.save(
        monosemanticity,
        os.path.join(args.activations_dir, args.output_subdir, "all_neurons_scores.pth"),
    )

    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])

    print(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}")
    print(f"Dead neurons:", nan_count.item())
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()

    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, 10)
    bottom_10_values, bottom_10_indices = torch.topk(valid_monosemanticity, 10, largest=False)

    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]

    # Print results
    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    # Save to file
    output_path = os.path.join(args.activations_dir, args.output_subdir, "metric_stats_new.txt")
    with open(output_path, "w") as file:
        file.write(
            f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}\n"
        )
        file.write(f"Dead neurons: {nan_count.item()}\n")
        file.write(f"Total neurons: {num_neurons}\n\n")

        file.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_10_indices, top_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")

        file.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bottom_10_indices, bottom_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
