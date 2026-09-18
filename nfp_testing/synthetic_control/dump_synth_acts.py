"""
Dump the synthetic representations h(V,t) from all_videos.pt as a single flat
[N*T, d] activation chunk that analysis/fit_pca_ica.py can consume.

This lets PCA/ICA be fit on EXACTLY the representations the synthetic SAE was
trained on, so the PCA/ICA synthetic baselines are a like-for-like comparison
(same input distribution, same Dictionary interface).
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_videos_path", required=True,
                    help="all_videos.pt saved by gen_synthetic_activations.py")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write synth_acts_part0.pt into")
    args = ap.parse_args()

    data = torch.load(args.all_videos_path, map_location="cpu")
    h = data["h"]                                    # [N, T, d]
    flat = h.reshape(-1, h.shape[-1]).float().contiguous()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "synth_acts_part0.pt"
    torch.save(flat, dst)
    print(f"Saved {tuple(flat.shape)} synthetic reps -> {dst}")


if __name__ == "__main__":
    main()
