"""Dump Raw Ball-Token Activations - NFP cache builder.

Dumps RAW ball-token layer-11 activations for the 3000 NFP ball videos.

One GPU pass; afterwards the NFP test for ANY dictionary (second SAE, PCA, etc.) is a
cheap CPU computation (encode + within-video covariance + t-test) via nfp_on_dict.py.
Identical pipeline to nfp_testing.nfp_test: VideoMAE -> layer-11 post-MLP residual ->
ball-token extraction with off-screen zeroing.

Output: {ball [N,8,768], tau [N,8,5], mask [N,8], video_ids}.

Usage:
  python -m nfp_testing.dump_ball_raw_acts --output_path local_runs/nfp_results/ball_raw_acts.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models.videomae import VideoMAE
from nfp_testing.nfp_test import (
    NFPDataset,
    make_collate,
    extract_ball_tracking,
    N_TEMPORAL,
    N_SPATIAL,
)


def main() -> None:
    """Extract and save raw ball-token activations for the NFP dataset.

    Runs a single VideoMAE forward pass over every video, gathers the layer-11
    post-MLP residual activation at the ball-containing spatial token (with
    off-screen zeroing), and saves the stacked ball activations, tau sequences,
    on-screen masks, and video ids to the output path.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/output/nfp")
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--attachment_point", default="post_mlp_residual")
    ap.add_argument("--batch_size", default=4, type=int)
    ap.add_argument("--num_workers", default=0, type=int)  # Windows: closure collate
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_path", default="local_runs/nfp_results/ball_raw_acts.pt")
    args = ap.parse_args()
    device = torch.device(args.device)

    model = VideoMAE(args.model_name, device)
    hook_key = f"{args.attachment_point}_{args.layer}"
    model.attach(args.attachment_point, args.layer, sae=None)
    ds = NFPDataset(Path(args.dataset_dir), tau_mode="first_frame")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(model.processor),
    )

    B_all, T_all, M_all, ids = [], [], [], []
    for inputs, tau, ball_tokens, video_ids in tqdm(dl, desc="Extract"):
        model.encode(inputs)
        acts = model.register[hook_key][0]
        B = acts.shape[0]
        acts_spatial = acts.view(B, N_TEMPORAL, N_SPATIAL, -1)
        ball_acts, mask = extract_ball_tracking(acts_spatial.to(device), ball_tokens.to(device))
        B_all.append(ball_acts.cpu())
        T_all.append(tau)
        M_all.append(mask.cpu())
        ids += list(video_ids)

    out = {
        "ball": torch.cat(B_all, 0),
        "tau": torch.cat(T_all, 0),
        "mask": torch.cat(M_all, 0),
        "video_ids": ids,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_path)
    print(f"saved {out['ball'].shape} -> {args.output_path}")


if __name__ == "__main__":
    main()
