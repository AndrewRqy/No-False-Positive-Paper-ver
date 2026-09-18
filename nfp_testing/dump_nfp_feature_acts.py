"""
Dump per-video SAE feature activations on the 3000 NFP ball videos.

The NFP test (analysis/nfp_test.py) computes the ball-tracking SAE activation
psi_i(V,t) [B,8,6144] for every video but only keeps the within-video covariance C.
To eyeball *which videos drive each significant feature* we need the activations
themselves. This re-runs the identical VideoMAE->ball-token->SAE.encode pipeline and
saves a per-video aggregate of each feature's activation, so videos can be ranked
per feature.

For each video V and feature i we store:
  feat_mean[V,i] = mean over ON-SCREEN steps of psi_i(V,t)   (0 if never on-screen)
  feat_max [V,i] = max  over the 8 steps of psi_i(V,t)
Ranking videos for a feature uses feat_mean (stable) by default.

Output: --output_path .pt with {feat_mean[N,6144], feat_max[N,6144],
onscreen[N], video_ids}. Mirrors nfp_test.py exactly (same model/layer/attachment,
same ball-tracking extraction and off-screen zeroing).

Usage (from sae-for-vlm/):
  python analysis/dump_nfp_feature_acts.py --dataset_dir data/output/nfp \
      --sae_path local_runs/sae/ae.pt --output_path local_runs/nfp_results/sae_feat_acts.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from utils.models.videomae import VideoMAE
from nfp_testing.nfp_test import (
    NFPDataset, make_collate, extract_ball_tracking,
    N_TEMPORAL, N_SPATIAL,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="data/output/nfp")
    p.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    p.add_argument("--output_path", default="local_runs/nfp_results/sae_feat_acts.pt")
    p.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    p.add_argument("--layer", default=11, type=int)
    p.add_argument("--attachment_point", default="post_mlp_residual")
    p.add_argument("--batch_size", default=4, type=int)
    # Windows: nfp_test.make_collate returns a local closure -> not picklable; keep 0.
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    hook_key = f"{args.attachment_point}_{args.layer}"
    model.attach(args.attachment_point, args.layer, sae=None)

    print(f"Loading SAE: {args.sae_path}")
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    sae.eval()

    ds = NFPDataset(Path(args.dataset_dir), tau_mode="first_frame")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(model.processor))
    print(f"Dataset: {len(ds)} videos")

    feat_mean_chunks, feat_max_chunks, onscreen_chunks, ids = [], [], [], []
    for inputs, tau, ball_tokens, video_ids in tqdm(dl, desc="Activations"):
        model.encode(inputs)
        acts = model.register[hook_key][0]                      # [B, 1568, 768]
        B = acts.shape[0]
        acts_spatial = acts.view(B, N_TEMPORAL, N_SPATIAL, -1)

        ball_acts, mask = extract_ball_tracking(
            acts_spatial.to(device), ball_tokens.to(device))    # [B,8,768], [B,8]

        with torch.no_grad():
            B2, T, Dh = ball_acts.shape
            feats = sae.encode(ball_acts.reshape(B2 * T, Dh)).reshape(B2, T, -1)
        m = mask.to(feats.device).unsqueeze(-1).float()         # [B,8,1]
        feats = feats * m                                       # zero off-screen

        cnt = mask.sum(dim=1).clamp(min=1).to(feats.device).float().unsqueeze(-1)  # [B,1]
        feat_mean = (feats.sum(dim=1) / cnt).cpu()              # [B,6144] masked mean
        feat_max = feats.max(dim=1).values.cpu()               # [B,6144]

        feat_mean_chunks.append(feat_mean)
        feat_max_chunks.append(feat_max)
        onscreen_chunks.append(mask.sum(dim=1).cpu())
        ids.extend(video_ids)

    out = {
        "feat_mean": torch.cat(feat_mean_chunks, 0),   # [N, 6144]
        "feat_max":  torch.cat(feat_max_chunks, 0),    # [N, 6144]
        "onscreen":  torch.cat(onscreen_chunks, 0),    # [N]
        "video_ids": ids,
    }
    outp = Path(args.output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, outp)
    print(f"\nSaved -> {outp}")
    print(f"  feat_mean {tuple(out['feat_mean'].shape)}  feat_max {tuple(out['feat_max'].shape)}")
    print(f"  videos: {len(ids)}  (e.g. {ids[:3]})")


if __name__ == "__main__":
    main()
