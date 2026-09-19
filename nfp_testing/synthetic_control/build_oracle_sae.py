"""
Build an oracle SAE - a positive control for the NFP test.

Injects linear probe directions for each of the 5 tau variables
(speed, vel_x, vel_y, accel_mag, direction) as the first 5 encoder
rows of the existing VideoMAE SAE.  These oracle features are
*guaranteed* to encode temporal motion concepts by construction.

Running nfp_test.py on this oracle SAE should flag features 0-4 as
statistically significant, validating that the NFP test can detect
genuine temporal encoders (positive control / theory check).

Construction:
  1. Run VideoMAE on the NFP dataset, extract ball-tracking reps [N,8,768]
  2. Fit OLS probes:  tau_k ≈ (x - sae.bias) @ w_k + b_k
  3. Replace sae.encoder.weight[k] = w_k / ||w_k||   (unit direction)
     Replace sae.encoder.bias[k]   = -mean_proj_k    (centers activation)
     Replace sae.decoder.weight[:,k] = w_k / ||w_k|| (keeps decoder normalized)
  4. Save state_dict → oracle_ae.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from utils.models.videomae import VideoMAE

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_TEMPORAL = 8
N_SPATIAL = 196
N_FRAMES = 16


# ---------------------------------------------------------------------------
# Dataset (identical structure to nfp_test.py)
# ---------------------------------------------------------------------------


class NFPDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.video_dirs = sorted(root_dir.glob("v*"))
        if not self.video_dirs:
            raise FileNotFoundError(f"No video dirs found in {root_dir}")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        frames = [Image.open(vdir / f"rgba_{i:05d}.png").convert("RGB") for i in range(N_FRAMES)]
        with open(vdir / "metadata.json") as f:
            meta = json.load(f)
        traj = meta["trajectory"]
        tau_steps, ball_token_steps = [], []
        for step in range(N_TEMPORAL):
            frame_rec = traj[step * 2]
            tau_steps.append([frame_rec["tau"][k] for k in TAU_KEYS])
            ball_token_steps.append(frame_rec["spatial_token"])
        return (
            frames,
            torch.tensor(tau_steps, dtype=torch.float32),  # [8, 5]
            torch.tensor(ball_token_steps, dtype=torch.long),  # [8]
        )


def make_collate(processor):
    def collate(batch):
        inputs = processor(images=[b[0] for b in batch], return_tensors="pt")
        tau = torch.stack([b[1] for b in batch])  # [B, 8, 5]
        ball_tokens = torch.stack([b[2] for b in batch])  # [B, 8]
        return inputs, tau, ball_tokens

    return collate


def extract_ball_tracking(acts_spatial, ball_tokens):
    B, T, S, D = acts_spatial.shape
    mask = ball_tokens >= 0
    safe_tokens = ball_tokens.clamp(min=0)
    idx = safe_tokens.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, D)
    ball_acts = acts_spatial.gather(2, idx).squeeze(2)
    ball_acts = ball_acts * mask.unsqueeze(-1).float()
    return ball_acts, mask


# ---------------------------------------------------------------------------
# Collect all ball-tracking representations
# ---------------------------------------------------------------------------


def collect_activations(model, ds, args, device):
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(model.processor),
    )
    hook_key = f"{args.attachment_point}_{args.layer}"
    model.attach(args.attachment_point, args.layer, sae=None)

    all_X, all_tau, all_mask = [], [], []
    for inputs, tau, ball_tokens in tqdm(dl, desc="Extracting VideoMAE reps"):
        model.encode(inputs)
        acts = model.register[hook_key][0]  # [B, 1568, 768]
        B = acts.shape[0]
        acts_spatial = acts.view(B, N_TEMPORAL, N_SPATIAL, -1)

        ball_acts, mask = extract_ball_tracking(acts_spatial.to(device), ball_tokens.to(device))
        all_X.append(ball_acts.cpu())
        all_tau.append(tau)
        all_mask.append(mask.cpu())

    X = torch.cat(all_X, dim=0)  # [N, 8, 768]
    tau = torch.cat(all_tau, dim=0)  # [N, 8, 5]
    mask = torch.cat(all_mask, dim=0)  # [N, 8]
    return X, tau, mask


# ---------------------------------------------------------------------------
# Fit OLS linear probes
# ---------------------------------------------------------------------------


def fit_probes(X_c, tau_flat, mask_flat):
    """
    X_c      : [M, 768]  ball-tracking reps centered by sae.bias (float32 numpy)
    tau_flat : [M, 5]
    mask_flat: [M] bool  - use only on-screen frames for fitting

    Returns unit-norm directions [5, 768].
    """
    on = mask_flat
    X_on = X_c[on]  # [M_on, 768]
    tau_on = tau_flat[on]  # [M_on, 5]

    # Augment with intercept column
    A = np.hstack([X_on, np.ones((X_on.shape[0], 1), dtype=np.float32)])

    directions = []
    print(f"\n{'Probe':>12}  {'||w||':>8}  {'R²':>8}  {'mean_proj':>10}")
    print("-" * 46)
    for k, name in enumerate(TAU_KEYS):
        y = tau_on[:, k]
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        w = coef[:768].astype(np.float32)
        norm = np.linalg.norm(w)
        if norm < 1e-8:
            print(f"  WARNING: probe for {name} near-zero, using random dir")
            w = np.random.randn(768).astype(np.float32)
            norm = np.linalg.norm(w)
        w_hat = w / norm
        proj = X_on @ w_hat
        mean_proj = proj.mean()
        y_hat = A @ coef
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"{name:>12}  {norm:>8.4f}  {r2:>8.4f}  {mean_proj:>10.4f}")
        directions.append(w_hat)

    return np.stack(directions, axis=0)  # [5, 768]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--sae_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    p.add_argument("--layer", default=11, type=int)
    p.add_argument("--attachment_point", default="post_mlp_residual")
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)

    ds = NFPDataset(Path(args.dataset_dir))
    print(f"Dataset: {len(ds)} videos")

    X, tau, mask = collect_activations(model, ds, args, device)
    # X: [N, 8, 768], tau: [N, 8, 5], mask: [N, 8]

    print(f"\nLoading SAE: {args.sae_path}")
    sae = AutoEncoder.from_pretrained(args.sae_path, device=torch.device("cpu"))
    sae.eval()

    N, T, D = X.shape
    X_flat = X.reshape(N * T, D).numpy()  # [N*8, 768]
    tau_flat = tau.reshape(N * T, 5).numpy()  # [N*8, 5]
    mask_flat = mask.reshape(N * T).numpy().astype(bool)

    # Center X by SAE's global bias - this is the input to encoder.linear
    bias_np = sae.bias.data.cpu().numpy()  # [768]
    X_c = X_flat - bias_np[None, :]  # [N*8, 768]

    print("\nFitting linear probes on centered activations...")
    directions = fit_probes(X_c, tau_flat, mask_flat)  # [5, 768] unit vectors

    print("\nInjecting oracle features into SAE (features 0–4)...")
    with torch.no_grad():
        for k, name in enumerate(TAU_KEYS):
            d_k = torch.tensor(directions[k], dtype=torch.float32)  # [768] unit vec

            # Encoder direction
            sae.encoder.weight.data[k] = d_k

            # Encoder bias: center activation around mean projection
            on = mask_flat
            mu_k = float((X_c[on] @ directions[k]).mean())
            sae.encoder.bias.data[k] = -mu_k
            print(f"  feature {k:2d} ({name:<12}): b_enc = {-mu_k:+.4f}")

            # Decoder: set to same unit direction so decoder norm = 1
            # normalize_decoder won't scale encoder when decoder col has norm 1
            sae.decoder.weight.data[:, k] = d_k

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), out_path)
    print(f"\nOracle SAE saved -> {out_path}")
    print(f"  Features 0-4 : linear probes for {TAU_KEYS}")
    print(f"  Features 5-{sae.dict_size - 1}: unchanged from original SAE")
    print(f"\nNext step: run nfp_test.py with --sae_path {out_path}")


if __name__ == "__main__":
    main()
