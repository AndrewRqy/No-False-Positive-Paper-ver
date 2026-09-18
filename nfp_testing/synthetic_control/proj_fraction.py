"""
Projection fraction metric for synthetic SAE encoder directions.

For each SAE encoder direction w_i in R^768, compute:
    proj_frac(w_i) = ||P_{W_tau} w_i||^2 / ||w_i||^2

where P_{W_tau} = W_tau^T W_tau is the orthogonal projector onto the 5-dimensional
temporal subspace (valid because W_tau has orthonormal rows, so W_tau @ W_tau.T = I_5).

This is strictly more informative than max cosine similarity:
- Max cosine measures alignment with the single closest temporal direction.
- Proj fraction measures the total squared length in the entire temporal subspace,
  capturing features that mix multiple temporal directions (e.g., speed+direction).

Compare significant vs non-significant features from the NFP test output.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_path",     required=True)
    p.add_argument("--matrices_path", required=True,
                   help="matrices.pt from gen_synthetic_activations.py")
    p.add_argument("--nfp_results",  required=True,
                   help="synthetic_nfp.pt from nfp_test_synthetic.py")
    p.add_argument("--alpha",        default=0.05, type=float)
    p.add_argument("--device",       default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading SAE: {args.sae_path}")
    sae = AutoEncoder.from_pretrained(args.sae_path,
                                      device=torch.device(args.device))
    enc_W = sae.encoder.weight.data.cpu().float()  # [F, 768]
    F, D = enc_W.shape

    print(f"Loading matrices: {args.matrices_path}")
    mat = torch.load(args.matrices_path, map_location="cpu")
    W_tau    = mat["W_tau"].float()     # [5, 768] orthonormal rows
    W_static = mat["W_static"].float()  # [N_STATIC, 768] orthonormal rows

    print(f"Loading NFP results: {args.nfp_results}")
    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_val = nfp["p_val"].numpy()  # [F, 5]
    bonf  = args.alpha / F

    sig_any = (p_val < bonf).any(axis=1)  # [F]
    sig_idx    = np.where(sig_any)[0]
    nonsig_idx = np.where(~sig_any)[0]

    # Projection fraction: ||W_tau w_i||^2 / ||w_i||^2
    # W_tau has orthonormal rows, so P_{W_tau} = W_tau^T W_tau
    # ||P w_i||^2 = ||W_tau w_i||^2  (since P is symmetric idempotent)
    proj_tau    = (enc_W @ W_tau.T)       # [F, 5]
    proj_static = (enc_W @ W_static.T)    # [F, N_STATIC]
    norm_sq     = (enc_W ** 2).sum(dim=1) + 1e-12  # [F]

    frac_tau    = (proj_tau    ** 2).sum(dim=1) / norm_sq  # [F]
    frac_static = (proj_static ** 2).sum(dim=1) / norm_sq  # [F]

    print(f"\n{'='*60}")
    print(f"Projection Fraction Metric  (Bonferroni p < {bonf:.2e})")
    print(f"  Temporal subspace dim  : {W_tau.shape[0]}")
    print(f"  Static subspace dim    : {W_static.shape[0]}")
    print(f"  SAE features total     : {F}")
    print(f"{'='*60}")

    for label, idx in [("Significant", sig_idx), ("Non-significant", nonsig_idx)]:
        n = len(idx)
        if n == 0:
            print(f"\n{label} ({n}): (none)")
            continue
        ft = frac_tau[idx].numpy()
        fs = frac_static[idx].numpy()
        print(f"\n{label} ({n} features):")
        print(f"  Proj frac in W_tau    : mean={ft.mean():.4f}  "
              f"median={np.median(ft):.4f}  max={ft.max():.4f}")
        print(f"  Proj frac in W_static : mean={fs.mean():.4f}  "
              f"median={np.median(fs):.4f}  max={fs.max():.4f}")
        print(f"  Residual (noise+other): mean={(1-ft-fs).mean():.4f}")

    # Per-tau breakdown: proj fraction of features sig for that tau only
    print(f"\n--- Per-tau projection fraction (features sig for that tau) ---")
    print(f"{'Tau':<12} {'N_sig':>6} {'Mean frac_tau':>14} {'Mean frac_static':>17}")
    for k, name in enumerate(TAU_KEYS):
        sig_k = (p_val[:, k] < bonf)
        n_k   = sig_k.sum()
        if n_k == 0:
            print(f"{name:<12} {0:>6}  (no significant features)")
            continue
        ft_k = frac_tau[sig_k].numpy()
        fs_k = frac_static[sig_k].numpy()
        print(f"{name:<12} {n_k:>6} {ft_k.mean():>14.4f} {fs_k.mean():>17.4f}")

    # Ratio: how much more temporal alignment do sig features have?
    if len(sig_idx) > 0 and len(nonsig_idx) > 0:
        ratio = frac_tau[sig_idx].mean().item() / (frac_tau[nonsig_idx].mean().item() + 1e-8)
        print(f"\nSig / non-sig proj-frac ratio (W_tau): {ratio:.1f}x")


if __name__ == "__main__":
    main()
