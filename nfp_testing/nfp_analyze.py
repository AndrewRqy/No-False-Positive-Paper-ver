"""
Analyse saved NFP test results.

Loads the .pt file from nfp_test.py / nfp_test_dino.py and computes
feature-tau correlations: corr(feature_j(t), tau_k(t)) across videos.

Outputs:
  feat_tau_corr  [8, dict_size, 5]  Pearson r between each feature and each
                                     tau variable at each time step
  Summary stats printed to stdout.

Usage:
  python analysis/nfp_analyze.py \
      --results_path /net/scratch2/renqy/nfp_results/videomae_deadpen_0p03.pt \
      --output_path  /net/scratch2/renqy/nfp_results/videomae_deadpen_0p03_analyzed.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def feature_tau_corr(features: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    features: [N, T, D]
    tau:      [N, T, 5]
    Returns:  [T, D, 5]  Pearson r between feature d and tau variable k at time t.
    """
    N, T, D = features.shape
    _, _, K = tau.shape

    f_mu  = features.mean(axis=0, keepdims=True)   # [1, T, D]
    f_std = features.std(axis=0, keepdims=True) + 1e-8
    f_z   = (features - f_mu) / f_std              # [N, T, D]

    t_mu  = tau.mean(axis=0, keepdims=True)        # [1, T, K]
    t_std = tau.std(axis=0, keepdims=True) + 1e-8
    t_z   = (tau - t_mu) / t_std                   # [N, T, K]

    # corr[t, d, k] = mean_n( f_z[n,t,d] * t_z[n,t,k] )
    # Compute via einsum: [N,T,D] x [N,T,K] -> [T,D,K]
    corr = np.einsum('ntd,ntk->tdk', f_z, t_z) / N  # [T, D, K]
    return corr.astype(np.float32)


def print_report(feat_tau_corr: np.ndarray, label: str = ""):
    T, D, K = feat_tau_corr.shape
    print(f"\n=== Feature-Tau Correlation Report {label}===")

    for k, name in enumerate(TAU_KEYS):
        corr_k = feat_tau_corr[:, :, k]   # [T, D]
        abs_k  = np.abs(corr_k)

        # Best feature per time step
        best_feat = abs_k.max(axis=1)      # [T]
        # Mean over time steps and features
        mean_abs = abs_k.mean()
        # Fraction of features with |r| > 0.1 at any time step
        frac_high = (abs_k.max(axis=0) > 0.1).mean()

        print(f"\n  {name}:")
        print(f"    Mean |r| across all features & time steps : {mean_abs:.4f}")
        print(f"    Frac features with max-over-time |r| > 0.1: {frac_high:.4f}")
        print(f"    Best |r| per time step: "
              + " ".join(f"t{t}={best_feat[t]:.3f}" for t in range(T)))

        # Top 5 features for this tau variable (max |r| over time)
        top5_idx = np.abs(corr_k).max(axis=0).argsort()[::-1][:5]
        print(f"    Top 5 features (max |r| over time):")
        for i, d in enumerate(top5_idx):
            best_t = np.abs(corr_k[:, d]).argmax()
            print(f"      feature {d:5d}  max |r|={abs_k[:, d].max():.4f}  "
                  f"at t={best_t}  r={corr_k[best_t, d]:+.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True)
    p.add_argument("--output_path",  required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.results_path}")
    data = torch.load(args.results_path, map_location="cpu")

    features = data["features"].numpy()   # [N, 8, D]
    tau      = data["tau"].numpy()        # [N, 8, 5]
    label    = data.get("model", "")

    print(f"Features: {features.shape}  Tau: {tau.shape}")
    print("Computing feature-tau correlations...")

    corr = feature_tau_corr(features, tau)   # [T, D, 5]
    print_report(corr, label=f"({label}) " if label else "")

    out = {**data, "feat_tau_corr": torch.from_numpy(corr)}
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
