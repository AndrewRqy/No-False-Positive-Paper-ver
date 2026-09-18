"""
Plot the NFP test-statistic distribution relative to the Bonferroni cutoff, per basis.

For each feature we take its strongest temporal signal = max over the 5 tau variables of the
|t-statistic| from the NFP within-video-covariance test. A feature is "significant" iff this
exceeds the Bonferroni cutoff (equivalently min p-value < alpha/D, since all features share the
same df, so max|t| is monotonic in the p-value). Histogramming max|t| per basis with the cutoff
line drawn on it shows directly where each basis's mass sits: the SAE piles up BELOW the cutoff
(sparse) while raw/PCA/ICA pile ABOVE it (dense).

Output: results/pca_ica_baselines/figures/cutoff_distribution.png  (+ a printed summary).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (label, tensor path, feature count D) — all from the same local pipeline
SOURCES = [
    ("SAE (6144)",            "local_runs/nfp_results/sae_nfp.pt"),
    ("Raw layer (768)",       "local_runs/nfp_results/raw_nfp.pt"),
    ("PCA sign_split (512)",  "local_runs/nfp_results/pca_sign_split_nfp.pt"),
    ("ICA sign_split (512)",  "local_runs/nfp_results/ica_sign_split_nfp.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--xmax", type=float, default=20.0, help="x-axis clip for display")
    ap.add_argument("--out", default="results/pca_ica_baselines/figures/cutoff_distribution.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(SOURCES), 1, figsize=(7.5, 9.0), sharex=True)
    print(f"{'basis':22s} {'D':>5} {'bonf p':>10} {'cutoff |t|':>11} {'#sig':>6} {'%sig':>7}")
    for ax, (label, path) in zip(axes, SOURCES):
        d = torch.load(path, map_location="cpu")
        p = d["p_val"].numpy()                       # [D, 5]
        t = np.nan_to_num(np.abs(d["t_stat"].numpy()), nan=0.0)   # dead features -> 0
        D = p.shape[0]
        bonf = args.alpha / D
        sig = (p < bonf).any(axis=1)                 # significant for >=1 tau
        max_t = t.max(axis=1)                        # per-feature strongest signal
        # cutoff line = smallest max|t| that still passed (empirical significance boundary)
        cutoff = float(max_t[sig].min()) if sig.any() else float("nan")
        n_sig = int(sig.sum())
        print(f"{label:22s} {D:>5} {bonf:>10.1e} {cutoff:>11.2f} {n_sig:>6} {100*n_sig/D:>6.1f}%")

        disp = np.clip(max_t, 0, args.xmax)          # lump the long right tail into the last bin
        ax.hist(disp, bins=60, range=(0, args.xmax), color="#4477aa",
                density=True, alpha=0.85)
        ax.axvline(cutoff, color="crimson", ls="--", lw=1.5,
                   label=f"Bonferroni cutoff |t|={cutoff:.2f}")
        ax.set_ylabel("density")
        ax.set_title(f"{label}   —   {n_sig}/{D} significant ({100*n_sig/D:.1f}%) right of cutoff",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(f"per-feature max |t-statistic| across the 5 τ variables "
                        f"(clipped at {args.xmax:.0f}); right of the dashed line = NFP-significant")
    fig.suptitle("NFP test statistic vs Bonferroni cutoff (within-video covariance t-test, "
                 "N=3000 ball videos)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
