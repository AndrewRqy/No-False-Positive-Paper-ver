"""Distribution of the NFP test statistic across the SAE dictionaries.

For each SAE dictionary, plots the histogram of max_k |t_k| over all features,
so the whole population is visible rather than only the count above the
Bonferroni threshold. Dead features (non-finite t) are excluded, matching the
NFP test. The threshold and the flagged count are drawn on each panel.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

E = "local_runs/expansion/"
MODELS = [
    ("VideoMAE finetuned", "local_runs/nfp_results/sae_nfp_v2.pt", 6144),
    ("V-JEPA2", E + "vjepa2_sae_l23.pt", 8192),
    ("TimeSformer", E + "timesformer_sae_l11.pt", 6144),
    ("VideoMAE", E + "videomae_pretrain_sae.pt", 6144),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="local_runs/expansion/tscore_dist")
    ap.add_argument("--n_videos", default=3000, type=int)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.4), sharex=True)
    for ax, (name, path, D) in zip(axes.flat, MODELS):
        d = torch.load(path, map_location="cpu", weights_only=False)
        t = d["t_stat"]
        finite = torch.isfinite(t).all(dim=1)
        maxabs = t.abs().max(dim=1).values[finite].numpy()
        thr = stats.t.ppf(1 - 0.05 / (2 * D), args.n_videos - 1)
        n_flag = int((maxabs > thr).sum())
        n_dead = int((~finite).sum())

        bins = np.linspace(0, max(maxabs.max(), thr * 1.1), 60)
        ax.hist(maxabs[maxabs <= thr], bins=bins, color="#9ecae1", edgecolor="none")
        ax.hist(maxabs[maxabs > thr], bins=bins, color="#08519c", edgecolor="none")
        ax.axvline(thr, color="k", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"{name} ({n_flag} of {D} flagged)", fontsize=9)
        ax.text(thr * 1.03, ax.get_ylim()[1] * 0.35,
                f"threshold {thr:.1f}", fontsize=7, rotation=90, va="top")
        ax.tick_params(labelsize=7)
        print(f"{name}: flagged={n_flag} dead={n_dead} "
              f"median max|t|={np.median(maxabs):.2f} p99={np.percentile(maxabs, 99):.1f} "
              f"max={maxabs.max():.1f} thr={thr:.2f}")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\max_k |t_k|$ per feature", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("features (log scale)", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=150)
    print(f"saved -> {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
