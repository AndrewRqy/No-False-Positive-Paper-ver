"""
Cross-search: are the SAE's NFP-temporal features its most monosemantic (high-MS) ones?

Joins, for the SAME SAE, the per-feature monosemanticity score (MS) with the NFP
temporal-significance mask, and asks whether the temporal features have high MS.

Inputs (local SAE run):
    --ms_path   all_neurons_scores.pth : MS per feature, [F]
    --nfp_path  sae_nfp.pt             : NFP tensors incl. p_val [F, 5]

Prints MS statistics for NFP-significant ("temporal") vs the rest, where the temporal
features rank in the MS distribution, and a per-tau breakdown.
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms_path", default="local_runs/sae_val/ms_dinov2/all_neurons_scores.pth")
    ap.add_argument("--nfp_path", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    ms = torch.load(args.ms_path, map_location="cpu").numpy().astype(float)   # [F]
    d = torch.load(args.nfp_path, map_location="cpu")
    p = d["p_val"].numpy()                                                    # [F, 5]
    taus = d["tau_keys"]

    F = ms.shape[0]
    bonf = args.alpha / F
    sig = (p < bonf).any(axis=1)          # NFP-significant (temporal) features
    valid = ~np.isnan(ms)                 # exclude dead features

    def stats(mask, label):
        m = ms[mask & valid]
        print(f"{label:28s} n={m.size:5d}  MS mean={m.mean():.4f}  median={np.median(m):.4f}  "
              f"max={m.max():.4f}  >0.6:{int((m > 0.6).sum())}  >0.7:{int((m > 0.7).sum())}")

    print(f"D={F}  NFP bar p<{bonf:.2e}  |  temporal(sig)={int(sig.sum())}  dead={int((~valid).sum())}")
    stats(np.ones(F, bool), "ALL features")
    stats(sig, "NFP-temporal")
    stats(~sig, "NFP-nonsignificant")

    order = np.argsort(-np.where(valid, ms, -np.inf))     # high -> low MS
    rank = {f: i for i, f in enumerate(order)}
    sig_ranks = np.array([rank[f] for f in np.where(sig & valid)[0]])
    pct = 100 * (1 - np.median(sig_ranks) / valid.sum())
    print(f"Temporal features' median MS percentile: {pct:.1f}th (100 = highest MS)")
    for topn in (50, 100):
        print(f"  of the top-{topn} MS features, # temporal: {sum(1 for f in order[:topn] if sig[f])}")
    for k, t in enumerate(taus):
        mk = (p[:, k] < bonf) & valid
        if mk.sum():
            print(f"  sig[{t:9s}] n={int(mk.sum()):3d}  MS mean={ms[mk].mean():.4f}  max={ms[mk].max():.4f}")


if __name__ == "__main__":
    main()
