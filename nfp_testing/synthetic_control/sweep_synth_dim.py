"""
Sweep the number of PCA/ICA components D on the SYNTHETIC positive control.

Mirrors analysis/sweep_pca_ica_dim.py, but the monosemanticity score does NOT apply to the
synthetic reps (no natural frames). The synthetic analog is ground-truth alignment, so in
place of MS this sweep reports, per D, the mean projection fraction of the NFP-flagged features
onto the temporal W_tau vs static W_static subspaces - i.e. whether a basis's "temporal"
features actually live in the planted temporal directions, and how that changes with D.

Everything is linear algebra on the cached synthetic reps h (no model forward), so the whole
sweep is fast and model-free. PCA is nested (fit once, slice); ICA is refit per D.

NFP significance is reported under both the adaptive alpha/M and the D-independent alpha/768
cutoffs (see sweep_pca_ica_dim.py / summary.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.sweep_common import (
    feat_count,
    make_linear_encoder,
    nfp_stats,
    feature_dirs,
    proj_fraction,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all_videos_path", required=True, help="all_videos.pt (h + tau)")
    p.add_argument("--matrices_path", required=True, help="matrices.pt (W_tau, W_static)")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--grid", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 768])
    p.add_argument("--methods", nargs="+", default=["pca", "ica"], choices=["pca", "ica"])
    p.add_argument(
        "--modes",
        nargs="+",
        default=["sign_split", "signed"],
        choices=["sign_split", "abs", "signed"],
    )
    p.add_argument("--fixed_denom", type=int, default=768)
    p.add_argument("--n_samples", type=int, default=500_000)
    p.add_argument("--ica_max_iter", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    grid = sorted(set(args.grid))

    data = torch.load(args.all_videos_path, map_location="cpu")
    h, tau = data["h"].float(), data["tau"].float()  # [N,T,768], [N,T,5]
    mask = torch.ones(h.shape[0], h.shape[1])  # synthetic: always on-screen
    mat = torch.load(args.matrices_path, map_location="cpu")
    W_tau, W_static = mat["W_tau"].float(), mat["W_static"].float()
    print(
        f"Synthetic reps h {tuple(h.shape)}; W_tau {tuple(W_tau.shape)}, "
        f"W_static {tuple(W_static.shape)}"
    )

    Xtrain = h.reshape(-1, h.shape[-1]).numpy()
    rng = np.random.default_rng(args.seed)
    if Xtrain.shape[0] > args.n_samples:
        Xtrain = Xtrain[rng.choice(Xtrain.shape[0], args.n_samples, replace=False)]
    print(f"  fit corpus: {Xtrain.shape}")

    pca_full = None
    if "pca" in args.methods:
        Dmax = max(grid)
        print(f"Fitting PCA once at D={Dmax} (nested)...")
        pca_full = PCA(n_components=Dmax, svd_solver="randomized", random_state=args.seed).fit(
            Xtrain
        )

    rows = []
    for D in grid:
        for method in args.methods:
            if method == "pca":
                mean, E = pca_full.mean_, pca_full.components_[:D]
            else:
                try:
                    ica = FastICA(
                        n_components=D,
                        whiten="unit-variance",
                        fun="logcosh",
                        max_iter=args.ica_max_iter,
                        tol=1e-3,
                        random_state=args.seed,
                    )
                    ica.fit(Xtrain)
                    mean, E = ica.mean_, ica.components_
                    if (getattr(ica, "n_iter_", 0) or 0) >= args.ica_max_iter:
                        print(f"  D={D} ica [NOT CONVERGED]")
                except Exception as ex:
                    for mode in args.modes:
                        rows.append(
                            (
                                D,
                                method,
                                mode,
                                feat_count(mode, D),
                                -1,
                                float("nan"),
                                -1,
                                float("nan"),
                                float("nan"),
                                float("nan"),
                                False,
                            )
                        )
                    print(f"  D={D:>4} ica  FAILED: {type(ex).__name__}: {ex}")
                    continue
            for mode in args.modes:
                enc = make_linear_encoder(mean, E, mode, device)
                st = nfp_stats(enc, h, tau, mask, device, fixed_denom=args.fixed_denom)
                M = st["M"]
                sig = st["sig"]
                nsig = int(sig.sum())
                sig_fx = int(st["sig_fixed"].sum())
                dirs = feature_dirs(E, mode)  # [M, 768]
                pf_tau = proj_fraction(dirs, W_tau)
                pf_static = proj_fraction(dirs, W_static)
                pft = float(pf_tau[sig].mean()) if nsig else float("nan")
                pfs = float(pf_static[sig].mean()) if nsig else float("nan")
                rows.append(
                    (
                        D,
                        method,
                        mode,
                        M,
                        nsig,
                        100 * nsig / M,
                        sig_fx,
                        100 * sig_fx / M,
                        pft,
                        pfs,
                        st["diag_dom"],
                    )
                )
                print(
                    f"  D={D:>4} {method:<3} {mode:<10} feats={M:<5} "
                    f"NFP sig={nsig}/{M} ({100*nsig/M:.2f}%) "
                    f"fix{args.fixed_denom}={sig_fx} ({100*sig_fx/M:.2f}%) "
                    f"PF[Wtau]={pft:.3f} PF[Wstat]={pfs:.3f} diag={st['diag_dom']}"
                )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(
            "D,method,mode,n_features,nfp_sig,nfp_pct,"
            f"nfp_sig_fixed{args.fixed_denom},nfp_pct_fixed{args.fixed_denom},"
            "projfrac_wtau_sig,projfrac_wstatic_sig,diag_dominant\n"
        )
        for r in rows:
            f.write(
                f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]},{r[7]:.4f},"
                f"{r[8]:.6f},{r[9]:.6f},{r[10]}\n"
            )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
