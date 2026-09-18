"""
Fit PCA and ICA decompositions on VideoMAE layer-11 activations (the same corpus
used to train the SAE), and save them as Dictionary checkpoints that the
activation-extraction pipeline can load via PCADict / ICADict.from_pretrained.

These serve as linear-decomposition baselines for the monosemanticity-score (MS)
comparison against the SAE: PCA = orthogonal variance-maximizing basis,
ICA = statistically-independent (max non-Gaussian) basis.

Fitting is done on a random subsample of the (very large) training corpus. Per
standard practice for FastICA at this scale, a few hundred thousand vectors is
ample: the unmixing matrix has activation_dim^2 entries (~590k for d=768), so
~500k samples comfortably over-determines it, and more samples mainly cost time.
PCA and ICA are fit on the *same* subsample for a symmetric comparison.

Checkpoint format (torch.save dict):
    method          : "pca" | "ica"
    activation_dim  : int (768)
    n_components    : int
    mean            : (activation_dim,)               data mean
    E               : (n_components, activation_dim)  encode matrix  s=(x-mean)@E.T
    D               : (n_components, activation_dim)  decode matrix  x_hat=s@D+mean
"""
import os
import glob
import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA


def get_args_parser():
    p = argparse.ArgumentParser("Fit PCA/ICA on activations", add_help=False)
    p.add_argument("--activations_dir", required=True, type=str,
                   help="Directory of *_part*.pt activation chunks (SAE training corpus).")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--n_components", default=768, type=int,
                   help="Number of components. Capped at activation_dim (768).")
    p.add_argument("--n_samples", default=500_000, type=int,
                   help="Number of activation vectors to subsample for fitting.")
    p.add_argument("--max_chunks", default=-1, type=int,
                   help="Load at most this many chunk files (-1 = all). Each chunk is "
                        "~50k vectors; loading ~12 chunks already exceeds the default "
                        "n_samples, so this caps I/O without biasing the subsample.")
    p.add_argument("--methods", nargs="+", default=["pca", "ica"],
                   choices=["pca", "ica"])
    p.add_argument("--ica_max_iter", default=2000, type=int)
    p.add_argument("--ica_tol", default=1e-3, type=float)
    p.add_argument("--l2_normalize", action="store_true",
                   help="L2-normalize each activation vector before fitting (improves "
                        "ICA conditioning; matches the ICA-Lens preprocessing).")
    p.add_argument("--seed", default=0, type=int)
    return p


def load_subsample(activations_dir, n_samples, max_chunks, seed):
    files = sorted(
        (f for f in glob.glob(os.path.join(activations_dir, "*.pt"))
         if not os.path.basename(f).startswith("all")),
        key=lambda x: int(x.split("_part")[-1].split(".pt")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *_part*.pt chunks found in {activations_dir}")

    rng = np.random.default_rng(seed)
    # Shuffle chunk order so a chunk cap samples across the corpus, not just the head.
    order = rng.permutation(len(files))
    if max_chunks > 0:
        order = order[:max_chunks]

    collected = []
    total = 0
    for idx in order:
        t = torch.load(files[idx], map_location="cpu")
        collected.append(t)
        total += t.shape[0]
        if total >= n_samples:
            break

    X = torch.cat(collected, dim=0).float().numpy()
    if X.shape[0] > n_samples:
        sel = rng.choice(X.shape[0], size=n_samples, replace=False)
        X = X[sel]
    print(f"Loaded subsample of shape {X.shape} from {len(collected)} chunk(s)")
    return X


def fit_pca(X, n_components, seed):
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(X)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"[PCA] {n_components} components, cumulative explained variance = {evr:.4f}")
    return {
        "method": "pca",
        "activation_dim": X.shape[1],
        "n_components": n_components,
        "mean": torch.from_numpy(pca.mean_).float(),
        "E": torch.from_numpy(pca.components_).float(),          # (n_comp, d)
        "D": torch.from_numpy(pca.components_).float(),          # PCA: decode = components
        "explained_variance_ratio": evr,
    }


def fit_ica(X, n_components, max_iter, tol, seed):
    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten="unit-variance",   # FastICA requires whitening; done internally via PCA
        fun="logcosh",
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
    )
    ica.fit(X)
    if getattr(ica, "n_iter_", None) is not None:
        if ica.n_iter_ < max_iter:
            status = "converged"
        else:
            status = "DID NOT CONVERGE — consider fewer components or larger max_iter"
        print(f"[ICA] n_iter = {ica.n_iter_} / {max_iter} ({status})")
    return {
        "method": "ica",
        "activation_dim": X.shape[1],
        "n_components": n_components,
        "mean": torch.from_numpy(ica.mean_).float(),
        "E": torch.from_numpy(ica.components_).float(),          # unmixing (n_comp, d)
        "D": torch.from_numpy(ica.mixing_.T).float(),            # mixing.T (n_comp, d)
        "n_iter": int(getattr(ica, "n_iter_", -1)),
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    X = load_subsample(args.activations_dir, args.n_samples, args.max_chunks, args.seed)

    if args.l2_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X = X / norms
        print("Applied L2 normalization to activation vectors")

    n_components = min(args.n_components, X.shape[1])

    if "pca" in args.methods:
        ckpt = fit_pca(X, n_components, args.seed)
        # persist the preprocessing flag so encode() applies the same normalization
        ckpt["l2_normalize"] = bool(args.l2_normalize)
        out = os.path.join(args.output_dir, "pca.pt")
        torch.save(ckpt, out)
        print(f"Saved PCA checkpoint to {out}")

    if "ica" in args.methods:
        ckpt = fit_ica(X, n_components, args.ica_max_iter, args.ica_tol, args.seed)
        ckpt["l2_normalize"] = bool(args.l2_normalize)
        out = os.path.join(args.output_dir, "ica.pt")
        torch.save(ckpt, out)
        print(f"Saved ICA checkpoint to {out}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
