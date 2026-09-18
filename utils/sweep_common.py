"""
Shared helpers for the PCA/ICA dimensionality sweeps (VideoMAE / DINO / synthetic).

Kept dependency-light on purpose: importing this must NOT pull in VideoMAE/DINOv2, so the
synthetic sweep (pure linear algebra on cached reps) stays fast and model-free.
"""
import glob
from pathlib import Path

import numpy as np
import torch
from scipy import stats


def feat_count(mode, D):
    """Number of features produced by `mode` from D components."""
    return 2 * D if mode == "sign_split" else D


def load_train_subsample(train_dir, n_samples, seed):
    """Concatenate *.pt activation chunks in train_dir, subsample to n_samples rows."""
    files = sorted(Path(train_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No *.pt in {train_dir}")
    X = torch.cat([torch.load(f, map_location="cpu") for f in files], dim=0).float().numpy()
    rng = np.random.default_rng(seed)
    if X.shape[0] > n_samples:
        X = X[rng.choice(X.shape[0], n_samples, replace=False)]
    return X


def make_linear_encoder(mean, E, mode, device):
    """Return a callable x -> features (sign_split/abs/signed), all on `device`."""
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    E = torch.as_tensor(E, dtype=torch.float32, device=device)  # [D, 768]

    def encode(x):  # x: [..., 768]
        s = (x - mean) @ E.t()
        if mode == "sign_split":
            return torch.cat([torch.relu(s), torch.relu(-s)], dim=-1)
        if mode == "abs":
            return s.abs()
        return s
    return encode


def feature_dirs(E, mode):
    """Per-feature direction in input space [F, d] for a fitted encode matrix E [D, d]."""
    E = torch.as_tensor(E, dtype=torch.float32)
    if mode == "sign_split":
        return torch.cat([E, -E], dim=0)   # [2D, d], matches cat([relu(s), relu(-s)])
    return E                               # abs/signed: one feature per component


def within_video_covariance_all(feats, tau):
    """feats [B,T,F], tau [B,T,K] -> C [B,F,K] (within-video covariance)."""
    psi_c = feats - feats.mean(dim=1, keepdim=True)
    tau_c = tau - tau.mean(dim=1, keepdim=True)
    return torch.einsum("btd,btk->bdk", psi_c, tau_c) / feats.shape[1]


def nfp_stats(encode, ball, tau, mask, device, alpha=0.05, fixed_denom=768):
    """Within-video covariance + one-sample t-test over videos.

    Returns a dict with both significance masks and the t/p arrays:
        sig       : [F] bool, significant for >=1 tau at the adaptive bar alpha/F
        sig_fixed : [F] bool, significant at the D-independent bar alpha/fixed_denom
        t, p      : [F, 5]
        M         : F
        diag_dom  : bool (diagonal is row-max of the selectivity matrix for sig-in-row
                    features, adaptive bar). Selectivity = mean |C_mean| with tau
                    z-scored globally (effect size on a common scale), NOT mean |t| —
                    t measures consistency across videos, not response strength.
    """
    V, T, _ = ball.shape
    flat = ball.reshape(V * T, -1).to(device).float()
    feats = encode(flat).reshape(V, T, -1).cpu()
    feats = feats * mask.unsqueeze(-1).float()
    C = within_video_covariance_all(feats, tau).numpy()        # [V, F, 5]
    M = C.shape[1]
    bonf = alpha / M
    bonf_fixed = alpha / fixed_denom
    t = np.zeros((M, 5), np.float32); p = np.ones_like(t)
    for k in range(5):
        t[:, k], p[:, k] = stats.ttest_1samp(C[:, :, k], 0.0)
    sig = (p < bonf).any(axis=1)
    sig_fixed = (p < bonf_fixed).any(axis=1)
    # z-scored-tau effect size: Cov(psi, tau_k/sigma_k) = Cov(psi, tau_k)/sigma_k
    tau_sigma = tau.reshape(-1, 5).numpy().std(axis=0) + 1e-12
    C_sel = C.mean(axis=0) / tau_sigma[None, :]                # [F, 5]
    diag_dom = True
    for kr in range(5):
        m = p[:, kr] < bonf
        if m.sum() == 0:
            continue
        row = [np.abs(C_sel[m, kc]).mean() for kc in range(5)]
        if int(np.argmax(row)) != kr:
            diag_dom = False
    return {"sig": sig, "sig_fixed": sig_fixed, "t": t, "p": p,
            "M": M, "diag_dom": bool(diag_dom)}


def proj_fraction(dirs, W):
    """Fraction of each feature direction's squared norm inside the subspace spanned by W.

    dirs [F, d], W [k, d] (rows need not be orthonormal). Returns [F].
    """
    dirs = torch.as_tensor(dirs, dtype=torch.float32)
    W = torch.as_tensor(W, dtype=torch.float32)
    proj = dirs @ W.t()                                # [F, k]
    return ((proj ** 2).sum(1) / ((dirs ** 2).sum(1) + 1e-12)).numpy()
