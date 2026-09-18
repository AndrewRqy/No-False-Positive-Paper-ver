"""
Sweep the number of PCA/ICA components D and measure how monosemanticity (MS)
and the NFP temporal-feature count vary with D.

Efficient design: the VideoMAE forward passes are independent of D, so we run them
ONCE and cache the raw activations, then sweep D as cheap linear algebra:
  - MS cache : per-token layer-11 acts for 800 SSv2-val clips  [800, 1568, 768]
  - NFP cache: ball-tracking acts for the 3000 ball videos     [3000, 8, 768] + tau + mask

PCA is nested (top-D of a full fit == the D-component fit), so PCA is fit once at the
largest D and sliced. ICA is NOT nested (FastICA picks a different solution per D), so
ICA is refit at each D.

Features use sign_split by default: each component c -> [ReLU(c), ReLU(-c)] (D -> 2D),
matching the ReLU SAE. MS is the weighted-pairwise-cosine metric (same as eval/metric.py);
NFP is within-video covariance + one-sample t-test (same as analysis/nfp_test.py).

Outputs a CSV + a printed table:  D, method, ms_mean, ms_std, ms_peak, nfp_sig, nfp_pct, diag_dominant
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.models.videomae import VideoMAE
from utils.datasets.ssv2 import SSv2Dataset
from utils import get_videomae_collate_fn
from nfp_testing.nfp_test import (
    NFPDataset, make_collate, extract_ball_tracking,
    within_video_covariance_all, TAU_KEYS, N_TEMPORAL, N_SPATIAL,
)

LAYER = 11
POINT = "post_mlp_residual"


# --------------------------------------------------------------------------- #
# Phase 1-2: cache raw VideoMAE activations (run once)
# --------------------------------------------------------------------------- #

def cache_ms_acts(model, ssv2_path, n_clips, batch, workers, device):
    """Per-token layer-11 acts for the first n_clips SSv2-val clips: [n_clips, 1568, 768]."""
    ds = SSv2Dataset(ssv2_path, split="val")
    ds = Subset(ds, list(range(min(n_clips, len(ds)))))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=get_videomae_collate_fn(model.processor))
    hook = f"{POINT}_{LAYER}"
    chunks = []
    for inputs in tqdm(dl, desc="MS cache (val)"):
        model.encode(inputs)
        chunks.append(model.register[hook][0].half())  # [B,1568,768] cpu fp16
    return torch.cat(chunks, dim=0)  # [N,1568,768]


def cache_nfp_acts(model, nfp_dir, batch, workers, device, tau_mode="first_frame"):
    """Ball-tracking acts + tau + mask for all ball videos: [V,8,768], [V,8,5], [V,8]."""
    ds = NFPDataset(Path(nfp_dir), tau_mode=tau_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=make_collate(model.processor))
    hook = f"{POINT}_{LAYER}"
    all_ball, all_tau, all_mask = [], [], []
    for inputs, tau, ball_tokens, _ in tqdm(dl, desc="NFP cache (ball)"):
        model.encode(inputs)
        acts = model.register[hook][0]                      # [B,1568,768]
        B = acts.shape[0]
        acts_sp = acts.view(B, N_TEMPORAL, N_SPATIAL, -1)
        ball, mask = extract_ball_tracking(acts_sp.to(device), ball_tokens.to(device))
        all_ball.append(ball.cpu()); all_tau.append(tau); all_mask.append(mask.cpu())
    return torch.cat(all_ball), torch.cat(all_tau), torch.cat(all_mask)


# --------------------------------------------------------------------------- #
# Fitting (reuse the train subsample already cached by the MS run)
# --------------------------------------------------------------------------- #

def load_train_subsample(train_dir, n_samples, seed):
    files = sorted(Path(train_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No *.pt in {train_dir} (run the MS pipeline first)")
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


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def ms_score(encode, ms_cache, embeds, device, clip_batch=50):
    """Weighted-pairwise-cosine MS (vectorised, matches eval/metric.py)."""
    N = ms_cache.shape[0]
    # max-pool encoded features over tokens, clip by clip to bound memory
    feats = []
    for i in range(0, N, clip_batch):
        x = ms_cache[i:i + clip_batch].to(device).float()       # [b,1568,768]
        f = encode(x)                                            # [b,1568,2D]
        feats.append(f.max(dim=1).values.cpu())                 # [b,2D]
    A = torch.cat(feats).to(device)                             # [N, M]
    # min-max scale per feature (as in metric.py)
    mn = A.min(0, keepdim=True).values
    mx = A.max(0, keepdim=True).values
    A = (A - mn) / (mx - mn).clamp_min(1e-12)                   # [N, M]
    # cosine-similarity matrix of DINOv2 embeddings (first-N clips align with the cache)
    e = torch.nn.functional.normalize(embeds[:N].to(device).float(), dim=1)
    S = e @ e.t()                                              # [N, N], S_ii = 1
    SA = S @ A                                                 # [N, M]
    sq = (A * A).sum(0)                                        # [M]
    num = (A * SA).sum(0) - sq                                 # sum_{i!=j} a_i a_j S_ij
    den = A.sum(0) ** 2 - sq                                   # sum_{i!=j} a_i a_j
    ms = torch.where(den > 0, num / den, torch.full_like(num, float("nan")))
    ms = ms.cpu().numpy()
    valid = ~np.isnan(ms)
    return float(np.mean(ms[valid])), float(np.std(ms[valid])), float(np.max(ms[valid])), int((~valid).sum())


def nfp_counts(encode, ball, tau, mask, device, alpha=0.05, fixed_denom=768):
    """Within-video covariance + t-test.

    Returns (#sig adaptive, #sig fixed-denom, M, diag_dominant). Two significance
    counts under two Bonferroni thresholds:
      - adaptive : alpha / M, where M = #features. The per-condition default; the
                   bar gets stricter as D grows, which mechanically suppresses the
                   count, so adaptive counts are NOT comparable across D.
      - fixed    : alpha / fixed_denom (default 768 = raw layer-11 dimensionality),
                   INDEPENDENT of D. Holding the denominator fixed isolates how the
                   raw count of temporal features changes with D from the changing
                   multiple-comparisons correction.
    Diagonal dominance uses the adaptive threshold (the per-condition definition).
    """
    V, T, _ = ball.shape
    flat = ball.reshape(V * T, -1).to(device).float()
    feats = encode(flat).reshape(V, T, -1).cpu()               # [V,T,M]
    feats = feats * mask.unsqueeze(-1).float()                 # re-zero off-screen
    C = within_video_covariance_all(feats, tau).numpy()        # [V, M, 5]
    M = C.shape[1]
    bonf = alpha / M
    bonf_fixed = alpha / fixed_denom
    t = np.zeros((M, 5), np.float32); p = np.ones_like(t)
    for k in range(5):
        t[:, k], p[:, k] = stats.ttest_1samp(C[:, :, k], 0.0)
    sig_any = (p < bonf).any(axis=1)
    sig_any_fixed = (p < bonf_fixed).any(axis=1)
    # diagonal dominance of the selectivity matrix (adaptive threshold)
    diag_dom = True
    for kr in range(5):
        m = p[:, kr] < bonf
        if m.sum() == 0:
            continue
        row = [np.abs(t[m, kc]).mean() for kc in range(5)]
        if int(np.argmax(row)) != kr:
            diag_dom = False
    return int(sig_any.sum()), int(sig_any_fixed.sum()), M, bool(diag_dom)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ssv2_path",   required=True)
    p.add_argument("--nfp_dir",     required=True)
    p.add_argument("--train_dir",   required=True)
    p.add_argument("--embeds_path", required=True)
    p.add_argument("--output_csv",  required=True)
    p.add_argument("--grid",        type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--methods",     nargs="+", default=["pca", "ica"],
                   choices=["pca", "ica"])
    p.add_argument("--modes",       nargs="+", default=["sign_split"],
                   choices=["sign_split", "abs", "signed"],
                   help="Run every (D, method) under each listed mode. The fit is "
                        "mode-independent, so evaluating multiple modes is nearly free.")
    p.add_argument("--fixed_denom", type=int, default=768,
                   help="D-independent Bonferroni denominator for the strict NFP cutoff "
                        "(alpha/fixed_denom, applied at every D; 768 = raw layer dim).")
    p.add_argument("--n_clips",     type=int, default=800)
    p.add_argument("--n_samples",   type=int, default=300_000)
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--ica_max_iter", type=int, default=1000)
    p.add_argument("--model_name",  default="MCG-NJU/videomae-base-finetuned-ssv2")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    grid = sorted(set(args.grid))

    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    model.attach(POINT, LAYER, sae=None)

    print("Caching raw activations (one VideoMAE pass each)...")
    ms_cache = cache_ms_acts(model, args.ssv2_path, args.n_clips, args.batch, args.workers, device)
    print(f"  MS cache: {tuple(ms_cache.shape)}")
    ball, tau, mask = cache_nfp_acts(model, args.nfp_dir, args.batch, args.workers, device)
    print(f"  NFP cache: ball {tuple(ball.shape)}")
    del model; torch.cuda.empty_cache()

    embeds = torch.load(args.embeds_path, map_location="cpu")
    print(f"  DINOv2 embeddings: {tuple(embeds.shape)}")

    print(f"Loading training subsample for fitting ({args.n_samples})...")
    Xtrain = load_train_subsample(args.train_dir, args.n_samples, args.seed)
    print(f"  train subsample: {Xtrain.shape}")

    # PCA is nested: fit once at the largest D, slice for each smaller D.
    pca_full = None
    if "pca" in args.methods:
        Dmax = max(grid)
        print(f"Fitting PCA once at D={Dmax} (nested; sliced for smaller D)...")
        pca_full = PCA(n_components=Dmax, svd_solver="randomized", random_state=args.seed).fit(Xtrain)

    def feat_count(mode, D):
        return 2 * D if mode == "sign_split" else D

    rows = []
    for D in grid:
        for method in args.methods:
            # Fit is mode-independent: fit once per (D, method), evaluate every mode.
            if method == "pca":
                mean, E = pca_full.mean_, pca_full.components_[:D]
            else:  # ica refit per D (not nested)
                try:
                    ica = FastICA(n_components=D, whiten="unit-variance", fun="logcosh",
                                  max_iter=args.ica_max_iter, tol=1e-3, random_state=args.seed)
                    ica.fit(Xtrain)
                    mean, E = ica.mean_, ica.components_
                    if (getattr(ica, "n_iter_", 0) or 0) >= args.ica_max_iter:
                        print(f"  D={D} ica [NOT CONVERGED]")
                except Exception as ex:
                    # FastICA can blow up numerically at high D (NaNs in decorrelation).
                    for mode in args.modes:
                        M = feat_count(mode, D)
                        rows.append((D, method, mode, M, float("nan"), float("nan"),
                                     float("nan"), -1, float("nan"), -1, float("nan"), False))
                    print(f"  D={D:>4} ica  FAILED: {type(ex).__name__}: {ex}")
                    continue
            for mode in args.modes:
                enc = make_linear_encoder(mean, E, mode, device)
                ms_mean, ms_std, ms_peak, _ = ms_score(enc, ms_cache, embeds, device)
                sig, sig_fx, M, diag = nfp_counts(enc, ball, tau, mask, device,
                                                  fixed_denom=args.fixed_denom)
                rows.append((D, method, mode, M, ms_mean, ms_std, ms_peak,
                             sig, 100 * sig / M, sig_fx, 100 * sig_fx / M, diag))
                print(f"  D={D:>4} {method:<3} {mode:<10} feats={M:<5} "
                      f"MS={ms_mean:.4f}±{ms_std:.4f} peak={ms_peak:.3f}  "
                      f"NFP sig={sig}/{M} ({100*sig/M:.2f}%) "
                      f"fix{args.fixed_denom}={sig_fx} ({100*sig_fx/M:.2f}%) diag={diag}")

    # write CSV
    out = Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("D,method,mode,n_features,ms_mean,ms_std,ms_peak,"
                f"nfp_sig,nfp_pct,nfp_sig_fixed{args.fixed_denom},nfp_pct_fixed{args.fixed_denom},"
                "diag_dominant\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f},"
                    f"{r[7]},{r[8]:.4f},{r[9]},{r[10]:.4f},{r[11]}\n")
    print(f"\nSaved -> {out}")

    # printed summary table
    fx = f"fix{args.fixed_denom} %"
    print(f"\n{'D':>5} {'method':<5} {'mode':<11} {'feats':>6} {'MS mean':>9} "
          f"{'MS peak':>8} {'NFP %sig':>9} {fx:>10} {'diag':>5}")
    for r in rows:
        print(f"{r[0]:>5} {r[1]:<5} {r[2]:<11} {r[3]:>6} {r[4]:>9.4f} {r[6]:>8.3f} "
              f"{r[8]:>8.2f}% {r[10]:>9.2f}% {str(r[11]):>5}")


if __name__ == "__main__":
    main()
