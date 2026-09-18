"""
Sweep the number of PCA/ICA components D on the DINOv2 NEGATIVE control.

Mirrors analysis/sweep_pca_ica_dim.py, but for DINOv2 the monosemanticity score does NOT
apply (DINOv2 is itself the MS image encoder -> circular), so this sweep is NFP-only:
how the count of "temporal" features varies with D. DINOv2 processes each frame
independently, so the expected answer is 0 at every D for every basis.

Efficient design: the DINOv2 ball-token activations are cached ONCE, then D is swept as
linear algebra. PCA is nested (fit once at max D, slice); ICA is refit per D. The fit corpus
is an independent set of DINO patch activations (extract_dino_patch_activations.py output),
matching how the DINO SAE was trained.

NFP significance is reported under two Bonferroni cutoffs: the adaptive alpha/M and the
D-independent alpha/768 (see sweep_pca_ica_dim.py / summary.md).
"""
import argparse
import sys
from pathlib import Path

import torch
from sklearn.decomposition import PCA, FastICA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

sys.path.insert(0, str(Path(__file__).parent.parent))
from nfp_testing.nfp_test_dino_patch import NFPDataset, make_collate, N_TEMPORAL
from utils.sweep_common import (
    feat_count, load_train_subsample, make_linear_encoder, nfp_stats,
)


def cache_dino_ball(model, processor, dataset_dir, batch, workers, device):
    """DINOv2 ball-token acts + tau + mask for all ball videos: [V,8,768], [V,8,5], [V,8]."""
    ds = NFPDataset(Path(dataset_dir))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=make_collate(processor))
    all_ball, all_tau, all_mask = [], [], []
    for inputs, tau, ball_tokens, _ in tqdm(dl, desc="DINO ball cache"):
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in inputs.items()})
            hidden = out.last_hidden_state                      # [B*8, 197, 768]
        flat_toks = ball_tokens.reshape(-1).to(device)          # [B*8]
        mask = (flat_toks >= 0)
        patch_idx = flat_toks.clamp(min=0) + 1                  # CLS at index 0
        br = torch.arange(hidden.shape[0], device=device)
        ball = hidden[br, patch_idx, :] * mask.float().unsqueeze(-1)   # [B*8, 768]
        B = tau.shape[0]
        all_ball.append(ball.reshape(B, N_TEMPORAL, -1).cpu())
        all_tau.append(tau)
        all_mask.append(mask.reshape(B, N_TEMPORAL).cpu())
    return torch.cat(all_ball), torch.cat(all_tau), torch.cat(all_mask)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="Ball NFP dataset (data/output/nfp)")
    p.add_argument("--train_dir",   required=True,
                   help="Dir of DINO patch activation chunks for fitting (extract_dino_patch_activations.py)")
    p.add_argument("--output_csv",  required=True)
    p.add_argument("--grid",        type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 768])
    p.add_argument("--methods",     nargs="+", default=["pca", "ica"], choices=["pca", "ica"])
    p.add_argument("--modes",       nargs="+", default=["sign_split", "signed"],
                   choices=["sign_split", "abs", "signed"])
    p.add_argument("--fixed_denom", type=int, default=768)
    p.add_argument("--n_samples",   type=int, default=500_000)
    p.add_argument("--batch",       type=int, default=8)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--ica_max_iter", type=int, default=2000)
    p.add_argument("--model_name",  default="dinov2-base")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    grid = sorted(set(args.grid))

    print(f"Loading DINOv2: facebook/{args.model_name}")
    processor = AutoImageProcessor.from_pretrained(f"facebook/{args.model_name}")
    model = Dinov2Model.from_pretrained(f"facebook/{args.model_name}").to(device)
    model.eval()

    print("Caching DINOv2 ball-token activations (one pass)...")
    ball, tau, mask = cache_dino_ball(model, processor, args.dataset_dir,
                                      args.batch, args.workers, device)
    print(f"  ball cache: {tuple(ball.shape)}")
    del model; torch.cuda.empty_cache()

    print(f"Loading fit corpus from {args.train_dir} (subsample {args.n_samples})...")
    Xtrain = load_train_subsample(args.train_dir, args.n_samples, args.seed)
    print(f"  fit subsample: {Xtrain.shape}")

    pca_full = None
    if "pca" in args.methods:
        Dmax = max(grid)
        print(f"Fitting PCA once at D={Dmax} (nested)...")
        pca_full = PCA(n_components=Dmax, svd_solver="randomized", random_state=args.seed).fit(Xtrain)

    rows = []
    for D in grid:
        for method in args.methods:
            if method == "pca":
                mean, E = pca_full.mean_, pca_full.components_[:D]
            else:
                try:
                    ica = FastICA(n_components=D, whiten="unit-variance", fun="logcosh",
                                  max_iter=args.ica_max_iter, tol=1e-3, random_state=args.seed)
                    ica.fit(Xtrain)
                    mean, E = ica.mean_, ica.components_
                    if (getattr(ica, "n_iter_", 0) or 0) >= args.ica_max_iter:
                        print(f"  D={D} ica [NOT CONVERGED]")
                except Exception as ex:
                    for mode in args.modes:
                        rows.append((D, method, mode, feat_count(mode, D), -1, float("nan"),
                                     -1, float("nan"), False))
                    print(f"  D={D:>4} ica  FAILED: {type(ex).__name__}: {ex}")
                    continue
            for mode in args.modes:
                enc = make_linear_encoder(mean, E, mode, device)
                st = nfp_stats(enc, ball, tau, mask, device, fixed_denom=args.fixed_denom)
                M = st["M"]; sig = int(st["sig"].sum()); sig_fx = int(st["sig_fixed"].sum())
                rows.append((D, method, mode, M, sig, 100 * sig / M,
                             sig_fx, 100 * sig_fx / M, st["diag_dom"]))
                print(f"  D={D:>4} {method:<3} {mode:<10} feats={M:<5} "
                      f"NFP sig={sig}/{M} ({100*sig/M:.2f}%) "
                      f"fix{args.fixed_denom}={sig_fx} ({100*sig_fx/M:.2f}%) diag={st['diag_dom']}")

    out = Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("D,method,mode,n_features,nfp_sig,nfp_pct,"
                f"nfp_sig_fixed{args.fixed_denom},nfp_pct_fixed{args.fixed_denom},diag_dominant\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]},{r[7]:.4f},{r[8]}\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
