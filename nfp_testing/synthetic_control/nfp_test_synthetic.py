"""
NFP test on the synthetic positive/negative control.

Uses pre-generated synthetic representations h(V,t) where we know:
  - W_tau  directions carry TEMPORAL structure (tau changes within each video)
  - W_static directions carry STATIC/content structure: they may fluctuate frame-to-frame
    (static_jitter > 0) but are exactly decorrelated from tau within every video

The SAE was trained on these representations independently — it learned sparse
codes without knowing which directions are temporal.  We then ask:

  (1) Do any SAE features significantly covary with tau within videos?
      (They should — the SAE must have encoded the temporal W_tau structure
       to reconstruct h, since it dominates the signal.)

  (2) Are significant features actually aligned with W_tau and NOT W_static?
      (Ground truth check — validates the NFP test is identifying the
       right subspace, not just firing randomly.)

  (3) Is the selectivity matrix diagonal-dominant?
      (Same claim as for VideoMAE — a speed-encoding feature should have
       low covariance with direction, etc.)

This is a genuine test: the SAE had no knowledge of tau during training.
NFP must discover which features the SAE used for temporal encoding.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder, PCADict, ICADict, LinearDict

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def feature_input_directions(dic):
    """Per-feature direction in input (activation) space, as a [F, d] tensor.

    For an AutoEncoder this is the encoder weight rows. For a PCA/ICA LinearDict
    the feature directions are the rows of the encode matrix E; under sign_split
    each component contributes two features (+c, -c) whose directions are [E; -E],
    matching the order of LinearDict.encode (cat([relu(s), relu(-s)])).
    """
    if isinstance(dic, LinearDict):
        E = dic.E.detach().cpu().float()        # [n_components, d]
        if dic.mode == "sign_split":
            return torch.cat([E, -E], dim=0)    # [2*n_components, d]
        return E                                # abs / signed: one feature per component
    return dic.encoder.weight.data.cpu().float()


def within_video_covariance_all(feats: torch.Tensor,
                                 tau:   torch.Tensor) -> torch.Tensor:
    """feats [B,T,D_feat], tau [B,T,K] -> C [B,D_feat,K]"""
    psi_c = feats - feats.mean(dim=1, keepdim=True)
    tau_c = tau   - tau.mean(dim=1,   keepdim=True)
    return torch.einsum("btd,btk->bdk", psi_c, tau_c) / feats.shape[1]


def ground_truth_alignment(sae, W_tau, W_static, sig_mask, p_val, bonf, W_pos=None):
    """
    Report max-cosine and projection-fraction alignment with W_tau / W_static
    (and W_pos, the position-dependent static block, when present).
    """
    enc_W = feature_input_directions(sae)                   # [F, 768] tensor
    enc_W_np = enc_W.numpy()

    def max_subspace_cos(feat_idx, W):
        W_np   = W.numpy().astype(np.float32)
        W_unit = W_np / np.linalg.norm(W_np, axis=1, keepdims=True)
        e_unit = enc_W_np[feat_idx] / (np.linalg.norm(enc_W_np[feat_idx], axis=1, keepdims=True) + 1e-8)
        return np.abs(e_unit @ W_unit.T).max(axis=1)        # [len(feat_idx)]

    def proj_frac(W):
        # fraction of squared norm in the SUBSPACE SPANNED by W's rows. Rows need not
        # be orthonormal (the oblique-geometry ablation): project onto an orthonormal
        # basis of the span (QR). For orthonormal input this is identical to before.
        Q, _ = torch.linalg.qr(W.T.double())               # [d, k] orthonormal basis
        proj = enc_W.double() @ Q                          # [F, k]
        norm_sq = (enc_W.double() ** 2).sum(dim=1) + 1e-12
        return ((proj ** 2).sum(dim=1) / norm_sq).float().numpy()

    sig_idx    = np.where(sig_mask)[0]
    nonsig_idx = np.where(~sig_mask)[0]

    pf_tau    = proj_frac(W_tau)      # [F]
    pf_static = proj_frac(W_static)   # [F]
    pf_pos    = proj_frac(W_pos) if W_pos is not None else None

    print(f"\n--- Ground truth subspace alignment ---")
    header = (f"{'Group':<22} {'MaxCos W_tau':>13} {'MaxCos W_static':>16} "
              f"{'ProjFrac W_tau':>15} {'ProjFrac W_static':>18}")
    if pf_pos is not None:
        header += f" {'ProjFrac W_pos':>15}"
    print(header)

    for label, idx in [("Significant", sig_idx), ("Non-significant", nonsig_idx)]:
        if len(idx) == 0:
            print(f"  {label:<22} (none)")
            continue
        mc_tau    = max_subspace_cos(idx, W_tau).mean()
        mc_static = max_subspace_cos(idx, W_static).mean()
        pft = pf_tau[idx].mean()
        pfs = pf_static[idx].mean()
        row = f"  {label:<22} {mc_tau:>13.4f} {mc_static:>16.4f} {pft:>15.4f} {pfs:>18.4f}"
        if pf_pos is not None:
            row += f" {pf_pos[idx].mean():>15.4f}"
        print(row)

    if len(sig_idx) > 0 and len(nonsig_idx) > 0:
        ratio = pf_tau[sig_idx].mean() / (pf_tau[nonsig_idx].mean() + 1e-8)
        print(f"\n  Proj-frac W_tau ratio (sig / non-sig): {ratio:.1f}x")

    # Position-block accounting: flagged features should NOT live in W_pos.
    # (A W_pos-heavy flagged feature would mean NFP mistook motion-generated
    #  position content for kinematic encoding.)
    if pf_pos is not None and len(sig_idx) > 0:
        pos_heavy = (pf_pos[sig_idx] > pf_tau[sig_idx])
        print(f"  Significant features more W_pos- than W_tau-aligned: "
              f"{int(pos_heavy.sum())}/{len(sig_idx)}")

    # Per-tau breakdown
    print(f"\n--- Per-tau proj-frac W_tau (features sig for that tau) ---")
    print(f"  {'Tau':<12} {'N_sig':>6} {'ProjFrac W_tau':>15}")
    for k, name in enumerate(TAU_KEYS):
        sig_k = (p_val[:, k] < bonf)
        n_k   = sig_k.sum()
        pft_k = pf_tau[sig_k].mean() if n_k > 0 else float('nan')
        print(f"  {name:<12} {n_k:>6} {pft_k:>15.4f}")


def print_report(t_stat, p_val, C_mean, W_tau, W_static, sae, alpha=0.05, bonf=None,
                 W_pos=None, tau_sigma=None):
    D, K = t_stat.shape
    if bonf is None:
        bonf = alpha / D

    print(f"\n{'='*68}")
    print(f"NFP Test — Synthetic SAE (positive/negative control)")
    print(f"  Temporal directions : W_tau   {list(W_tau.shape)} — should trigger NFP")
    print(f"  Static directions   : W_static {list(W_static.shape)} — fluctuate but are "
          f"tau-decorrelated; should NOT trigger NFP")
    print(f"  Bonferroni threshold: p < {bonf:.2e}")
    print(f"{'='*68}")
    print(f"{'Tau':<12} {'Sig+':>6} {'Sig-':>6} {'Total%':>8} {'Mean|t|':>9}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*9}")

    sig_any = np.zeros(D, dtype=bool)
    for k, name in enumerate(TAU_KEYS):
        t_k = t_stat[:, k]
        p_k = p_val[:, k]
        sp  = int(((p_k < bonf) & (t_k > 0)).sum())
        sn  = int(((p_k < bonf) & (t_k < 0)).sum())
        sig_any |= (p_k < bonf)
        print(f"{name:<12} {sp:>6} {sn:>6} {100*(sp+sn)/D:>7.2f}% "
              f"{np.abs(t_k[np.isfinite(t_k)]).mean():>9.4f}")

    n_sig    = sig_any.sum()
    n_nonsig = (~sig_any).sum()
    print(f"\nFeatures significant for at least one tau : {n_sig} ({100*n_sig/D:.2f}%)")
    print(f"Features non-significant for all taus     : {n_nonsig} ({100*n_nonsig/D:.2f}%)")

    non_sig_C = C_mean[~sig_any]
    print(f"\nClaim (2) — non-significant features:")
    print(f"  Mean |C_mean| across all taus : {np.abs(non_sig_C).mean():.6f}")

    # Selectivity matrix — mean |C_mean| with tau z-scored globally (columns are
    # effect sizes on one common scale; raw column / global sigma_k is identical).
    # NOT mean |t|: t measures consistency, not response strength. Flagging (rows)
    # stays t-based; the matrix is a post-hoc effect-size diagnostic.
    C_sel = C_mean / tau_sigma[None, :] if tau_sigma is not None else C_mean
    unit = "z-scored tau" if tau_sigma is not None else "RAW tau — pass tau_sigma!"
    print(f"\nClaim (3) — selectivity (mean |C_mean| on {unit}, sig-in-row across columns):")
    header = f"{'':12}" + "".join(f"{n:>11}" for n in TAU_KEYS)
    print(header)
    for k_row, name_row in enumerate(TAU_KEYS):
        sig_mask_k = (p_val[:, k_row] < bonf)
        if sig_mask_k.sum() == 0:
            row = f"{name_row:<12}" + "".join(f"{'(none)':>11}" for _ in TAU_KEYS)
        else:
            row = f"{name_row:<12}"
            for k_col, _ in enumerate(TAU_KEYS):
                mc     = np.abs(C_sel[sig_mask_k, k_col]).mean()
                marker = " <--" if k_col == k_row else "    "
                row   += f"{mc:>10.4f}{marker}"
        print(row)

    # Ground truth alignment
    ground_truth_alignment(sae, W_tau, W_static, sig_any, p_val, bonf, W_pos=W_pos)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all_videos_path", required=True,
                   help="all_videos.pt saved by gen_synthetic_activations.py")
    p.add_argument("--matrices_path",   required=True,
                   help="matrices.pt saved by gen_synthetic_activations.py")
    p.add_argument("--sae_path",        required=True,
                   help="Dictionary checkpoint (SAE ae.pt, or PCA/ICA pca.pt/ica.pt).")
    p.add_argument("--sae_model",       default="standard",
                   choices=["standard", "pca", "ica"],
                   help="standard=ReLU SAE; pca/ica=linear decomposition fit on the "
                        "synthetic representations.")
    p.add_argument("--decomp_mode",     default="sign_split",
                   choices=["sign_split", "abs", "signed"],
                   help="For pca/ica: how signed components map to features.")
    p.add_argument("--output_path",     required=True)
    p.add_argument("--alpha",           default=0.05, type=float)
    p.add_argument("--device",          default="cuda:0")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading synthetic data: {args.all_videos_path}")
    data     = torch.load(args.all_videos_path, map_location="cpu")
    h        = data["h"]        # [N, 8, D]
    tau      = data["tau"]      # [N, 8, 5]  original unnormalized tau

    print(f"Loading ground truth matrices: {args.matrices_path}")
    mat      = torch.load(args.matrices_path, map_location="cpu")
    W_tau    = mat["W_tau"]           # [5, D]
    W_static = mat["W_static"]        # [N_STATIC, D]
    W_pos    = mat.get("W_pos")       # [N_POS, D] or None (older data)

    print(f"Loading dictionary ({args.sae_model}): {args.sae_path}")
    if args.sae_model == "standard":
        sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    elif args.sae_model == "pca":
        sae = PCADict.from_pretrained(args.sae_path, device=device, mode=args.decomp_mode)
    elif args.sae_model == "ica":
        sae = ICADict.from_pretrained(args.sae_path, device=device, mode=args.decomp_mode)
    else:
        raise ValueError(f"Unknown sae_model: {args.sae_model}")
    sae.eval()

    N, T, D = h.shape
    print(f"\nDataset  : {N} videos, {T} temporal steps, {D}-dim reps")
    print(f"SAE size : {sae.dict_size} features")

    # Run SAE encoder on all synthetic representations
    with torch.no_grad():
        feats_flat = sae.encode(h.reshape(N * T, D).to(device))   # [N*8, F]
    feats = feats_flat.cpu().reshape(N, T, -1)                     # [N, 8, F]

    pct_active = (feats > 0).float().mean().item()
    print(f"Mean feature activation rate: {100*pct_active:.1f}%")

    # Within-video covariance [N, F, 5]
    print("Computing within-video covariances...")
    C    = within_video_covariance_all(feats, tau)
    C_np = C.numpy()

    # One-sample t-test across videos
    print("Running t-tests...")
    F      = sae.dict_size
    t_stat = np.zeros((F, len(TAU_KEYS)), dtype=np.float32)
    p_val  = np.ones_like(t_stat)
    for k in range(len(TAU_KEYS)):
        t_stat[:, k], p_val[:, k] = stats.ttest_1samp(C_np[:, :, k], 0.0)

    C_mean = C_np.mean(axis=0)   # [F, 5]
    # global per-tau std: converts selectivity columns to z-scored-tau units
    tau_sigma = tau.reshape(-1, len(TAU_KEYS)).numpy().std(axis=0)
    print_report(t_stat, p_val, C_mean, W_tau, W_static, sae, alpha=args.alpha,
                 bonf=args.alpha / sae.dict_size, W_pos=W_pos, tau_sigma=tau_sigma)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "C":         C,
        "tau":       tau,
        "t_stat":    torch.from_numpy(t_stat),
        "p_val":     torch.from_numpy(p_val),
        "C_mean":    torch.from_numpy(C_mean),
        "W_tau":     W_tau,
        "W_static":  W_static,
        "W_pos":     W_pos,
        "tau_keys":  TAU_KEYS,
    }, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
