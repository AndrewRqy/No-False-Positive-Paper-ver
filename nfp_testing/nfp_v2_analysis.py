"""
Experiment F3 — NFP on the v2 decorrelated dataset, compared head-to-head with v1.

Same SAE (trained on SSv2 activations, unchanged), same NFP statistic, same Bonferroni
bar; the only change is the probe stimulus (v2: all 10 pairwise tau couplings zeroed by
the joint LP design; v1: direction-vel_y coupling +0.68).

Predictions being tested (from FINDINGS steps 34-36):
  P1. cos(c_bar[direction], c_bar[vel_y]) drops from 0.96 (v1) to ~0.
  P2. The direction row of the selectivity matrix (mean |C| on z-scored tau) becomes
      diagonal-dominant (v1: peaked on vel_y).
  P3. More direction-dominant features than v1's 2/85.
Also reported: flag-set overlap v1 vs v2, per-tau breakdowns, the full selectivity
matrix, and the c_bar pairwise cosine matrix for v2.

Usage (from sae-for-vlm/):
  python analysis/nfp_v2_analysis.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def nfp_stats(ball, tau, mask, sae, device):
    N, T, _ = ball.shape
    with torch.no_grad():
        feats = sae.encode(ball.reshape(N * T, -1).to(device).float()).cpu().reshape(N, T, -1)
    feats = feats * mask.unsqueeze(-1).float()
    psi_c = feats - feats.mean(1, keepdim=True)
    tau_c = tau - tau.mean(1, keepdim=True)
    C = torch.einsum("btd,btk->bdk", psi_c, tau_c).numpy() / T
    D = C.shape[1]
    t = np.zeros((D, 5), np.float32); p = np.ones_like(t)
    for k in range(5):
        t[:, k], p[:, k] = stats.ttest_1samp(C[:, :, k], 0.0)
    return C, t, p


def report(tag, C, t, p, tau, bonf):
    D = t.shape[0]
    sig_any = (p < bonf).any(1)
    sig = [int(i) for i in np.where(sig_any)[0]]
    dom = {i: TAU[int(np.argmax(np.abs(np.nan_to_num(t[i]))))] for i in sig}
    print(f"\n=== {tag}: {len(sig)} flagged ({100*len(sig)/D:.2f}%) ===")
    print("  per-tau significant (any) / dominant:")
    for k, name in enumerate(TAU):
        n_sig = int((p[:, k] < bonf).sum())
        n_dom = sum(1 for i in sig if dom[i] == name)
        print(f"    {name:<10} sig={n_sig:<4} dominant={n_dom}")
    sigma = tau.reshape(-1, 5).numpy().std(axis=0)
    C_sel = C.mean(axis=0) / sigma[None, :]
    print("  selectivity (mean |C| on z-scored tau, sig-in-row x columns):")
    print(f"  {'':12}" + "".join(f"{n:>11}" for n in TAU))
    diag_flags = {}
    for kr, nr in enumerate(TAU):
        m = p[:, kr] < bonf
        if m.sum() == 0:
            print(f"  {nr:<12}" + "".join(f"{'(none)':>11}" for _ in TAU))
            continue
        vals = [float(np.abs(C_sel[m, kc]).mean()) for kc in range(5)]
        dd = int(np.argmax(vals)) == kr
        diag_flags[nr] = dd
        row = f"  {nr:<12}"
        for kc, v in enumerate(vals):
            row += f"{v:>10.4f}" + ("<" if kc == kr else " ")
        print(row + ("  DIAG" if dd else f"  no (max={TAU[int(np.argmax(vals))]})"))
    return sig, dom, diag_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--v1_acts", default="local_runs/nfp_results/ball_raw_acts.pt")
    ap.add_argument("--v2_acts", default="local_runs/nfp_results/ball_raw_acts_v2.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expF3_nfp_v2.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    bonf = 0.05 / sae.dict_size

    out = {}
    cbars = {}
    results = {}
    for tag, path in [("v1", args.v1_acts), ("v2", args.v2_acts)]:
        d = torch.load(path, map_location="cpu")
        ball, tau, mask = d["ball"].float(), d["tau"].float(), d["mask"]
        C, t, p = nfp_stats(ball, tau, mask, sae, device)
        sig, dom, diag = report(tag, C, t, p, tau, bonf)
        results[tag] = {"sig": sig, "dom": dom}
        out[tag] = {"n_flagged": len(sig),
                    "per_tau_dominant": {n: sum(1 for i in sig if dom[i] == n) for n in TAU},
                    "diag_dominant": diag}
        # c_bar for this stimulus
        hc = ball - ball.mean(1, keepdim=True)
        tc = tau - tau.mean(1, keepdim=True)
        cb = (torch.einsum("btd,btk->bdk", hc, tc) / ball.shape[1]).mean(0)  # [768,5]
        cbars[tag] = cb

    # P1: c_bar cosine structure
    print("\n=== c_bar pairwise cosines ===")
    for tag in ["v1", "v2"]:
        cb = cbars[tag]
        U = cb / cb.norm(dim=0, keepdim=True)
        cosm = (U.T @ U).numpy()
        print(f"  {tag}:")
        for i, n in enumerate(TAU):
            print(f"    {n:<10}" + "".join(f"{cosm[i, j]:+8.2f}" for j in range(5)))
        out[tag]["cbar_cos_dir_vely"] = round(float(cosm[TAU.index("direction"),
                                                         TAU.index("vel_y")]), 3)

    # flag overlap
    s1, s2 = set(results["v1"]["sig"]), set(results["v2"]["sig"])
    print(f"\nflag overlap: v1={len(s1)}, v2={len(s2)}, intersection={len(s1 & s2)}")
    out["overlap"] = {"v1": len(s1), "v2": len(s2), "both": len(s1 & s2)}
    new_dir = [i for i in results["v2"]["sig"]
               if results["v2"]["dom"][i] == "direction"]
    print(f"v2 direction-dominant features: {sorted(new_dir)}")
    out["v2_direction_features"] = sorted(new_dir)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
