"""
Find narrow-scope ("monosemantic temporal") SAE features as steering candidates.

A good steering target activates to ONE temporal concept and stays low on the others.
The NFP test already measures, per feature, the coupling |t-stat| to each of the 5 tau
(speed, vel_x, vel_y, accel_mag, direction). So narrow scope = one large |t| with all
others near zero. We score each significant feature by:

  dom_tau      = argmax_k |t_k|            (the concept it couples to most)
  dom_t        = |t_dom|                   (how strong that coupling is)
  runnerup_t   = 2nd largest |t|
  selectivity  = dom_t / runnerup_t        (>>1 = exclusive to dom_tau)
  n_sig_tau    = # tau with p < 0.05/6144  (1 = significant for a single concept)

Strong candidate = n_sig_tau == 1 AND high selectivity AND solid dom_t. We corroborate
with the ground-truth aggregates in feature_summary.csv (top-50 videos): an exclusive
acceleration feature should have its top videos consistently accelerating while their
speed / heading vary freely.

Output: local_runs/nfp_feature_gifs/steering_candidates.csv (ranked), + console report.
"""

import csv
from pathlib import Path

import numpy as np
import torch

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
NFP = "local_runs/nfp_results/sae_nfp.pt"
SUMMARY = "local_runs/nfp_feature_gifs/feature_summary.csv"
OUT = "local_runs/nfp_feature_gifs/steering_candidates.csv"


def main():
    nfp = torch.load(NFP, map_location="cpu")
    t = nfp["t_stat"].numpy()  # [6144, 5]
    p = nfp["p_val"].numpy()
    D = t.shape[0]
    bonf = 0.05 / D
    sig_any = (p < bonf).any(1)
    sig_idx = np.where(sig_any)[0]

    # load ground-truth aggregates keyed by feature id string
    summ = {}
    if Path(SUMMARY).exists():
        for r in csv.DictReader(open(SUMMARY)):
            summ[r["feature"]] = r

    rows = []
    for i in sig_idx:
        at = np.abs(t[i])  # |t| per tau
        order = np.argsort(-at)
        dom_k = int(order[0])
        dom_t = float(at[dom_k])
        run_t = float(at[order[1]])
        n_sig = int((p[i] < bonf).sum())
        sel = dom_t / max(run_t, 1e-6)
        rows.append(
            {
                "feature": f"feat{i:05d}",
                "idx": int(i),
                "dom_tau": TAU_KEYS[dom_k],
                "dom_t": round(dom_t, 2),
                "dom_t_signed": round(float(t[i, dom_k]), 2),
                "runnerup_tau": TAU_KEYS[int(order[1])],
                "runnerup_t": round(run_t, 2),
                "selectivity": round(sel, 2),
                "n_sig_tau": n_sig,
                **{f"t_{k}": round(float(t[i, j]), 2) for j, k in enumerate(TAU_KEYS)},
            }
        )

    # rank: single-concept first, then by selectivity, then by strength
    rows.sort(key=lambda r: (r["n_sig_tau"], -r["selectivity"], -r["dom_t"]))

    # attach a few ground-truth columns for context
    gt_cols = [
        "top_dominant_profile",
        "top_mean_speed",
        "top_mean_speed_range",
        "top_mean_accel",
        "top_frac_accel",
        "top_mean_turn_deg",
        "top_frac_turning",
        "n_nonzero_videos",
    ]
    for r in rows:
        g = summ.get(r["feature"], {})
        for c in gt_cols:
            r[c] = g.get(c, "")

    fieldnames = (
        [
            "feature",
            "idx",
            "dom_tau",
            "dom_t",
            "dom_t_signed",
            "runnerup_tau",
            "runnerup_t",
            "selectivity",
            "n_sig_tau",
        ]
        + [f"t_{k}" for k in TAU_KEYS]
        + gt_cols
    )
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} ranked features -> {OUT}\n")

    # console: best single-concept candidate per dom_tau
    print("=== Cleanest single-concept candidates (n_sig_tau==1), per concept ===")
    singles = [r for r in rows if r["n_sig_tau"] == 1]
    for tau in TAU_KEYS:
        cands = [r for r in singles if r["dom_tau"] == tau]
        cands.sort(key=lambda r: -r["selectivity"])
        print(f"\n-- {tau} -- ({len(cands)} single-concept features)")
        for r in cands[:3]:
            print(
                f"  {r['feature']}  dom_t={r['dom_t_signed']:+6.2f}  sel={r['selectivity']:5.1f}x  "
                f"|t|=[" + " ".join(f"{r['t_'+k]:+5.1f}" for k in TAU_KEYS) + "]  "
                f"profile={r['top_dominant_profile']:<14} "
                f"frac_accel={r['top_frac_accel']} frac_turn={r['top_frac_turning']} "
                f"spd={r['top_mean_speed']} spdrange={r['top_mean_speed_range']}"
            )


if __name__ == "__main__":
    main()
