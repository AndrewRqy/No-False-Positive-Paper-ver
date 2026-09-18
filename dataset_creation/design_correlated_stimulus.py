"""
Experiment F3 — NFP stimulus with ONE temporal pair held at a target correlation.

v2 (analysis/design_decorrelated_stimulus.py) zeroes every within-video coupling
so a recovered c_bar isolates a single concept direction. v3 asks the opposite
question: if two temporal concepts are strongly correlated within each video,
does the NFP test still separate their features, or does it conflate them? We
build a set in which one target pair carries within-video correlation ~=0.9
while every other pair stays ~0.

Two variants (both target correlation on ACCELERATION MAGNITUDE):
  S : corr(accel_mag, speed) = 0.9,  all other pairs ~ 0.
  X : corr(accel_mag, vel_x) = 0.9,  vel_y / direction pairs ~ 0.

Mechanism. accel_mag is a magnitude, so any stimulus that makes accel_mag track
a concept must speed the ball up and slow it down. A geometric speed profile
s(t) = s0 * r^t has s(t+1)-s(t) = (r-1) s(t), so within every such video
accel_mag(t) = |r-1| s(t) is exactly proportional to speed: within-video
corr(accel_mag, speed) = 1 and corr(accel_mag, vel_x) = sign(cos theta). We add
this geometric family to the candidate pool (data/nfp_ball_dataset.py,
GEOMETRIC_TYPES) and let the same reweighting LP as v2 mix them with the
decorrelated families to hit the target pair while zeroing the rest.

Geometric constraint (reported, not worked around). Because accel_mag couples to
SPEED, not to a signed axis, variant X cannot raise corr(accel_mag, vel_x)
without also raising corr(accel_mag, speed) and corr(speed, vel_x): the rightward
geometric videos that create the vel_x coupling create the speed coupling too.
Variant X therefore leaves those two pairs free (they come out ~0.9 as well); it
zeroes only the pairs that are geometrically separable (vel_y and direction). The
LP equality set encodes exactly this.

The output per-video profile spec matches v2's format (family, profile_type,
speed_mps, direction_deg) and re-renders through profile_from_spec, so S1/S3 and
the position sampling are untouched. Geometric entries carry profile_type in
GEOMETRIC_TYPES, which _family_a_speeds now understands.

Usage (from sae-for-vlm/):
  python analysis/design_correlated_stimulus.py --N 3000
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_creation.nfp_ball_dataset import (sample_velocity_profile, compute_tau, T,
                                    GEOMETRIC_TYPES, _family_a_speeds)
from dataset_creation.design_decorrelated_stimulus import TAU, N_STEPS, profile_tau

# TAU index map: speed=0 vel_x=1 vel_y=2 accel_mag=3 direction=4
SPEED, VELX, VELY, ACCEL, DIR = 0, 1, 2, 3, 4
ALL_PAIRS = [(k, a) for a in range(5) for k in range(a)]  # 10 pairs, k<a


def make_geometric_profile(gtype, speed_mps, direction_deg):
    """A geometric-speed profile dict (family A) in the same shape the LP and
    profile_from_spec expect."""
    s = float(speed_mps)
    theta = math.radians(float(direction_deg))
    speeds = _family_a_speeds(gtype, s)
    return {"family": "A", "profile_type": gtype, "speed_mps": s,
            "direction_deg": float(direction_deg),
            "vx": speeds * math.cos(theta), "vy": speeds * math.sin(theta)}


def build_pool(M_base, n_geo_dirs, geo_dir_lo, geo_dir_hi, seed):
    """Decorrelated base families (existing sampler) + a geometric family spanning
    a grid of directions in [geo_dir_lo, geo_dir_hi] deg and both speed extremes."""
    rng = np.random.default_rng(seed)
    profs = [sample_velocity_profile(rng) for _ in range(M_base)]
    dirs = np.linspace(geo_dir_lo, geo_dir_hi, n_geo_dirs, endpoint=False)
    speeds = np.linspace(1.0, 3.5, 6)
    for gt in GEOMETRIC_TYPES:
        for d in dirs:
            for s in speeds:
                profs.append(make_geometric_profile(gt, s, d))
    return profs


def pair_cov_var(taus):
    """taus [M,8,5] -> C [M,10] within-video covariances (one per ALL_PAIRS),
    V [M,5] within-video variances."""
    tc = taus - taus.mean(1, keepdims=True)
    C = np.stack([(tc[:, :, k] * tc[:, :, a]).mean(1) for k, a in ALL_PAIRS], 1)
    V = np.stack([(tc[:, :, a] ** 2).mean(1) for a in range(5)], 1)
    return C, V


def per_video_corr(taus, pair):
    """Per-video Pearson corr of `pair` and a mask of videos where both tau vary.
    This is the quantity validate() reports (mean within-video correlation)."""
    k, a = pair
    tc = taus - taus.mean(1, keepdims=True)
    sk, sa = tc[:, :, k].std(1), tc[:, :, a].std(1)
    varies = (sk > 1e-9) & (sa > 1e-9)
    denom = np.where(varies, sk * sa, 1.0)
    rho = np.where(varies, (tc[:, :, k] * tc[:, :, a]).mean(1) / denom, 0.0)
    return rho, varies


def wmean_corr(rho, varies, w):
    """w-weighted mean of per-video correlations over varying videos."""
    num = float((w * rho * varies).sum())
    den = float((w * varies).sum())
    return num / den if den > 1e-12 else 0.0


def weighted_corr(wC, wV, pair):
    k, a = pair
    d = math.sqrt(max(wV[k], 1e-12) * max(wV[a], 1e-12))
    return wC[ALL_PAIRS.index(pair)] / d if d > 0 else 0.0


def solve_target(C, V, primary, free_pairs, c_target, var_floor):
    """LP feasibility: primary-pair weighted cov = c_target, every pair not in
    free_pairs (and != primary) = 0, weighted var >= floor per tau, sum w = 1,
    w >= 0. Returns w or None."""
    M = C.shape[0]
    eq_pairs = [p for p in ALL_PAIRS if p != primary and p not in free_pairs]
    rows_eq = [C[:, ALL_PAIRS.index(p)] for p in eq_pairs]          # = 0
    rows_eq.append(C[:, ALL_PAIRS.index(primary)])                  # = c_target
    rows_eq.append(np.ones(M))                                      # = 1
    b_eq = [0.0] * len(eq_pairs) + [c_target, 1.0]
    A_ub = -V.T                                                     # -Var . w <= -floor
    b_ub = -var_floor
    for cap_mult in [3, 10, 50, None]:
        bounds = [(0, cap_mult / M if cap_mult else None)] * M
        r = linprog(np.zeros(M), A_ub=A_ub, b_ub=b_ub,
                    A_eq=np.vstack(rows_eq), b_eq=np.array(b_eq),
                    bounds=bounds, method="highs")
        if r.status == 0:
            return r.x, f"cap={cap_mult}"
    return None, "infeasible"


def scan_correlation(C, V, taus, primary, free_pairs, var_floor, rho_target):
    """Scan the primary-pair covariance target so the achieved MEAN WITHIN-VIDEO
    (per-video Pearson) correlation lands on rho_target. Selecting on the same
    quantity validate() reports keeps design and re-rendered result aligned.
    Returns (w, c_target, how, rho_pv)."""
    k, a = primary
    scale = math.sqrt(V[:, k].mean() * V[:, a].mean())
    rho_pv, varies = per_video_corr(taus, primary)
    best = None
    for frac in np.linspace(0.2, 3.0, 43):
        c_target = frac * rho_target * scale
        w, how = solve_target(C, V, primary, free_pairs, c_target, var_floor)
        if w is None:
            continue
        rho = wmean_corr(rho_pv, varies, w)
        cand = (abs(rho - rho_target), w, c_target, how, rho)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        return None, None, "infeasible at all targets", None
    _, w, c_target, how, rho = best
    return w, c_target, how, rho


def emit_spec(profs, w, N, seed, out_path, meta):
    """largest-remainder allocation to N videos; v2-compatible per-video spec."""
    raw = N * w
    counts = np.floor(raw).astype(int)
    rem = N - counts.sum()
    counts[np.argsort(-(raw - counts))[:rem]] += 1
    assert counts.sum() == N
    videos = []
    for j in np.where(counts > 0)[0]:
        p = profs[j]
        e = {"family": p["family"], "profile_type": p["profile_type"],
             "speed_mps": p["speed_mps"], "direction_deg": p["direction_deg"]}
        if "delta_deg" in p:
            e["delta_deg"] = p["delta_deg"]
        videos += [e] * int(counts[j])
    np.random.default_rng(seed + 7).shuffle(videos)
    spec = {"n_videos": N, **meta, "videos": videos}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(spec, open(out_path, "w"), indent=1)
    fam = {}
    for v in videos:
        key = f"{v['profile_type']}"
        fam[key] = fam.get(key, 0) + 1
    geo = sum(c for k, c in fam.items() if k.startswith("geo_"))
    print(f"  alloc: geometric={geo}/{N} ({100*geo/N:.0f}%)  "
          + "  ".join(f"{k}={c}" for k, c in sorted(fam.items(), key=lambda kv: -kv[1])[:6]))
    print(f"  spec -> {out_path}")
    return spec


def validate(spec_path):
    """Re-render every video's tau from the saved spec and report the achieved
    mean within-video correlation matrix (independent of the design math)."""
    from dataset_creation.nfp_ball_dataset import profile_from_spec
    spec = json.load(open(spec_path))
    cors = {p: [] for p in ALL_PAIRS}
    for e in spec["videos"]:
        prof = profile_from_spec(e)
        taus = profile_tau(prof)
        tc = taus - taus.mean(0, keepdims=True)
        for p in ALL_PAIRS:
            k, a = p
            sk, sa = tc[:, k].std(), tc[:, a].std()
            if sk > 1e-9 and sa > 1e-9:
                cors[p].append(float((tc[:, k] * tc[:, a]).mean() / (sk * sa)))
    print(f"  validated {len(spec['videos'])} videos; mean within-video corr:")
    for p in ALL_PAIRS:
        vals = cors[p]
        m = np.mean(vals) if vals else float('nan')
        tag = "  <-- TARGET" if p == tuple(spec.get("primary_pair", ())) else ""
        print(f"    {TAU[p[0]]:>9s}-{TAU[p[1]]:<9s} {m:+.3f}  (n={len(vals)}){tag}")
    return cors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M_base", default=4000, type=int)
    ap.add_argument("--N", default=3000, type=int)
    ap.add_argument("--rho", default=0.9, type=float)
    ap.add_argument("--var_frac", default=0.25, type=float,
                    help="weighted Var(tau) floor as a fraction of pool mean")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--outdir", default="local_runs/steering")
    ap.add_argument("--variant", default="both", choices=["S", "X", "both"])
    args = ap.parse_args()
    outdir = Path(args.outdir)

    specs = {}
    variants = ["S", "X"] if args.variant == "both" else [args.variant]
    for v in variants:
        if v == "S":
            # accel-speed correlated; geometric directions uniform so vel_x/vel_y
            # decorrelate from accel and from each other.
            primary = (SPEED, ACCEL)
            free = []                       # zero all other 9 pairs
            profs = build_pool(args.M_base, 24, 0.0, 360.0, args.seed)
        else:
            # accel-vel_x correlated; geometric directions near-horizontal so the
            # coupling has a sign. speed-accel and speed-velx are geometrically
            # forced -> left free; vel_y and direction pairs zeroed.
            primary = (VELX, ACCEL)
            free = [(SPEED, ACCEL), (SPEED, VELX)]
            profs = build_pool(args.M_base, 12, -35.0, 35.0, args.seed)

        taus = np.stack([profile_tau(p) for p in profs])
        C, V = pair_cov_var(taus)
        var_floor = args.var_frac * V.mean(0)
        # direction variance is tiny once the pool is geometric-heavy; relax its floor
        var_floor[DIR] *= 0.1
        print(f"\n=== variant {v}: target corr({TAU[primary[0]]},{TAU[primary[1]]}) "
              f"= {args.rho}; pool M={len(profs)} ===")
        w, c_target, how, rho = scan_correlation(C, V, taus, primary, free, var_floor, args.rho)
        if w is None:
            print(f"  {how}")
            continue
        ess = 1.0 / float((w ** 2).sum())
        wC, wV = C.T @ w, V.T @ w
        print(f"  solved ({how}); c_target={c_target:.4f}; achieved corr={rho:+.3f}; "
              f"ESS={ess:.0f}/{len(profs)}")
        print("  weighted within-video corr per pair:")
        for p in ALL_PAIRS:
            r = weighted_corr(wC, wV, p)
            flag = " *" if (p == primary or p in free) else ""
            print(f"    {TAU[p[0]]:>9s}-{TAU[p[1]]:<9s} {r:+.3f}{flag}")
        out_path = outdir / f"nfp_v3_{v}_profile_spec.json"
        meta = {"variant": v, "primary_pair": list(primary),
                "target_corr": args.rho, "achieved_corr_design": round(rho, 4),
                "free_pairs": [list(p) for p in free], "ess": round(ess, 1),
                "how": how}
        emit_spec(profs, w, args.N, args.seed, out_path, meta)
        print("  --- re-rendered validation ---")
        validate(out_path)
        specs[v] = str(out_path)

    print(f"\nDesign complete. specs: {specs}")


if __name__ == "__main__":
    main()
