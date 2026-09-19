"""Decorrelated Stimulus Design - build the v2 NFP ball-video set with zeroed tau couplings.

Experiment F2: structured decorrelation of the NFP stimulus set (new theory section 4).

Goal: a v2 ball-video design in which, for a target concept tau_a, every confound
concept tau_k has exactly zero weighted-average within-video covariance with tau_a:

  sum_j w_j Cov_t(tau_k^(pi_j), tau_a^(pi_j)) = 0   for all k != a,
  w_j >= 0, sum_j w_j = 1, sum_j w_j Var_t(tau_a^(pi_j)) > 0.

Under an additive encoding h = sum_k tau_k d_k + static, this makes the recovered
c_bar_{tau_a} proportional to the true direction d_a alone (section 4.1). The theory
also proves a hard limit (Chebyshev's sum inequality): any confound that is a monotone
function of the target has single-signed Cov_t across ALL profiles and cannot be
decorrelated by any stimulus design.

Procedure (section 4.3):
  1. Sample a large pool of M candidate velocity profiles from the existing NFP
     generator (dataset_creation/nfp_ball_dataset.py, same families and parameter ranges).
  2. Compute tau at the 8 tubelet steps (frame 2*step, "first_frame" mode, matching
     the NFP pipeline) and the covariance matrix C[j, k] = Cov_t(tau_k, tau_a) per
     profile, plus Var_t(tau_a).
  3. Empirical Chebyshev check: for each (target, confound), do the per-profile
     covariances carry both signs? Single-signed columns are certified
     non-decorrelatable and are dropped from the constraint set (reported).
  4. Solve the QP: minimize sum w^2 subject to C^T w = 0, sum w = 1, w >= 0,
     weighted Var >= var_frac * mean Var. cvxpy if available, else a projected
     alternating scheme on top of scipy linprog feasibility.
  5. Verify achieved covariances, report effective sample size 1/sum(w^2) and the
     weight distribution over profile types; save weights + profile parameters for
     dataset regeneration (round(N*w_j) videos of profile pi_j, positions sampled
     uniformly as before, S1/S3 unchanged, so the NFP guarantee is preserved).

Usage (from repo root):
  python -m dataset_creation.design_decorrelated_stimulus --M 4000 --N 3000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_creation.nfp_ball_dataset import sample_velocity_profile, compute_tau, T

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_STEPS = 8


def profile_tau(prof: Dict[str, Any]) -> np.ndarray:
    """Evaluate tau at the 8 tubelet steps for one velocity profile.

    Args:
        prof: Velocity profile dict with per-frame "vx" and "vy" arrays.

    Returns:
        A [8, 5] array of the five tau concepts at each tubelet step.
    """
    rows = []
    for step in range(N_STEPS):
        td = compute_tau(step * 2, prof["vx"], prof["vy"])
        rows.append([td[k] for k in TAU])
    return np.array(rows)


def covs_for_target(taus: np.ndarray, a: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Per-profile covariances of every confound with the target concept.

    Args:
        taus: [M, 8, 5] array of tau values for M profiles at the 8 tubelet steps.
        a: Index of the target concept (column of the last axis).

    Returns:
        A tuple (C, var, ks) where C is [M, 4] within-video covariances of each
        confound with the target, var is [M] within-video variances of the target,
        and ks is the list of confound concept indices (all indices except a).
    """
    tc = taus - taus.mean(1, keepdims=True)
    target = tc[:, :, a]
    var = (target**2).mean(1)
    ks = [k for k in range(5) if k != a]
    C = np.stack([(tc[:, :, k] * target).mean(1) for k in ks], 1)
    return C, var, ks


def solve_qp(C: np.ndarray, var: np.ndarray, var_min: float) -> Tuple[Optional[np.ndarray], str]:
    """Solve the decorrelation QP for one target concept.

    Minimizes sum w^2 subject to C^T w = 0, sum w = 1, w >= 0, and var . w >= var_min.
    Uses cvxpy when it is importable, otherwise falls back to a scipy linprog
    feasibility search with a progressively loosened spread cap.

    Args:
        C: [M, K] per-profile covariances of the K kept confounds with the target.
        var: [M] per-profile within-video variances of the target concept.
        var_min: Lower bound the weighted target variance var . w must reach.

    Returns:
        A tuple (w, how) where w is the [M] weight vector (or None if infeasible)
        and how is a short string naming the solver / status.
    """
    M = C.shape[0]
    try:
        import cvxpy as cp

        w = cp.Variable(M, nonneg=True)
        cons = [C.T @ w == 0, cp.sum(w) == 1, var @ w >= var_min]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(w)), cons)
        prob.solve()
        if w.value is None:
            return None, "infeasible (cvxpy)"
        return np.maximum(w.value, 0), "cvxpy"
    except ImportError:
        pass
    # fallback: linprog feasibility with a spread cap, loosened until feasible
    from scipy.optimize import linprog

    for cap_mult in [3, 10, 50, 200, None]:
        bounds = [(0, cap_mult / M if cap_mult else None)] * M
        A_eq = np.vstack([C.T, np.ones(M)])
        b_eq = np.zeros(A_eq.shape[0])
        b_eq[-1] = 1.0
        A_ub = -var[None, :]
        b_ub = np.array([-var_min])
        r = linprog(
            np.zeros(M), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )
        if r.status == 0:
            return r.x, f"linprog(cap={cap_mult})"
    return None, "infeasible (linprog all caps)"


def solve_joint(
    taus: np.ndarray,
    profs: List[Dict[str, Any]],
    N: int,
    var_frac: float,
    seed: int,
    out_path: str,
) -> Optional[Dict[str, Any]]:
    """Solve one dataset with ALL 10 pairwise couplings zeroed simultaneously.

    Constraints: sum_j w_j Cov_t(tau_k, tau_a)^(j) = 0 for every pair k < a;
    sum w = 1; w >= 0; weighted Var(tau_a) >= var_frac * pool mean for all a.
    Emits an exactly-N per-video profile list (largest-remainder rounding,
    deterministic shuffle) for nfp_ball_dataset.py --profile_spec.

    Args:
        taus: [M, 8, 5] array of tau values for M candidate profiles.
        profs: The M candidate profile dicts aligned with taus.
        N: Number of videos in the regenerated dataset.
        var_frac: Fraction of the pool-mean variance each target must retain.
        seed: RNG seed; the deterministic shuffle uses seed + 1.
        out_path: Path to write the per-video profile spec JSON.

    Returns:
        The written spec dict, or None if the joint design is infeasible at all caps.
    """
    M = taus.shape[0]
    tc = taus - taus.mean(1, keepdims=True)
    pairs = [(k, a) for a in range(5) for k in range(a)]
    C = np.stack([(tc[:, :, k] * tc[:, :, a]).mean(1) for k, a in pairs], 1)  # [M,10]
    V = np.stack([(tc[:, :, a] ** 2).mean(1) for a in range(5)], 1)  # [M,5]
    var_min = var_frac * V.mean(0)
    print(f"\n=== JOINT design: {len(pairs)} pairwise constraints, {M} profiles ===")
    from scipy.optimize import linprog

    w = None
    for cap_mult in [3, 10, 50, None]:
        bounds = [(0, cap_mult / M if cap_mult else None)] * M
        A_eq = np.vstack([C.T, np.ones(M)])
        b_eq = np.zeros(A_eq.shape[0])
        b_eq[-1] = 1.0
        r = linprog(
            np.zeros(M),
            A_ub=-V.T,
            b_ub=-var_min,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if r.status == 0:
            w = r.x
            how = f"linprog(cap={cap_mult})"
            break
    if w is None:
        print("  JOINT DESIGN INFEASIBLE at all caps")
        return None
    resid = np.abs(C.T @ w)
    wvar = V.T @ w
    ess = 1.0 / float((w**2).sum())
    print(f"  solved ({how}); ESS={ess:.0f}/{M}")
    print(
        "  residual |Cov| per pair: "
        + "  ".join(f"{TAU[k]}-{TAU[a]}={resid[i]:.1e}" for i, (k, a) in enumerate(pairs))
    )
    print(
        "  weighted Var per tau: "
        + "  ".join(f"{TAU[a]}={wvar[a]:.4f} (floor {var_min[a]:.4f})" for a in range(5))
    )

    # exact-N allocation: largest remainder
    raw = N * w
    counts = np.floor(raw).astype(int)
    rem = N - counts.sum()
    order = np.argsort(-(raw - counts))
    counts[order[:rem]] += 1
    assert counts.sum() == N
    videos = []
    for j in np.where(counts > 0)[0]:
        p = profs[j]
        entry = {
            "family": p["family"],
            "profile_type": p["profile_type"],
            "speed_mps": p["speed_mps"],
            "direction_deg": p["direction_deg"],
        }
        if "delta_deg" in p:
            entry["delta_deg"] = p["delta_deg"]
        videos += [entry] * int(counts[j])
    sh = np.random.default_rng(seed + 1)
    sh.shuffle(videos)
    spec = {
        "n_videos": N,
        "how": how,
        "ess": round(ess, 1),
        "residual_max": float(resid.max()),
        "weighted_var": {TAU[a]: float(wvar[a]) for a in range(5)},
        "videos": videos,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(spec, open(out_path, "w"), indent=2)
    fam = {}
    for v in videos:
        key = f"{v['family']}:{v['profile_type']}"
        fam[key] = fam.get(key, 0) + 1
    print(
        "  video allocation: "
        + "  ".join(f"{k}={c}" for k, c in sorted(fam.items(), key=lambda kv: -kv[1]))
    )
    print(f"  per-video spec ({N} videos) -> {out_path}")
    return spec


def main() -> None:
    """Parse CLI arguments, run the per-target and optional joint designs, save JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", default=4000, type=int)
    ap.add_argument("--N", default=3000, type=int, help="videos in the regenerated set")
    ap.add_argument(
        "--var_frac",
        default=0.5,
        type=float,
        help="weighted Var(target) must be >= this fraction of the pool mean",
    )
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument(
        "--joint",
        action="store_true",
        help="also solve the joint all-pairs design and emit the per-video "
        "profile spec for nfp_ball_dataset.py --profile_spec",
    )
    ap.add_argument("--joint_out", default="local_runs/steering/nfp_v2_profile_spec.json")
    ap.add_argument("--out", default="local_runs/steering/expF2_decorr_design.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"sampling {args.M} velocity profiles from the NFP generator...")
    profs = [sample_velocity_profile(rng) for _ in range(args.M)]
    taus = np.stack([profile_tau(p) for p in profs])  # [M,8,5]

    out = {"M": args.M, "N": args.N, "targets": {}}
    for a, aname in enumerate(TAU):
        C, var, ks = covs_for_target(taus, a)
        knames = [TAU[k] for k in ks]
        print(f"\n=== target {aname} (decorrelate against {knames}) ===")
        # empirical Chebyshev check
        keep, dropped = [], []
        for i, kn in enumerate(knames):
            frac_pos = float((C[:, i] > 1e-12).mean())
            frac_neg = float((C[:, i] < -1e-12).mean())
            both = frac_pos > 0.005 and frac_neg > 0.005
            print(
                f"  Cov({kn}, {aname}): {100*frac_pos:.0f}% positive, "
                f"{100*frac_neg:.0f}% negative -> "
                f"{'decorrelatable' if both else 'SINGLE-SIGNED: provably impossible'}"
            )
            (keep if both else dropped).append((i, kn))
        var_min = args.var_frac * float(var.mean())
        Ck = C[:, [i for i, _ in keep]] if keep else np.zeros((args.M, 0))
        w, how = solve_qp(Ck, var, var_min)
        rec = {
            "confounds_kept": [kn for _, kn in keep],
            "confounds_impossible": [kn for _, kn in dropped],
        }
        if w is None:
            print(f"  -> {how}")
            rec["status"] = how
        else:
            resid = {kn: float(abs(C[:, i] @ w)) for i, kn in keep}
            resid_dropped = {kn: float(C[:, i] @ w) for i, kn in dropped}
            ess = 1.0 / float((w**2).sum())
            fam = {}
            for j, p in enumerate(profs):
                key = f"{p['family']}:{p['profile_type']}"
                fam[key] = fam.get(key, 0.0) + float(w[j])
            print(
                f"  -> solved ({how}); weighted Var({aname}) = {float(var @ w):.4f} "
                f"(pool mean {var.mean():.4f}); effective sample size = {ess:.0f}/{args.M}"
            )
            print(
                f"     residual |Cov| on kept confounds: "
                + "  ".join(f"{k}={v:.2e}" for k, v in resid.items())
            )
            if resid_dropped:
                print(
                    f"     UNAVOIDABLE Cov on impossible confounds: "
                    + "  ".join(f"{k}={v:+.4f}" for k, v in resid_dropped.items())
                )
            print(
                f"     weight by profile type: "
                + "  ".join(
                    f"{k}={100*v:.0f}%"
                    for k, v in sorted(fam.items(), key=lambda kv: -kv[1])
                    if v > 0.005
                )
            )
            nz = int((w > 1e-6 / args.M).sum())
            rec.update(
                {
                    "status": f"solved ({how})",
                    "ess": round(ess, 1),
                    "n_nonzero": nz,
                    "weighted_var": float(var @ w),
                    "residuals": {k: float(f"{v:.3e}") for k, v in resid.items()},
                    "unavoidable": resid_dropped,
                    "weights_by_type": {k: round(v, 4) for k, v in fam.items()},
                    "video_counts_toptypes": {},
                }
            )
            # regeneration spec: round(N*w) per profile, keep profiles with >=1 video
            counts = np.round(args.N * w).astype(int)
            rec["n_videos_allocated"] = int(counts.sum())
            keep_idx = [int(j) for j in np.where(counts > 0)[0]]
            rec["n_profiles_used"] = len(keep_idx)
            spec = [
                {
                    "count": int(counts[j]),
                    "family": profs[j]["family"],
                    "type": profs[j]["profile_type"],
                    "speed_mps": profs[j]["speed_mps"],
                    "direction_deg": profs[j]["direction_deg"],
                    **({"delta_deg": profs[j]["delta_deg"]} if "delta_deg" in profs[j] else {}),
                }
                for j in keep_idx
            ]
            rec["profile_spec"] = spec[:2000]
        out["targets"][aname] = rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")

    if args.joint:
        solve_joint(taus, profs, args.N, args.var_frac, args.seed, args.joint_out)


if __name__ == "__main__":
    main()
