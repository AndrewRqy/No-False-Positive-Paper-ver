"""
Variance-Matched Stimulus Design - equalize per-concept within-video variance.

The decorrelated (v2) and correlated (v3) designs control the within-video
*couplings* between temporal concepts. This design instead controls their
within-video *variances*. The default NFP stimulus gives acceleration a ~20x
smaller within-video variance than speed (acceleration is a difference of adjacent
velocities, which is small and steady for smooth motion), so the NFP test is far
less sensitive to it. This design builds a decorrelated set in which the five
concepts have as-equal-as-possible within-video variance, putting every concept on
an equal detectability footing.

Mechanism: raising acceleration's within-video variance requires a speed that
genuinely swings within a clip, so an oscillating-speed family
(``OSCILLATING_TYPES`` in ``nfp_ball_dataset``) is added to the candidate pool. A
linear program then reweights the pooled families to maximize the smallest
per-concept within-video variance (a max-min objective) while zeroing all ten
pairwise couplings, so the set stays decorrelated. The output per-video profile
spec matches the v2/v3 format (family, profile_type, speed_mps, direction_deg) and
re-renders through ``profile_from_spec``, so the position sampling is untouched and
the NFP proof holds exactly as in v1/v2/v3.

Usage:
  python -m dataset_creation.design_variance_matched --N 3000
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_creation.nfp_ball_dataset import (
    compute_tau, T, GEOMETRIC_TYPES, OSCILLATING_TYPES,
    FAMILY_A_TYPES, FAMILY_B_TYPES, _family_a_speeds, _family_b_directions,
    V_MIN_MPS, V_MAX_MPS, profile_from_spec,
)
from dataset_creation.design_decorrelated_stimulus import TAU, profile_tau

ALL_PAIRS = [(k, a) for a in range(5) for k in range(a)]  # 10 pairs


def straight_profile(ptype: str, s: float, direction_deg: float) -> Dict[str, Any]:
    theta = math.radians(direction_deg)
    speeds = _family_a_speeds(ptype, s)
    return {"family": "A", "profile_type": ptype, "speed_mps": float(s),
            "direction_deg": float(direction_deg),
            "vx": speeds * math.cos(theta), "vy": speeds * math.sin(theta)}


def turn_profile(ptype: str, s: float, direction_deg: float, delta_deg: float) -> Dict[str, Any]:
    theta = math.radians(direction_deg)
    delta = math.pi if ptype == "back_and_forth" else math.radians(delta_deg)
    dirs = _family_b_directions(ptype, theta, delta)
    return {"family": "B", "profile_type": ptype, "speed_mps": float(s),
            "direction_deg": float(direction_deg), "delta_deg": float(delta_deg),
            "vx": s * np.cos(dirs), "vy": s * np.sin(dirs)}


def build_pool(seed: int) -> List[Dict[str, Any]]:
    """Straight families (including the oscillating family for acceleration
    variance) at uniform headings, plus turning families for direction variance."""
    rng = np.random.default_rng(seed)
    profs: List[Dict[str, Any]] = []
    speeds = np.linspace(1.0, 3.3, 6)
    dirs = np.linspace(0, 360, 24, endpoint=False)
    straight_types = list(FAMILY_A_TYPES) + list(GEOMETRIC_TYPES) + list(OSCILLATING_TYPES)
    for pt in straight_types:
        for s in speeds:
            for d in dirs:
                profs.append(straight_profile(pt, s, d))
    for pt in FAMILY_B_TYPES:
        for s in speeds:
            for d in dirs:
                for delta in (45.0, 90.0, 150.0):
                    profs.append(turn_profile(pt, s, d, delta))
    return profs


def moments(taus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """taus [M,8,5] -> C [M,10] within-video covariances, V [M,5] within-video variances."""
    tc = taus - taus.mean(1, keepdims=True)
    C = np.stack([(tc[:, :, k] * tc[:, :, a]).mean(1) for k, a in ALL_PAIRS], 1)
    V = np.stack([(tc[:, :, a] ** 2).mean(1) for a in range(5)], 1)
    return C, V


def solve_maxmin(C: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """Maximize the smallest per-concept weighted variance subject to all couplings
    zero, sum(w)=1, w>=0. Variables are [w (M), Vt (1)]."""
    M = C.shape[0]
    c = np.zeros(M + 1); c[-1] = -1.0                       # maximize Vt
    # -V_k . w + Vt <= 0   for each concept k
    A_ub = np.hstack([-V.T, np.ones((5, 1))]); b_ub = np.zeros(5)
    # couplings = 0, sum w = 1  (Vt column is zero)
    A_eq = np.vstack([np.hstack([C.T, np.zeros((10, 1))]),
                      np.hstack([np.ones((1, M)), np.zeros((1, 1))])])
    b_eq = np.concatenate([np.zeros(10), [1.0]])
    for cap in [5, 20, 100, None]:
        bounds = [(0, cap / M if cap else None)] * M + [(0, None)]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if r.status == 0:
            return r.x[:M], float(r.x[M]), f"cap={cap}"
    return None, 0.0, "infeasible"


def emit(profs: List[Dict[str, Any]], w: np.ndarray, N: int, seed: int,
         out_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    raw = N * w
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts))[: N - counts.sum()]] += 1
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
    fam: Dict[str, int] = {}
    for v in videos:
        fam[v["profile_type"]] = fam.get(v["profile_type"], 0) + 1
    top = sorted(fam.items(), key=lambda kv: -kv[1])[:8]
    print("  alloc:", "  ".join(f"{k}={c}" for k, c in top))
    print(f"  spec -> {out_path}")
    return spec


def validate(spec_path: str) -> None:
    spec = json.load(open(spec_path))
    Vs = {a: [] for a in range(5)}
    cors = {p: [] for p in ALL_PAIRS}
    for e in spec["videos"]:
        taus = profile_tau(profile_from_spec(e))
        tc = taus - taus.mean(0, keepdims=True)
        for a in range(5):
            Vs[a].append((tc[:, a] ** 2).mean())
        for k, a in ALL_PAIRS:
            sk, sa = tc[:, k].std(), tc[:, a].std()
            if sk > 1e-9 and sa > 1e-9:
                cors[(k, a)].append(float((tc[:, k] * tc[:, a]).mean() / (sk * sa)))
    print("  re-rendered within-video variance per concept:")
    for a in range(5):
        print(f"    {TAU[a]:>10s}: {np.mean(Vs[a]):.4f}")
    worst = max(abs(np.mean(cors[p])) if cors[p] else 0 for p in ALL_PAIRS)
    print(f"  max |mean within-video correlation| over all pairs: {worst:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default=3000, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default="dataset_creation/specs/nfp_varmatched_profile_spec.json")
    args = ap.parse_args()
    profs = build_pool(args.seed)
    taus = np.stack([profile_tau(p) for p in profs])
    C, V = moments(taus)
    print(f"pool M={len(profs)}; per-concept max achievable within-video variance: "
          + "  ".join(f"{TAU[a]}={V[:, a].max():.3f}" for a in range(5)))
    w, Vt, how = solve_maxmin(C, V)
    if w is None:
        print("INFEASIBLE")
        return
    wv = V.T @ w
    ess = 1.0 / float((w ** 2).sum())
    print(f"solved ({how}); max-min variance Vt={Vt:.4f}; ESS={ess:.0f}/{len(profs)}")
    print("  weighted within-video variance per concept:")
    for a in range(5):
        print(f"    {TAU[a]:>10s}: {wv[a]:.4f}")
    meta = {"design": "variance_matched", "maxmin_variance": round(Vt, 4),
            "ess": round(ess, 1), "how": how}
    emit(profs, w, args.N, args.seed, args.out, meta)
    print("  --- re-rendered validation ---")
    validate(args.out)


if __name__ == "__main__":
    main()
