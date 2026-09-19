"""ProjFrac Sweep - positive control that maps NFP detection against ground-truth temporal content.

Constructs unit test directions d = beta * u_tau + sqrt(1 - beta^2) * u_perp
with exact ground-truth temporal content ProjFrac(d) = beta^2, computes the
scalar readout psi(V, t) = <h(V, t), d> on the existing synthetic
representation, and runs the standard NFP test (within-video covariance with
the five z-scored concept profiles, one-sample t over videos, Bonferroni at
alpha / M for the sweep's own dictionary size M).

Grid: log-spaced beta^2 (the detection transition sits at very small beta^2
because a pure temporal direction has near-deterministic covariance). At each
level, R random u_tau draws plus the five axis-aligned directions (all
temporal content in one concept), each with independent u_perp. PosFrac,
the fraction of each direction's norm in the position-dependent subspace,
is tracked as a diagnostic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def main() -> None:
    """Run the ProjFrac sweep, run the NFP test at each level, and write JSON plus figures.

    Loads the synthetic representation and its ground-truth direction blocks, builds a grid
    of test directions with known ProjFrac = beta^2 (plus null, position-null, and
    static-null contrasts), computes the scalar readout on every video/step, runs the
    within-video covariance one-sample t-test with Bonferroni correction, and reports which
    directions are flagged. Results are saved as ``<out>.json`` and a scatter of max|t| vs
    ProjFrac colored by PosFrac as ``<out>.pdf`` / ``<out>.png``.

    Returns:
        None. Writes the JSON summary and figures to ``args.out``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_videos", default="local_runs/synth_data_pv/all_videos.pt")
    ap.add_argument("--matrices", default="local_runs/synth_data_pv/matrices.pt")
    ap.add_argument("--levels", default=26, type=int, help="log-spaced beta^2 levels in [1e-5, 1]")
    ap.add_argument("--reps", default=20, type=int, help="random u_tau draws per level")
    ap.add_argument("--n_null", default=100, type=int, help="directions at beta^2 = 0")
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default="local_runs/expansion/projfrac_sweep")
    args = ap.parse_args()

    d = torch.load(args.all_videos, map_location="cpu", weights_only=False)
    m = torch.load(args.matrices, map_location="cpu", weights_only=False)
    h = d["h"].float()  # [N, 8, 768]
    tau = d["tau_norm"].float()  # [N, 8, 5]
    W_tau = m["W_tau"].float()  # [5, 768] orthonormal rows
    W_pos = m["W_pos"].float()  # [50, 768] orthonormal rows
    N, T, D = h.shape
    rng = np.random.RandomState(args.seed)

    def unit(v: np.ndarray) -> np.ndarray:
        """Return v scaled to unit L2 norm (with a small epsilon for stability)."""
        return v / (np.linalg.norm(v) + 1e-12)

    Wt = W_tau.numpy()

    def rand_u_tau() -> np.ndarray:
        """Sample a random unit direction lying entirely in the temporal subspace."""
        return unit(rng.randn(5)) @ Wt

    def rand_u_perp() -> np.ndarray:
        """Sample a random unit direction orthogonal to the temporal subspace."""
        v = rng.randn(D)
        v = v - (v @ Wt.T) @ Wt
        return unit(v)

    Wp = W_pos.numpy()
    Ws = m["W_static"].float().numpy()

    beta2_levels = [0.0] + list(np.logspace(-5, 0, args.levels))
    dirs, meta = [], []
    for b2 in beta2_levels:
        beta = float(np.sqrt(b2))
        if b2 == 0.0:
            for _ in range(args.n_null):
                dirs.append(rand_u_perp())
                meta.append({"beta2": 0.0, "kind": "null", "concept": ""})
            # hardest nulls: all norm inside the position-dependent subspace
            for _ in range(args.n_null // 2):
                dirs.append(unit(rng.randn(50)) @ Wp)
                meta.append({"beta2": 0.0, "kind": "null_pos", "concept": ""})
            # contrast nulls: all norm inside the constant-static subspace
            for _ in range(args.n_null // 2):
                dirs.append(unit(rng.randn(100)) @ Ws)
                meta.append({"beta2": 0.0, "kind": "null_static", "concept": ""})
            continue
        for _ in range(args.reps):
            dv = beta * rand_u_tau() + np.sqrt(1 - b2) * rand_u_perp()
            dirs.append(unit(dv))
            meta.append({"beta2": b2, "kind": "random", "concept": ""})
        for k in range(5):
            dv = beta * Wt[k] + np.sqrt(1 - b2) * rand_u_perp()
            dirs.append(unit(dv))
            meta.append({"beta2": b2, "kind": "axis", "concept": TAU_KEYS[k]})
    Dmat = torch.tensor(np.stack(dirs), dtype=torch.float32)  # [M, 768]
    M = Dmat.shape[0]
    print(f"directions: {M} ({len(beta2_levels)} levels)")

    posfrac = (Dmat @ W_pos.T).pow(2).sum(1).numpy()  # rows orthonormal

    psi = torch.einsum("ntd,md->ntm", h, Dmat)  # [N, 8, M]
    psi_c = psi - psi.mean(1, keepdim=True)
    tau_c = tau - tau.mean(1, keepdim=True)
    C = torch.einsum("ntm,ntk->nmk", psi_c, tau_c) / T  # [N, M, 5]
    Cm = C.mean(0)  # [M, 5]
    Cs = C.std(0, unbiased=True)
    t = (Cm / (Cs / np.sqrt(N))).numpy()  # [M, 5]

    thr = stats.t.ppf(1 - args.alpha / (2 * M * 5), N - 1)
    thr_simple = stats.t.ppf(1 - args.alpha / (2 * M), N - 1)
    maxabs = np.abs(t).max(1)
    flagged = maxabs > thr_simple
    print(f"Bonferroni |t| threshold (alpha/M): {thr_simple:.2f}  (alpha/(M K): {thr:.2f})")

    rows = []
    for i, mt in enumerate(meta):
        rows.append(
            {
                **mt,
                "posfrac": round(float(posfrac[i]), 4),
                "max_abs_t": round(float(maxabs[i]), 3),
                "t": [round(float(x), 3) for x in t[i]],
                "flagged": bool(flagged[i]),
            }
        )
    out = Path(args.out)
    json.dump(
        {
            "M": M,
            "threshold": float(thr_simple),
            "alpha": args.alpha,
            "tau_keys": TAU_KEYS,
            "rows": rows,
        },
        open(str(out) + ".json", "w"),
        indent=1,
    )

    # summary to stdout
    for kind in ["null", "null_pos", "null_static"]:
        nul = [r for r in rows if r["kind"] == kind]
        print(
            f"beta2=0 [{kind}]: n={len(nul)} flagged={sum(r['flagged'] for r in nul)} "
            f"max|t|={max(r['max_abs_t'] for r in nul):.2f} "
            f"(max posfrac {max(r['posfrac'] for r in nul):.2f})"
        )
    for b2 in beta2_levels[1:]:
        lv = [r for r in rows if abs(r["beta2"] - b2) < 1e-12 and r["kind"] == "random"]
        ax = [r for r in rows if abs(r["beta2"] - b2) < 1e-12 and r["kind"] == "axis"]
        fr = sum(r["flagged"] for r in lv)
        print(
            f"beta2={b2:.1e}: random {fr}/{len(lv)} flagged, "
            f"median max|t|={np.median([r['max_abs_t'] for r in lv]):.1f}, "
            f"axis max|t| {[round(r['max_abs_t'], 1) for r in ax]}"
        )

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axp = plt.subplots(figsize=(6.4, 4.2))
    x0 = 2e-6  # pseudo-position for beta2 = 0 on the log axis
    rnd = [r for r in rows if r["kind"] in ("random", "null", "null_pos", "null_static")]
    xs = [max(r["beta2"], x0) for r in rnd]
    ys = [max(r["max_abs_t"], 1e-2) for r in rnd]
    cs = [r["posfrac"] for r in rnd]
    sc = axp.scatter(xs, ys, c=cs, cmap="viridis", s=14, alpha=0.75, linewidths=0)
    for k, key in enumerate(TAU_KEYS):
        ax_rows = sorted(
            [r for r in rows if r["kind"] == "axis" and r["concept"] == key],
            key=lambda r: r["beta2"],
        )
        axp.plot(
            [r["beta2"] for r in ax_rows], [r["max_abs_t"] for r in ax_rows], lw=1.2, label=key
        )
    axp.axhline(thr_simple, color="k", ls="--", lw=1)
    axp.text(1.2e-5, thr_simple * 1.15, f"Bonferroni threshold ({thr_simple:.1f})", fontsize=8)
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.set_xlabel(r"ProjFrac $\beta^2$ (0 shown at left edge)")
    axp.set_ylabel(r"$\max_k |t_k|$")
    axp.legend(fontsize=7, title="axis-aligned", loc="lower right")
    cb = fig.colorbar(sc, ax=axp)
    cb.set_label("PosFrac")
    fig.tight_layout()
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=150)
    print(f"saved -> {out}.json / .pdf / .png")


if __name__ == "__main__":
    main()
