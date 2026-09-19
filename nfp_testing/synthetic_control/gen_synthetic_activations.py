"""Synthetic Activation Generator - build oracle positive/negative-control video representations.

For each video V at temporal step t, THREE signal types plus noise:

    h(V,t) = tau_norm(V,t) @ W_tau        [TEMPORAL - 5 dirs, kinematics-coupled]
           + static_code(V) @ W_static     [CONSTANT STATIC - N_STATIC dirs]
           + g(pos(V,t))    @ W_pos        [POSITION-DEPENDENT STATIC - N_POS dirs]
           + noise

Where:
  tau_norm(V,t)   - Z-scored tau variables at step t (changes as ball moves)
  static_code(V)  - random vector drawn once per video (the video's "content identity")
  g(pos(V,t))     - fixed random smooth functions of the ball's TRUE position at step t
                    (random Fourier features of pos_m; "place-cell"-like channels)
  W_tau / W_static / W_pos - mutually orthonormal direction blocks in R^768
  noise           - small i.i.d. Gaussian

N_STATIC >> 5 so the SAE has much more static content to encode than temporal,
mimicking the real VideoMAE setting where scene/object content dominates and
motion signal is a small fraction.

Why position channels: in the real model, "content" feature activations are NOT
constant across frames - they change smoothly BECAUSE the scene changes as things
move. A control whose static directions have literally zero within-video variance
makes the temporal/static separation artificially easy. Position channels
reproduce the realistic confound: temporally smooth, autocorrelated fluctuation
driven by the same trajectory as the kinematics - yet encoding WHERE the ball is,
not HOW it moves.

Why the t-test should still ignore them (--pos_mode):
  pervideo (default): each video draws its OWN random function - a different scene
    per video. Within-video covariance with tau is real (position integrates
    velocity), but E_phi[g]=0 makes the expected covariance exactly zero per video
    over the function draw, so across videos there is no consistent sign and the
    t-test faces its proper null. (A UNIVERSAL function does leak: a fixed spatial
    gradient picks up a cos^2-theta displacement-velocity term identical across
    videos - measured max |t| ~ 20 at N=3000.)
  universal_resid: keep one shared function bank and force exact per-video zero by
    linearly residualizing each channel's 8-step series against [1, tau courses]
    (SVD projector). Exact guarantee, but a degenerate point-mass-at-zero null.
Either way we MEASURE the leakage (oracle t-test on the g time-courses) and print
it; the leakage table is part of the control's ground truth.

NFP should find the temporal features among the crowd, and correctly ignore both
the constant-static AND the position-dependent directions.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_TEMPORAL = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for synthetic-activation generation.

    Returns:
        The parsed arguments, including representation dimension, static/position block
        sizes, position-channel mode and basis, oblique-geometry knobs, noise scale, and
        the train/val split fraction.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--nfp_dir", required=True, help="NFP dataset root (v0000/, v0001/, ...)")
    p.add_argument("--output_dir", required=True, help="Where to save activations and matrices")
    p.add_argument("--dim", default=768, type=int, help="Representation dimension")
    p.add_argument(
        "--n_static",
        default=100,
        type=int,
        help="Number of constant static directions (>> 5 to mimic VideoMAE)",
    )
    p.add_argument(
        "--n_pos",
        default=50,
        type=int,
        help="Number of position-dependent static directions (0 = old "
        "constant-only behavior). 5 + n_static + n_pos must be <= dim.",
    )
    p.add_argument(
        "--pos_mode",
        default="pervideo",
        choices=["pervideo", "universal_resid"],
        help="pervideo: each video draws its own random position function "
        "(dataset-level zero covariance in expectation; per-video "
        "covariance real). universal_resid: one shared function bank "
        "+ per-video linear residualization (exact per-video zero).",
    )
    p.add_argument(
        "--pos_basis",
        default="gaussian",
        choices=["gaussian", "fourier"],
        help="gaussian: localized signed bumps c_j(p)=+/-exp(-|p-mu_j|^2/2l^2) "
        "(sparse, place-cell-like; random sign makes E[c]=0 pointwise). "
        "fourier: global cosines with random phase.",
    )
    p.add_argument(
        "--pos_lengthscale",
        default=1.0,
        type=float,
        help="RFF length scale of the position channels, in units of the "
        "position std (~field size). Smaller = spatially busier channels.",
    )
    p.add_argument(
        "--tau_cos",
        default=0.0,
        type=float,
        help="Pairwise cosine among the 5 temporal directions (0 = orthogonal, "
        "the default). Implemented by mixing each with one shared "
        "direction: W_tau_i = sqrt(1-c) q_i + sqrt(c) q_shared.",
    )
    p.add_argument(
        "--tau_static_cos",
        default=0.0,
        type=float,
        help="Squared-cosine overlap of each temporal direction with one "
        "constant-static direction (0 = disjoint subspaces, default). "
        "W_tau_i <- sqrt(1-x) W_tau_i + sqrt(x) W_static_i.",
    )
    p.add_argument("--noise_scale", default=0.05, type=float)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--val_frac", default=0.2, type=float)
    return p.parse_args()


def main() -> None:
    """Generate, diagnose, and save the synthetic control dataset.

    Loads tau kinematics and ball positions from the NFP metadata, z-scores tau globally,
    builds mutually orthonormal temporal / constant-static / position-static direction
    blocks (optionally made oblique), composes ``h = h_temporal + h_static + h_posdep +
    noise``, and reports signal diagnostics and oracle position-channel kinematic leakage.
    Saves the projection matrices, the full ``[N, 8, D]`` video structure, and chunked flat
    activations split into train/val for SAE training.

    Returns:
        None. Writes ``matrices.pt``, ``all_videos.pt``, and ``train``/``val`` chunk
        directories under ``args.output_dir``.
    """
    args = parse_args()
    torch.manual_seed(args.seed)

    nfp_dir = Path(args.nfp_dir)
    video_dirs = sorted(nfp_dir.glob("v*"))
    N = len(video_dirs)
    D = args.dim
    N_STATIC = args.n_static
    print(f"Found {N} videos | dim={D} | n_static={N_STATIC}")

    # --- Load all tau values [N, 8, 5] and positions [N, 8, 2] from metadata ---
    all_tau, all_pos = [], []
    for vdir in tqdm(video_dirs, desc="Loading metadata"):
        with open(vdir / "metadata.json") as f:
            meta = json.load(f)
        traj = meta["trajectory"]
        steps, positions = [], []
        for step in range(N_TEMPORAL):
            rec = traj[step * 2]
            steps.append([rec["tau"][k] for k in TAU_KEYS])
            positions.append(rec["pos_m"])
        all_tau.append(steps)
        all_pos.append(positions)

    tau = torch.tensor(all_tau, dtype=torch.float32)  # [N, 8, 5]
    pos = torch.tensor(all_pos, dtype=torch.float32)  # [N, 8, 2]

    # --- Global Z-score normalization (using all N*8 frames) ---
    tau_flat = tau.reshape(-1, 5)
    tau_mean_global = tau_flat.mean(0)  # [5]
    tau_std_global = tau_flat.std(0).clamp(min=1e-6)  # [5]
    tau_norm = (tau - tau_mean_global) / tau_std_global  # [N, 8, 5]

    N_POS = args.n_pos
    assert (
        5 + N_STATIC + N_POS <= D
    ), f"5 + n_static ({N_STATIC}) + n_pos ({N_POS}) must be <= dim ({D})"

    # --- Random orthonormal projection matrices ---
    # QR gives 5 + N_STATIC + N_POS (+1 shared) orthonormal columns in R^D
    assert 5 + N_STATIC + N_POS + 1 <= D
    Q, _ = torch.linalg.qr(torch.randn(D, 5 + N_STATIC + N_POS + 1))
    W_tau = Q[:, :5].T.clone()  # [5,        D] temporal subspace
    W_static = Q[:, 5 : 5 + N_STATIC].T  # [N_STATIC, D] constant statics
    W_pos = Q[:, 5 + N_STATIC : 5 + N_STATIC + N_POS].T  # [N_POS,    D] position statics
    q_shared = Q[:, 5 + N_STATIC + N_POS]  # [D] for oblique tau geometry

    # --- Optional non-orthogonal geometry (ablation of the orthogonality restriction) ---
    # tau_cos c: every pair of temporal directions has cosine exactly c (shared-component
    # construction). tau_static_cos x: each temporal direction additionally leans into one
    # constant-static direction with squared cosine x, so the temporal and static
    # subspaces are no longer disjoint. Unit norms are preserved in both cases.
    if args.tau_cos > 0:
        c = args.tau_cos
        W_tau = (1 - c) ** 0.5 * W_tau + c**0.5 * q_shared.unsqueeze(0)
    if args.tau_static_cos > 0:
        x = args.tau_static_cos
        W_tau = (1 - x) ** 0.5 * W_tau + x**0.5 * W_static[:5]
    if args.tau_cos > 0 or args.tau_static_cos > 0:
        G = (W_tau @ W_tau.T).numpy()
        print(
            "oblique geometry: tau Gram matrix (should have off-diagonal ~"
            f"{args.tau_cos * (1 - args.tau_static_cos):.2f}):"
        )
        for i, name in enumerate(TAU_KEYS):
            print(f"  {name:<10}" + "".join(f"{G[i, j]:+7.3f}" for j in range(5)))
        if args.tau_static_cos > 0:
            cs_ = (W_tau @ W_static[:5].T).diagonal().numpy()
            print("  cos(tau_i, static_i): " + str([round(float(v), 3) for v in cs_]))

    # --- Temporal component: tau_norm projected through W_tau ---
    # Shape [N, 8, D] - different at each step because tau changes
    h_temporal = tau_norm @ W_tau  # [N, 8, D]

    # --- Constant static component: one random code per video, frozen across steps ---
    static_codes = torch.randn(N, N_STATIC)  # [N, N_STATIC]
    h_static = (static_codes @ W_static).unsqueeze(1).expand(N, N_TEMPORAL, D)

    # --- Position-dependent static component ---
    # Random smooth channels of the ball's true position ("place cells"):
    #   g_j(p) = sqrt(2) * cos(omega_j . p_std + phi_j),  omega ~ N(0, I)/lengthscale
    # evaluated at pos(V,t), then globally z-scored per channel.
    #
    # Two modes for where the randomness lives:
    #
    # pervideo (default): EVERY VIDEO draws its own (omega, phi) per channel - each
    #   video is a different scene with its own content landscape. Within-video
    #   covariance with tau is genuinely nonzero (content covaries with motion inside
    #   a clip, as in reality), but for uniform random phase E_phi[g] = 0, hence
    #   E[Cov_t(g_j, tau_k)] = 0 EXACTLY per video over the function draw - no
    #   reliance on trajectory-ensemble symmetries. Across N videos the covariance
    #   has no consistent sign, so the NFP t-test faces its PROPER null: real
    #   per-video covariance scatter, zero dataset-level mean. Expect oracle t-stats
    #   to look like standard-normal draws (max ~3.5 over 250 tests), not ~0.
    #
    # universal_resid: one function bank shared by all videos ("same world every
    #   video") + per-video linear residualization against [1, tau courses] via an
    #   SVD projector. Without the residualization a fixed spatial gradient picks up
    #   a cos^2-theta displacement-velocity term that is IDENTICAL across videos
    #   (measured max |t| ~ 20 at N=3000); the projection forces Cov_t = 0 exactly
    #   per video, at the cost of a degenerate (point-mass-at-zero) null.
    if N_POS > 0:
        pos_mean = pos.reshape(-1, 2).mean(0)
        pos_std = pos.reshape(-1, 2).std(0).clamp(min=1e-6)
        pos_std_units = (pos - pos_mean) / pos_std  # [N, 8, 2]
        L = args.pos_lengthscale
        p_lo = pos_std_units.reshape(-1, 2).min(0).values
        p_hi = pos_std_units.reshape(-1, 2).max(0).values
        pervid = args.pos_mode == "pervideo"
        rff_omega = rff_phi = gauss_mu = gauss_sign = None

        def raw_channels() -> torch.Tensor:
            """Evaluate the raw position channels at every video/step.

            Draws the channel parameters according to the selected basis
            (``fourier`` random Fourier features or ``gaussian`` signed bumps) and mode
            (``pervideo`` per-video draws or a single shared universal bank), stashing them
            in the enclosing ``rff_omega`` / ``rff_phi`` / ``gauss_mu`` / ``gauss_sign``
            variables for later saving.

            Returns:
                Raw (pre-normalization) position-channel values, shape [N, 8, N_POS].
            """
            nonlocal rff_omega, rff_phi, gauss_mu, gauss_sign
            if args.pos_basis == "fourier":
                # random phase makes E_phi[g] = 0 pointwise (exact)
                if pervid:
                    rff_omega = torch.randn(N, 2, N_POS) / L
                    rff_phi = torch.rand(N, 1, N_POS) * 2 * torch.pi
                    z = torch.einsum("ntc,ncj->ntj", pos_std_units, rff_omega) + rff_phi
                else:
                    rff_omega = torch.randn(2, N_POS) / L
                    rff_phi = torch.rand(N_POS) * 2 * torch.pi
                    z = pos_std_units @ rff_omega + rff_phi
                return torch.sqrt(torch.tensor(2.0)) * torch.cos(z)
            # gaussian: localized signed bumps; the random sign makes E_sign[c] = 0
            # pointwise (exact), the analogue of the random phase - random centers
            # alone would NOT suffice over a finite canvas (boundary effects leave
            # E_mu[c(p)] position-dependent).
            if pervid:
                gauss_mu = p_lo + (p_hi - p_lo) * torch.rand(N, N_POS, 2)
                gauss_sign = torch.where(torch.rand(N, 1, N_POS) < 0.5, -1.0, 1.0)
                d2 = ((pos_std_units.unsqueeze(2) - gauss_mu.unsqueeze(1)) ** 2).sum(-1)
            else:
                gauss_mu = p_lo + (p_hi - p_lo) * torch.rand(N_POS, 2)
                gauss_sign = torch.where(torch.rand(N_POS) < 0.5, -1.0, 1.0)
                d2 = ((pos_std_units.unsqueeze(2) - gauss_mu.view(1, 1, N_POS, 2)) ** 2).sum(-1)
            return gauss_sign * torch.exp(-d2 / (2 * L * L))

        g = raw_channels()
        g_flat = g.reshape(-1, N_POS)
        g = (g - g_flat.mean(0)) / g_flat.std(0).clamp(min=1e-6)  # [N, 8, N_POS]

        def leak_t(gg: torch.Tensor) -> torch.Tensor:
            """Oracle |t| of within-video position-channel/tau covariance across videos.

            Args:
                gg: Position-channel time-courses, shape [N, 8, N_POS].

            Returns:
                Absolute t-statistics of the mean within-video covariance, shape
                [N_POS, 5].
            """
            gc = gg - gg.mean(dim=1, keepdim=True)
            tc_ = tau - tau.mean(dim=1, keepdim=True)
            C_ = torch.einsum("btj,btk->bjk", gc, tc_) / N_TEMPORAL
            m_ = C_.mean(0)
            se_ = C_.std(0) / (N**0.5)
            return (m_ / se_.clamp(min=1e-12)).abs()

        if not pervid:
            print(
                f"  universal bank ({args.pos_basis}) leakage BEFORE residualization: "
                f"max|t| = {leak_t(g).max():.1f}"
            )
            # per-video linear decorrelation (SVD projector; rank-safe against the
            # piecewise-constant tau of step/turn profiles that breaks plain lstsq)
            X = torch.cat(
                [torch.ones(N, N_TEMPORAL, 1, dtype=torch.float64), tau_norm.double()], dim=2
            )  # [N, 8, 6]
            U, S, _ = torch.linalg.svd(X, full_matrices=False)  # U [N,8,6]
            keep = (S > S[:, :1] * 1e-9).unsqueeze(1)  # [N,1,6] rank mask
            U = U * keep
            g64 = g.double()
            g_res = g64 - U @ (U.transpose(1, 2) @ g64)  # [N, 8, N_POS]
            gr_flat = g_res.reshape(-1, N_POS)
            g = ((g_res - gr_flat.mean(0)) / gr_flat.std(0).clamp(min=1e-9)).float()

        h_posdep = g @ W_pos  # [N, 8, D]

        # --- Oracle kinematic-leakage report ---
        # Within-video covariance of each position channel with each tau, t-tested
        # across videos - the exact statistic NFP uses. Position integrates velocity,
        # so per-video covariances are nonzero; the dataset's symmetries (independent
        # spawn, uniform direction, sign-paired profiles) should leave no consistent
        # sign. We report the worst channel rather than assuming zero.
        gc = g - g.mean(dim=1, keepdim=True)
        tc = tau - tau.mean(dim=1, keepdim=True)
        Cg = torch.einsum("btj,btk->bjk", gc, tc) / N_TEMPORAL  # [N, N_POS, 5]
        m = Cg.mean(0)
        se = Cg.std(0) / (N**0.5)
        t_leak = (m / se.clamp(min=1e-12)).abs()  # [N_POS, 5]
        bar = 4.0  # ~Bonferroni for N_POS*5 tests at alpha=0.05
        n_leaky = int((t_leak > bar).sum())
        print(f"\nPosition-channel kinematic leakage (oracle, {N_POS}x5, mode={args.pos_mode}):")
        if args.pos_mode == "pervideo":
            print(f"  expectation: per-video Cov REAL (content covaries with motion within a")
            print(
                f"  clip), dataset-level mean ~0, t-stats null-calibrated (max ~3.5, none >{bar})"
            )
        else:
            print(f"  expectation: per-video Cov ~float eps (exact linear residualization)")
        print(f"  max |mean Cov_t| = {m.abs().max():.3e}")
        print(f"  max per-video |Cov_t| = {Cg.abs().max():.3e}")
        print(
            f"  max |t| = {t_leak.max():.2f}   mean |t| = {t_leak.mean():.2f}   "
            f"channels x taus with |t| > {bar}: {n_leaky}"
        )
        for k, name in enumerate(TAU_KEYS):
            print(f"    {name:<10} max|t|={t_leak[:, k].max():.2f}")
    else:
        g, pos_mean, pos_std = None, None, None
        rff_omega = rff_phi = gauss_mu = gauss_sign = None
        h_posdep = torch.zeros(N, N_TEMPORAL, D)

    noise = torch.randn(N, N_TEMPORAL, D) * args.noise_scale
    h = h_temporal + h_static + h_posdep + noise  # [N, 8, D]

    print(f"\nSignal diagnostics:")
    print(f"  Temporal component std   : {h_temporal.std():.4f}  ({5} directions)")
    print(f"  Constant static std      : {h_static.std():.4f}  ({N_STATIC} directions)")
    print(f"  Position-dep static std  : {h_posdep.std():.4f}  ({N_POS} directions)")
    print(f"  Noise std                : {noise.std():.4f}")
    print(f"  Total h std              : {h.std():.4f}")
    print(
        f"  (static+pos)/temporal    : "
        f"{(h_static.std()**2 + h_posdep.std()**2)**0.5 / h_temporal.std():.2f}x"
    )

    # --- Save projection matrices + normalization for NFP test ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "W_tau": W_tau,
            "W_static": W_static,
            "W_pos": W_pos if N_POS > 0 else None,
            "tau_mean_global": tau_mean_global,
            "tau_std_global": tau_std_global,
            "tau_keys": TAU_KEYS,
            "dim": D,
            "n_static": N_STATIC,
            "n_pos": N_POS,
            "tau_cos": args.tau_cos,
            "tau_static_cos": args.tau_static_cos,
            "pos_mode": args.pos_mode,
            "pos_lengthscale": args.pos_lengthscale,
            "pos_basis": args.pos_basis,
            "rff_omega": rff_omega,
            "rff_phi": rff_phi,
            "gauss_mu": gauss_mu,
            "gauss_sign": gauss_sign,
            "pos_mean": pos_mean,
            "pos_std": pos_std,
            "pos_leak_t": t_leak if N_POS > 0 else None,
            "seed": args.seed,
            "noise_scale": args.noise_scale,
        },
        out_dir / "matrices.pt",
    )

    # Save the full [N, 8, D] structure for NFP test (video-level grouping needed)
    # Filename starts with 'all' so ActivationsDataset ignores it
    torch.save(
        {
            "h": h,  # [N, 8, D]
            "tau": tau,  # [N, 8, 5]  original (unnormalized)
            "tau_norm": tau_norm,  # [N, 8, 5]  Z-scored
            "pos": pos,  # [N, 8, 2]  ball position (m)
            "g": g,  # [N, 8, N_POS] position-channel values (or None)
            "video_ids": [vd.name for vd in video_dirs],
        },
        out_dir / "all_videos.pt",
    )

    # --- Chunked flat activations for SAE training ---
    n_val = int(N * args.val_frac)
    n_train = N - n_val
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    h_train = h[train_idx].reshape(-1, D)  # [n_train*8, D]
    h_val = h[val_idx].reshape(-1, D)  # [n_val*8,   D]

    def save_chunks(data: torch.Tensor, split_dir: Path, chunk_size: int = 4096) -> None:
        """Save a flat activation tensor to disk as fixed-size chunk files.

        Args:
            data: Flat activations, shape [n_steps, D].
            split_dir: Destination directory (created if absent).
            chunk_size: Maximum number of rows per saved chunk file.

        Returns:
            None. Writes ``activations_part{i}.pt`` files into ``split_dir``.
        """
        split_dir.mkdir(parents=True, exist_ok=True)
        n = data.shape[0]
        for i, start in enumerate(range(0, n, chunk_size)):
            torch.save(data[start : start + chunk_size], split_dir / f"activations_part{i}.pt")
        print(f"  Saved {i+1} chunk(s) -> {split_dir}")

    print(f"\nSaving SAE training chunks...")
    save_chunks(h_train, out_dir / "train")
    save_chunks(h_val, out_dir / "val")

    print(f"\nDone.")
    print(f"  Train: {n_train} videos  ({h_train.shape[0]} time-steps)")
    print(f"  Val:   {n_val}   videos  ({h_val.shape[0]} time-steps)")
    print(f"  W_tau and W_static saved to: {out_dir / 'matrices.pt'}")
    print(f"  Full video structure saved to: {out_dir / 'all_videos.pt'}")


if __name__ == "__main__":
    main()
