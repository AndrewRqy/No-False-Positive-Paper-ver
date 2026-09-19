"""
Enrich each significant-feature ranking.csv with ground-truth kinematics, and write a
top-level feature_summary.csv aggregating the top-K videos per feature.

The NFP ball videos carry full ground truth in metadata.json (per-frame tau + the
generative `profile`). Pure eyeballing of the grid GIFs is hard, so for every video that
ranks high on a feature we attach what the ball actually did:

  per-video (rewritten into each feat*/ranking.csv):
    activation, profile_type, family, on_screen,
    mean_speed, min_speed, max_speed, speed_range,      (does it speed up / slow down?)
    mean_accel, max_accel,                               (does it accelerate?)
    turn_total_deg, net_turn_deg,                        (does direction change? by how much?)
    mean_vel_x, mean_vel_y,                              (net drift / dominant axis)
    profile_speed_mps, profile_delta_deg                (generative params)

  per-feature (top-level feature_summary.csv): one row per feature aggregating its top-K -
    the dominant profile_type (+ its fraction), mean speed / speed-range / accel / turn,
    fraction of clips that turn / that accelerate, alongside the feature's significant tau(s)
    and t-stats. Scan these 85 rows to spot what each feature selects for.

Direction is in radians in tau; step-to-step deltas are wrapped to (-pi, pi] before summing,
so the angle wraparound (e.g. +2.57 -> -2.54 rad = a 67 deg turn) is measured correctly.

Usage (from repo root):
  python analysis/enrich_feature_rankings.py
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
TURN_THRESH_DEG = 20.0  # a clip "turns" if total heading change exceeds this
ACCEL_THRESH = 0.10  # a clip "accelerates" if max accel_mag exceeds this


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def video_descriptors(vdir: Path):
    """Ground-truth kinematic summary for one video from metadata.json."""
    m = json.loads((vdir / "metadata.json").read_text())
    prof = m.get("profile", {})
    traj = m["trajectory"]
    spd = [f["tau"]["speed"] for f in traj]
    acc = [f["tau"]["accel_mag"] for f in traj]
    dirs = [f["tau"]["direction"] for f in traj]
    vx = [f["tau"]["vel_x"] for f in traj]
    vy = [f["tau"]["vel_y"] for f in traj]
    ons = sum(1 for f in traj if f.get("on_screen", True))
    turn_total = sum(abs(wrap_pi(dirs[i + 1] - dirs[i])) for i in range(len(dirs) - 1))
    net_turn = wrap_pi(dirs[-1] - dirs[0])
    return {
        "profile_type": prof.get("profile_type", "?"),
        "family": prof.get("family", "?"),
        "on_screen": ons,
        "mean_speed": float(np.mean(spd)),
        "min_speed": float(np.min(spd)),
        "max_speed": float(np.max(spd)),
        "speed_range": float(np.max(spd) - np.min(spd)),
        "mean_accel": float(np.mean(acc)),
        "max_accel": float(np.max(acc)),
        "turn_total_deg": math.degrees(turn_total),
        "net_turn_deg": math.degrees(net_turn),
        "mean_vel_x": float(np.mean(vx)),
        "mean_vel_y": float(np.mean(vy)),
        "profile_speed_mps": float(prof.get("speed_mps", float("nan"))),
        "profile_delta_deg": float(prof.get("delta_deg", 0.0)),
    }


RANK_COLS = [
    "rank",
    "video_id",
    "activation",
    "profile_type",
    "family",
    "on_screen",
    "mean_speed",
    "min_speed",
    "max_speed",
    "speed_range",
    "mean_accel",
    "max_accel",
    "turn_total_deg",
    "net_turn_deg",
    "mean_vel_x",
    "mean_vel_y",
    "profile_speed_mps",
    "profile_delta_deg",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--feat_acts", default="local_runs/nfp_results/sae_feat_acts.pt")
    ap.add_argument("--frames_dir", default="data/output/nfp")
    ap.add_argument("--out_dir", default="local_runs/nfp_feature_gifs")
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--top_folder", default=50, type=int)
    args = ap.parse_args()

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_val = nfp["p_val"].numpy()
    t_stat = nfp["t_stat"].numpy()
    D = p_val.shape[0]
    bonf = args.alpha / D
    sig_idx = np.where((p_val < bonf).any(axis=1))[0]

    fa = torch.load(args.feat_acts, map_location="cpu")
    act = fa["feat_mean"].numpy()
    vids = list(fa["video_ids"])
    print(f"{len(sig_idx)} significant features; enriching top-{args.top_folder} each")

    cache = {}

    def desc(vid):
        if vid not in cache:
            cache[vid] = video_descriptors(Path(args.frames_dir) / vid)
        return cache[vid]

    out_root = Path(args.out_dir)
    summary = [
        "feature,taus,n_nonzero_videos,"
        + ",".join(f"t_{k}" for k in TAU_KEYS)
        + ","
        + "top_dominant_profile,top_profile_frac,top_mean_activation,"
        "top_mean_speed,top_mean_speed_range,top_mean_accel,top_frac_accel,"
        "top_mean_turn_deg,top_frac_turning,top_mean_onscreen"
    ]

    for i in sig_idx:
        taus = [TAU_KEYS[k] for k in range(5) if p_val[i, k] < bonf]
        tag = "-".join(taus)
        col = act[:, i]
        order = np.argsort(-col)[: args.top_folder]
        fdir = out_root / f"feat{i:05d}_{tag}"
        if not fdir.exists():
            # tolerate a different existing tag spelling - find by prefix
            cand = list(out_root.glob(f"feat{i:05d}_*"))
            fdir = cand[0] if cand else fdir
            fdir.mkdir(parents=True, exist_ok=True)

        rows = [",".join(RANK_COLS)]
        ds = []
        for rank, j in enumerate(order):
            vid = vids[j]
            d = desc(vid)
            ds.append(d)
            rows.append(
                ",".join(
                    [
                        str(rank),
                        vid,
                        f"{col[j]:.5f}",
                        d["profile_type"],
                        d["family"],
                        str(d["on_screen"]),
                        f"{d['mean_speed']:.4f}",
                        f"{d['min_speed']:.4f}",
                        f"{d['max_speed']:.4f}",
                        f"{d['speed_range']:.4f}",
                        f"{d['mean_accel']:.4f}",
                        f"{d['max_accel']:.4f}",
                        f"{d['turn_total_deg']:.2f}",
                        f"{d['net_turn_deg']:.2f}",
                        f"{d['mean_vel_x']:.4f}",
                        f"{d['mean_vel_y']:.4f}",
                        f"{d['profile_speed_mps']:.4f}",
                        f"{d['profile_delta_deg']:.2f}",
                    ]
                )
            )
        (fdir / "ranking.csv").write_text("\n".join(rows) + "\n")

        # aggregate across the top-K for the summary row
        profs = Counter(d["profile_type"] for d in ds)
        dom, dom_n = profs.most_common(1)[0]
        n = len(ds)
        summary.append(
            ",".join(
                [
                    f"feat{i:05d}",
                    tag,
                    str(int((col > 0).sum())),
                    *[f"{t_stat[i,k]:.2f}" for k in range(5)],
                    dom,
                    f"{dom_n/n:.2f}",
                    f"{float(np.mean(col[order])):.3f}",
                    f"{np.mean([d['mean_speed'] for d in ds]):.3f}",
                    f"{np.mean([d['speed_range'] for d in ds]):.3f}",
                    f"{np.mean([d['mean_accel'] for d in ds]):.3f}",
                    f"{np.mean([1.0 if d['max_accel'] > ACCEL_THRESH else 0.0 for d in ds]):.2f}",
                    f"{np.mean([d['turn_total_deg'] for d in ds]):.2f}",
                    f"{np.mean([1.0 if d['turn_total_deg'] > TURN_THRESH_DEG else 0.0 for d in ds]):.2f}",
                    f"{np.mean([d['on_screen'] for d in ds]):.2f}",
                ]
            )
        )

    (out_root / "feature_summary.csv").write_text("\n".join(summary) + "\n")
    print(f"Rewrote {len(sig_idx)} ranking.csv + feature_summary.csv -> {out_root}")


if __name__ == "__main__":
    main()
