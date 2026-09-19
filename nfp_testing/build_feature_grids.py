"""
Per-significant-feature demo assets for eyeballing what drives each temporal feature.

For every VideoMAE SAE feature the NFP test deems significant (p < 0.05/6144 for >=1 tau),
rank the 3000 NFP ball videos by that feature's activation (per-video masked-mean over
on-screen steps, from analysis/dump_nfp_feature_acts.py) and produce:

  feat<idx>_<taus>/
      feat<idx>_top16_grid.gif   4x4 montage of the 16 highest-activation videos, all 16
                                 frames played in lockstep so shared motion pops out.
      top50/                     the 50 highest-activation videos as individual GIFs
                                 (copied from data/output/nfp_gifs, prefixed rank00.. so
                                 order is visible), plus ranking.csv (rank, video, activation,
                                 profile_type, max_speed, on_screen).

A grid GIF adds NO frames: it tiles the same 16 existing frames of 16 videos side by side
and steps through t=0..15 together. Each cell carries a small id + activation caption.

Usage (from repo root):
  python analysis/build_feature_grids.py \
     --nfp_results local_runs/nfp_results/sae_nfp.pt \
     --feat_acts   local_runs/nfp_results/sae_feat_acts.pt \
     --frames_dir  data/output/nfp --gifs_dir data/output/nfp_gifs \
     --out_dir     local_runs/nfp_feature_gifs
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

N_FRAMES = 16
TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def load_frames(vdir: Path):
    """16 BGR uint8 frames for a video."""
    fr = []
    for i in range(N_FRAMES):
        img = cv2.imread(str(vdir / f"rgba_{i:05d}.png"))
        fr.append(img)
    return fr


def make_grid_gif(top_vids, acts, frames_dir, out_path, cell=192, cap=18, fps=8):
    """top_vids: list of (video_id, activation). 4x4 grid, synced 16 frames."""
    import imageio.v2 as imageio

    cols = rows = 4
    cache = {}

    def frames_for(vid):
        if vid not in cache:
            cache[vid] = load_frames(Path(frames_dir) / vid)
        return cache[vid]

    gif_frames = []
    for t in range(N_FRAMES):
        canvas = np.zeros((rows * (cell + cap), cols * cell, 3), np.uint8)
        for k, (vid, a) in enumerate(top_vids[: rows * cols]):
            r, c = divmod(k, cols)
            img = frames_for(vid)[t]
            img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
            y0, x0 = r * (cell + cap), c * cell
            canvas[y0 : y0 + cell, x0 : x0 + cell] = img
            cv2.putText(
                canvas,
                f"{vid} a={a:.2f}",
                (x0 + 3, y0 + cell + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    imageio.mimsave(str(out_path), gif_frames, duration=1.0 / fps, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--feat_acts", default="local_runs/nfp_results/sae_feat_acts.pt")
    ap.add_argument("--frames_dir", default="data/output/nfp")
    ap.add_argument("--gifs_dir", default="data/output/nfp_gifs")
    ap.add_argument("--out_dir", default="local_runs/nfp_feature_gifs")
    ap.add_argument("--manifest", default="data/output/nfp_gifs/manifest.csv")
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--rank_by", default="mean", choices=["mean", "max"])
    ap.add_argument("--top_grid", default=16, type=int)
    ap.add_argument("--top_folder", default=50, type=int)
    args = ap.parse_args()

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_val = nfp["p_val"].numpy()  # [6144, 5]
    t_stat = nfp["t_stat"].numpy()
    D = p_val.shape[0]
    bonf = args.alpha / D
    sig_any = (p_val < bonf).any(axis=1)
    sig_idx = np.where(sig_any)[0]
    print(f"{len(sig_idx)} significant features (p < {bonf:.2e})")

    fa = torch.load(args.feat_acts, map_location="cpu")
    act = fa["feat_mean" if args.rank_by == "mean" else "feat_max"].numpy()  # [N, 6144]
    vids = list(fa["video_ids"])
    assert act.shape[1] == D, f"feature dim mismatch {act.shape[1]} vs {D}"

    # per-video metadata for the ranking csv
    meta = {}
    mpath = Path(args.manifest)
    if mpath.exists():
        for line in mpath.read_text().splitlines()[1:]:
            f = line.split(",")
            if len(f) >= 6:
                meta[f[0]] = (f[1], f[3], f[4])  # profile_type, max_speed, on_screen

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    index = ["feature,taus,n_nonzero_videos," + ",".join(f"t_{k}" for k in TAU_KEYS)]

    for n, i in enumerate(sig_idx):
        taus = [TAU_KEYS[k] for k in range(5) if p_val[i, k] < bonf]
        tag = "-".join(taus)
        col = act[:, i]
        order = np.argsort(-col)
        nz = int((col > 0).sum())
        fdir = out_root / f"feat{i:05d}_{tag}"
        (fdir / "top50").mkdir(parents=True, exist_ok=True)

        top16 = [(vids[j], float(col[j])) for j in order[: args.top_grid]]
        make_grid_gif(top16, col, args.frames_dir, fdir / f"feat{i:05d}_top16_grid.gif")

        rows = ["rank,video_id,activation,profile_type,max_speed,on_screen"]
        for rank, j in enumerate(order[: args.top_folder]):
            vid = vids[j]
            src = Path(args.gifs_dir) / f"{vid}.gif"
            if src.exists():
                shutil.copyfile(src, fdir / "top50" / f"rank{rank:02d}_{vid}.gif")
            pt, ms, ons = meta.get(vid, ("?", "?", "?"))
            rows.append(f"{rank},{vid},{col[j]:.5f},{pt},{ms},{ons}")
        (fdir / "ranking.csv").write_text("\n".join(rows) + "\n")

        index.append(f"feat{i:05d},{tag},{nz}," + ",".join(f"{t_stat[i,k]:.2f}" for k in range(5)))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(sig_idx)} features done")

    (out_root / "index.csv").write_text("\n".join(index) + "\n")
    print(f"\nDone. {len(sig_idx)} feature dirs -> {out_root}")
    print(f"Index: {out_root / 'index.csv'}")


if __name__ == "__main__":
    main()
