"""
Convert ALL NFP ball videos (16 PNG frames each) into looping GIFs for eyeballing.

Same construction as the demos (analysis/make_nfp_demo.py): a GIF is just the 16
existing frames sequenced at a fixed fps - NO extra/interpolated frames are added.
The only transforms are an optional nearest-neighbor upscale (for visibility) and
BGR->RGB. Raw frames, no text/marker overlay, so the actual ball motion is clean to
inspect. One GIF per video, named by its video id (v00000.gif ...), plus a manifest.csv
mapping id -> profile_type / mean speed / max speed / on_screen count so high-activation
ids can be correlated against motion type.

Usage (from repo root):
    python analysis/make_all_nfp_gifs.py --nfp_dir data/output/nfp --out_dir data/output/nfp_gifs
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

N_FRAMES = 16


def video_meta(vdir: Path):
    meta = json.loads((vdir / "metadata.json").read_text())
    traj = meta.get("trajectory", [])
    spd = [f["tau"]["speed"] for f in traj] or [0.0]
    pt = (meta.get("profile") or {}).get("profile_type", "unknown")
    ons = sum(1 for f in traj if f.get("on_screen", True))
    return pt, sum(spd) / len(spd), max(spd), ons, len(traj)


def make_one(args):
    """Module-level worker (picklable for ProcessPoolExecutor on Windows)."""
    vdir_str, out_str, scale, fps = args
    vdir, out_dir = Path(vdir_str), Path(out_str)
    import imageio.v2 as imageio

    frames = []
    for i in range(N_FRAMES):
        img = cv2.imread(str(vdir / f"rgba_{i:05d}.png"))
        if img is None:
            return (vdir.name, None, f"missing frame {i}")
        if scale != 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(str(out_dir / f"{vdir.name}.gif"), frames, duration=1.0 / fps, loop=0)
    try:
        info = video_meta(vdir)
    except Exception:
        info = ("unknown", 0.0, 0.0, N_FRAMES, N_FRAMES)
    return (vdir.name, info, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfp_dir", default="data/output/nfp")
    ap.add_argument("--out_dir", default="data/output/nfp_gifs")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N (smoke test)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = sorted(Path(args.nfp_dir).glob("v*"))
    if args.limit:
        dirs = dirs[: args.limit]
    print(
        f"Converting {len(dirs)} videos -> {out_dir}  (fps={args.fps}, scale={args.scale}, "
        f"workers={args.workers})"
    )

    tasks = [(str(d), str(out_dir), args.scale, args.fps) for d in dirs]
    rows, fails, done = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(make_one, t) for t in tasks]
        for fut in as_completed(futs):
            name, info, err = fut.result()
            done += 1
            if err:
                fails.append((name, err))
            else:
                pt, mean_s, max_s, ons, total = info
                rows.append((name, pt, mean_s, max_s, ons, total))
            if done % 250 == 0:
                print(f"  {done}/{len(dirs)} done ({len(fails)} failed)")

    rows.sort()
    lines = ["video_id,profile_type,mean_speed,max_speed,on_screen,total_frames"]
    for name, pt, mean_s, max_s, ons, total in rows:
        lines.append(f"{name},{pt},{mean_s:.4f},{max_s:.4f},{ons},{total}")
    (out_dir / "manifest.csv").write_text("\n".join(lines) + "\n")

    print(f"\nDone. {len(rows)} GIFs written, {len(fails)} failed -> {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")
    if fails:
        print("Failures (first 10):", fails[:10])


if __name__ == "__main__":
    main()
