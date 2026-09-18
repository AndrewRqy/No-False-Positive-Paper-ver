"""
Stitch the 16 frames of an NFP ball video into a playable demo clip (MP4 + GIF).

The NFP ball dataset stores each synthetic video as 16 PNGs (rgba_00000..rgba_00015.png)
plus a metadata.json with per-frame ground-truth kinematics (tau) and the ball position.
This turns a handful of those into short clips so the dataset / NFP test can be demoed.

With --annotate (default on) each frame is overlaid with: the frame index, the per-frame
speed and direction (two of the five tau variables the NFP test correlates against), and a
circle marking the ball-containing position — i.e. exactly the signal the test tracks.

Usage (from repo root):
    python analysis/make_nfp_demo.py --nfp_dir data/output/nfp --out_dir demo/nfp --n 10
"""
import argparse
import json
from pathlib import Path

import cv2


N_FRAMES = 16


def load_frames(vdir):
    frames = []
    for i in range(N_FRAMES):
        f = vdir / f"rgba_{i:05d}.png"
        img = cv2.imread(str(f))                      # BGR uint8
        if img is None:
            raise FileNotFoundError(f"missing frame {f}")
        frames.append(img)
    return frames


def annotate(frames, meta):
    """Overlay frame index + per-frame speed/direction TEXT only.

    Deliberately does NOT draw a marker on the ball: a drawn circle reads as a second
    ("hollow") ball next to the real rendered one. Text-only keeps the demo unambiguous.
    """
    traj = meta.get("trajectory", [])
    vid = meta.get("video_id", "")
    out = []
    for i, img in enumerate(frames):
        im = img.copy()
        rec = traj[i] if i < len(traj) else {}
        tau = rec.get("tau", {})
        cv2.putText(im, f"{vid}  frame {i:02d}/15", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"speed={tau.get('speed', 0):.2f}  dir={tau.get('direction', 0):.2f}",
                    (5, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        out.append(im)
    return out


def resize(frames, scale):
    if scale == 1:
        return frames
    h, w = frames[0].shape[:2]
    return [cv2.resize(f, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) for f in frames]


def write_mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def write_gif(frames, path, fps):
    import imageio.v2 as imageio
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(str(path), rgb, duration=1.0 / fps, loop=0)


def pick_dirs(nfp_dir, n, ids=None):
    root = Path(nfp_dir)
    if ids:
        dirs = [root / v for v in ids]
        for d in dirs:
            if not d.exists():
                raise FileNotFoundError(d)
        return dirs
    dirs = sorted(root.glob("v*"))
    if not dirs:
        raise FileNotFoundError(f"no video dirs in {nfp_dir}")
    if n >= len(dirs):
        return dirs
    # spread evenly across the dataset for variety, not just the first n
    step = len(dirs) / n
    return [dirs[int(i * step)] for i in range(n)]


def video_info(vdir):
    """(meta, profile_type, mean_speed, max_speed, on_screen_count) from metadata."""
    meta = json.loads((vdir / "metadata.json").read_text())
    traj = meta.get("trajectory", [])
    spd = [f["tau"]["speed"] for f in traj] or [0.0]
    pt = (meta.get("profile") or {}).get("profile_type", "unknown")
    ons = sum(1 for f in traj if f.get("on_screen", True))
    return meta, pt, sum(spd) / len(spd), max(spd), ons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfp_dir", default="data/output/nfp")
    ap.add_argument("--out_dir", default="demo/nfp")
    ap.add_argument("--n", type=int, default=10, help="number of demo videos (ignored if --ids given)")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit video dir names (e.g. v02460 v01394); overrides --n")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=2, help="upscale factor (nearest-neighbor)")
    ap.add_argument("--annotate", action="store_true",
                    help="overlay frame index + speed/direction TEXT (no ball marker); raw frames by default")
    ap.add_argument("--name_by_profile", action="store_true",
                    help="prefix output filenames with profile_type + max speed (self-documenting)")
    ap.add_argument("--no-gif", dest="gif", action="store_false")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = pick_dirs(args.nfp_dir, args.n, args.ids)
    print(f"Writing {len(dirs)} demo clips -> {out}  (fps={args.fps}, scale={args.scale}, "
          f"annotate={args.annotate})")
    manifest = []
    for vdir in dirs:
        meta, pt, mean_s, max_s, ons = video_info(vdir)
        off = "_offscreen" if ons < len(meta.get("trajectory", [])) else ""
        base = f"{pt}_s{max_s:.1f}{off}_{vdir.name}" if args.name_by_profile else vdir.name
        frames = load_frames(vdir)
        if args.annotate:
            frames = annotate(frames, meta)
        frames = resize(frames, args.scale)
        write_mp4(frames, out / f"{base}.mp4", args.fps)
        if args.gif:
            write_gif(frames, out / f"{base}.gif", args.fps)
        manifest.append((base, vdir.name, pt, mean_s, max_s, ons, len(meta.get("trajectory", []))))
        print(f"  {vdir.name}  profile={pt:14s} mean_spd={mean_s:.2f} max_spd={max_s:.2f} "
              f"on_screen={ons}/{len(meta.get('trajectory', []))} -> {base}")

    # self-documenting manifest of the demo set
    lines = ["# NFP demo clips\n",
             "Each clip is the 16 frames of one ball video (MP4 + GIF). `speed` is in m/s; "
             "`profile_type` is the motion profile (constant = linear/steady; turn/sinusoidal/"
             "back_and_forth = non-linear path; accel/decel/slow_fast_slow = non-linear speed). "
             "`on_screen` is how many of the 16 frames contain the ball — category demos are kept "
             "fully on-screen; clips tagged `_offscreen` deliberately show the ball leaving frame.\n",
             "| file | video | profile_type | mean speed | max speed | on_screen |",
             "|---|---|---|---|---|---|"]
    for base, vid, pt, mean_s, max_s, ons, total in sorted(manifest, key=lambda r: -r[4]):
        lines.append(f"| `{base}.mp4` / `.gif` | {vid} | {pt} | {mean_s:.2f} | {max_s:.2f} | {ons}/{total} |")
    (out / "manifest.md").write_text("\n".join(lines) + "\n")
    print(f"Done. Wrote {out / 'manifest.md'}")


if __name__ == "__main__":
    main()
