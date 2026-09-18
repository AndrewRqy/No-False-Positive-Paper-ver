"""
Experiment D3 — reverse-play mediation (arrow-of-time test).

Corruption that isolates ONE temporal property: play the video backward. Direction
percepts invert; appearance and speed statistics are identical (same frames, same
inter-frame magnitudes). Three questions:
  (a) Does reversal move the model's prediction toward the opposite direction class?
  (b) Does patching the 85 NFP features' FORWARD-run activations into the reversed run
      restore the forward percept? (vs random-85 / static-85 / all-6144 ceiling)
  (c) Tag signature: which features change activation under reversal? vel_x/vel_y-tagged
      features should change most; speed-tagged and static features least.

Inputs: the 5 direction pairs (pushing/pulling/camera left-right, moving/camera up-down),
12 videos per side. Reversal = frame order flipped (index 15..0). Restoration uses the
SteerLayer token-level patch mode (f[..., idx] <- forward values, decode + re-add error).

Replication: seeds fixed (0); videos are the first 12 per class of the seed-1-shuffled
validation list, identical to expB2/expC1. Model MCG-NJU/videomae-base-finetuned-ssv2;
SAE local_runs/sae/ae.pt (layer-11 post-MLP, dict 6144); NFP flags = Bonferroni 0.05/6144
on local_runs/nfp_results/sae_nfp.pt; static pool = finite t, max|t| < 2.

Usage (from sae-for-vlm/):
  python analysis/steer_reverse_play.py --per_class 12
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from causal_analysis.common.steer_ssv2_logits import SteerLayer, ssv2_collate
from causal_analysis.common.steer_pair_screen import ItemFrames

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
DIR_PAIRS = [
    ("push_lr", "Pushing [something] from left to right", "Pushing [something] from right to left"),
    ("pull_lr", "Pulling [something] from left to right", "Pulling [something] from right to left"),
    ("cam_lr",  "Turning the camera left while filming [something]",
                "Turning the camera right while filming [something]"),
    ("move_ud", "Moving [something] up", "Moving [something] down"),
    ("cam_ud",  "Turning the camera upwards while filming [something]",
                "Turning the camera downwards while filming [something]"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD3_reverse_play.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy(); t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = [int(i) for i in np.where((p_all < bonf).any(1))[0]]
    dom = {i: TAU[int(np.argmax(np.abs(t_all[i])))] for i in sig}
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    rng = np.random.RandomState(args.seed + 7)
    rnd85 = sorted(rng.choice([k for k in range(sae.dict_size) if k not in set(sig)],
                              85, replace=False))
    stat85 = sorted(rng.choice(static_pool, 85, replace=False))
    sets = [("NFP-85", torch.tensor(sorted(sig))),
            ("random-85", torch.tensor(rnd85)),
            ("static-85", torch.tensor(stat85)),
            ("ALL-6144", torch.arange(sae.dict_size))]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def probs(cache, patch=None):
        outs = []
        for bi, pv in enumerate(cache):
            if patch is not None:
                steer.patch_idx = patch[0]; steer.patch_vals = patch[1][bi]
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
            steer.patch_idx = steer.patch_vals = None
        return torch.cat(outs, 0).numpy()

    def capture(cache, idx):
        steer.record_tokens_idx = idx; steer.captured_tokens = []
        for pv in cache:
            with torch.no_grad():
                clf(pixel_values=pv.to(device))
        steer.record_tokens_idx = None
        vals, i = [], 0
        allv = torch.cat(steer.captured_tokens, 0)
        for pv in cache:
            vals.append(allv[i:i + pv.shape[0]]); i += pv.shape[0]
        return vals

    # (c) tag signature accumulator: per-feature |mean act fwd - mean act rev|
    act_fwd_sum = np.zeros(sae.dict_size); act_rev_sum = np.zeros(sae.dict_size); n_clips = 0

    results = {"pairs": [], "sets": [n for n, _ in sets]}
    agg = {"fwd": [], "rev": [], **{n: [] for n, _ in sets}}
    for key, pos_l, neg_l in DIR_PAIRS:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        if cp < 0 or cn < 0 or min(len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, []))) < args.per_class:
            continue
        rec = {"key": key, "sides": {}}
        for side, cls_l, own, other in [("pos", pos_l, cp, cn), ("neg", neg_l, cn, cp)]:
            cache = cache_of(by_tmpl[cls_l][: args.per_class])
            cache_rev = [pv.flip(1) for pv in cache]     # frame order reversed

            steer.enabled = False
            # mean-pooled activation signature (record mode)
            steer.record = True; steer.captured = []
            P_fwd = probs(cache)
            A_fwd = torch.cat(steer.captured, 0).numpy(); steer.captured = []
            P_rev = probs(cache_rev)
            A_rev = torch.cat(steer.captured, 0).numpy(); steer.record = False
            act_fwd_sum += A_fwd.sum(0); act_rev_sum += A_rev.sum(0); n_clips += A_fwd.shape[0]

            def lo(P):
                return float(np.mean(np.log(P[:, own] + 1e-12) - np.log(P[:, other] + 1e-12)))
            row = {"fwd": lo(P_fwd), "rev": lo(P_rev),
                   "rev_flip_rate": float((P_rev[:, other] > P_rev[:, own]).mean())}
            agg["fwd"].append(row["fwd"]); agg["rev"].append(row["rev"])
            for name, idx in sets:
                fvals = capture(cache, idx)
                P_r = probs(cache_rev, patch=(idx, fvals))
                row[name] = lo(P_r)
                agg[name].append(row[name])
            rec["sides"][side] = {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in row.items()}
        print(f"  {key:<9} fwd LO={np.mean([rec['sides'][s]['fwd'] for s in rec['sides']]):+.2f} "
              f"rev={np.mean([rec['sides'][s]['rev'] for s in rec['sides']]):+.2f} "
              f"rev-flip={np.mean([rec['sides'][s]['rev_flip_rate'] for s in rec['sides']]):.2f}")
        results["pairs"].append(rec)

    f, r = np.mean(agg["fwd"]), np.mean(agg["rev"])
    print(f"\n=== MEAN LO_own: forward {f:+.3f}  reversed {r:+.3f}  (destroyed {f - r:.3f}) ===")
    summary = {"fwd": round(float(f), 3), "rev": round(float(r), 3)}
    for name, _ in sets:
        x = np.mean(agg[name])
        rev_frac = (x - r) / (f - r + 1e-9)
        summary[name] = {"LO": round(float(x), 3), "recovery": round(float(rev_frac), 3)}
        print(f"  restore {name:<10} {x:+.3f}   recovery = {100*rev_frac:.0f}%")

    # tag signature
    d_act = np.abs(act_fwd_sum - act_rev_sum) / max(n_clips, 1)
    base = act_fwd_sum / max(n_clips, 1) + 1e-6
    rel = d_act / base
    groups = {t: [k for k in sig if dom[k] == t] for t in TAU}
    groups["static-85"] = stat85
    print("\n=== reversal sensitivity |d mean act| / mean act, by feature tag ===")
    sig_tbl = {}
    for g, ks in groups.items():
        if not ks:
            continue
        v = float(np.median(rel[ks]))
        sig_tbl[g] = round(v, 3)
        print(f"  {g:<10} n={len(ks):3d}  median rel change = {v:.3f}")
    results["summary"] = summary
    results["tag_signature"] = sig_tbl
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
