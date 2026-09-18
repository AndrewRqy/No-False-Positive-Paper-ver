"""
Experiment C2 — shuffle-and-restore (causal tracing for the temporal features).

Corrupt the input's temporal structure, then restore only the identified features and
measure how much of the temporal percept comes back.

Corruption: permute the 8 two-frame tubelet blocks of the input video. One permutation
is drawn per class-side evaluation set and applied to every clip on that side, shared
across conditions (NOT per video; the all-class variant steer_global_restore samples a
fresh permutation per video). Every frame's appearance is intact; motion order is
destroyed. This is the standard temporal-corruption control in video interpretability.

Restoration: run the shuffled video, but at layer 11 replace the chosen features'
token-level activations with their values from the CLEAN run (decode + re-add error).
If the NFP features carry the temporal percept, restoring 85 of 6144 features (1.4%)
should recover a disproportionate share of the discrimination the shuffle destroyed.

Recovery on the pair task: for videos of temporal class pairs, LO_own = log P(own class)
- log P(other class). recovery = (restored - shuffled) / (clean - shuffled).

Sets: NFP-85, flippers-12, random-85, static-85, all-6144 (ceiling).

Usage (from sae-for-vlm/):
  python analysis/steer_shuffle_restore.py --per_class 12
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
from causal_analysis.common.steer_pair_screen import PAIRS, ItemFrames
from utils.config import add_config_arg, apply_config


# shared shuffle primitives (re-exported so importers keep working)
from causal_analysis.common.shuffle_util import shuffle_blocks, random_block_perm  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expC2_shuffle_restore.json")
    args = ap.parse_args()
    apply_config(ap, args)
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
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    screen = json.load(open(args.screen_json))
    flippers = [r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1]
    rng = np.random.RandomState(args.seed + 7)
    sets = [("NFP temporal x85", sig),
            (f"flippers x{len(flippers)}", flippers),
            ("random x85", rng.choice([k for k in range(sae.dict_size)
                                       if k not in set(sig)], 85, replace=False).tolist()),
            ("static x85", rng.choice(static_pool, 85, replace=False).tolist()),
            ("ALL 6144 (ceiling)", list(range(sae.dict_size)))]

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
            steer.patch_idx = None; steer.patch_vals = None
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

    prng = np.random.RandomState(args.seed + 99)

    temporal_pairs = [p for p in PAIRS if p[1] != "depth"]
    results = {"pairs": [], "sets": [n for n, _ in sets]}
    agg = {"clean": [], "shuffled": [], **{n: [] for n, _ in sets}}
    for key, axis_tag, pos_l, neg_l in temporal_pairs:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        if cp < 0 or cn < 0 or min(len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, []))) < args.per_class:
            continue
        rec = {"key": key, "sides": {}}
        for side, cls_l, own, other in [("pos", pos_l, cp, cn), ("neg", neg_l, cn, cp)]:
            cache = cache_of(by_tmpl[cls_l][: args.per_class])
            # one permutation per class-side, shared across every clip and condition
            perm = random_block_perm(prng)
            cache_sh = [shuffle_blocks(pv, perm) for pv in cache]

            steer.enabled = False
            P_clean = probs(cache)
            P_sh = probs(cache_sh)

            def lo(P):
                return float(np.mean(np.log(P[:, own] + 1e-12) - np.log(P[:, other] + 1e-12)))
            row = {"clean": lo(P_clean), "shuffled": lo(P_sh)}
            agg["clean"].append(row["clean"]); agg["shuffled"].append(row["shuffled"])
            for name, ks in sets:
                idx = torch.tensor(sorted(set(ks)))
                cvals = capture(cache, idx)                     # clean activations
                P_r = probs(cache_sh, patch=(idx, cvals))       # restore into shuffled run
                row[name] = lo(P_r)
                agg[name].append(row[name])
            rec["sides"][side] = {k: round(v, 3) for k, v in row.items()}
        drop = np.mean([rec["sides"][s]["clean"] - rec["sides"][s]["shuffled"]
                        for s in rec["sides"]])
        print(f"  {key:<11} clean LO={np.mean([rec['sides'][s]['clean'] for s in rec['sides']]):+.2f} "
              f"shuffled={np.mean([rec['sides'][s]['shuffled'] for s in rec['sides']]):+.2f} "
              f"(drop {drop:+.2f})")
        results["pairs"].append(rec)

    print("\n=== MEAN LO_own ACROSS PAIRS (recovery of the shuffle-destroyed signal) ===")
    c, s = np.mean(agg["clean"]), np.mean(agg["shuffled"])
    print(f"  clean    {c:+.3f}")
    print(f"  shuffled {s:+.3f}   (destroyed: {c - s:.3f})")
    summary = {"clean": round(float(c), 3), "shuffled": round(float(s), 3)}
    for name, _ in sets:
        r = np.mean(agg[name])
        rev = (r - s) / (c - s + 1e-9)
        summary[name] = {"LO": round(float(r), 3), "recovery": round(float(rev), 3)}
        print(f"  restore {name:<22} {r:+.3f}   recovery = {100*rev:.0f}%")
    results["summary"] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
