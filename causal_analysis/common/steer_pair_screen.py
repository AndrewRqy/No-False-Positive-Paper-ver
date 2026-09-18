"""
Experiment B2 — systematic pair screen: ALL identified temporal features x ALL SSv2
temporal class pairs, steered in both directions.

Purpose (original claim, clean logic): NFP flagged 85 features as temporal with NO reference
to the classifier. Steer each of them, unselected, on every SSv2 class pair that differs only
in a temporal property, in both clamp directions, and record whether classification shifts
between the pair classes. First deliverable is the raw feature x pair matrix; pattern-finding
(consistency across same-axis pairs, per-tau structure) comes after; the random-feature
population comparison is deferred until we see which features move anything.

Per (feature, pair):
  - 12 real videos per class side, cached once per pair.
  - clamp feature to s = +S and s = -S on all 1568 tokens of every video (both sides).
  - LO(video) = log P(pos_class) - log P(neg_class); pair axis position of the prediction.
  - delta = mean over videos of [LO(+S) - LO(-S)] / 2   (signed steering power along the axis)
  - flips, best orientation: orientation A uses +S to push toward pos and -S toward neg;
    orientation B the mirror. flip = mean(rate(neg-side videos rank pos>neg under the
    toward-pos clamp), rate(pos-side videos rank neg>pos under the toward-neg clamp)).
    best_flip = max(A, B); top1 variant analogous with strict argmax == target.

Usage (from sae-for-vlm/):
  python analysis/steer_pair_screen.py --per_class 12 --s_abs 150
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from causal_analysis.common.steer_ssv2_logits import SteerLayer, SSv2Frames, ssv2_collate

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]

# every SSv2 pair distinguished (near-)only by a temporal property; axis tag = the
# kinematic quantity that differs. Missing labels are skipped with a printed notice.
PAIRS = [
    ("push_lr",   "vel_x", "Pushing [something] from left to right",
                           "Pushing [something] from right to left"),
    ("pull_lr",   "vel_x", "Pulling [something] from left to right",
                           "Pulling [something] from right to left"),
    ("cam_lr",    "vel_x", "Turning the camera left while filming [something]",
                           "Turning the camera right while filming [something]"),
    ("move_ud",   "vel_y", "Moving [something] up",
                           "Moving [something] down"),
    ("cam_ud",    "vel_y", "Turning the camera upwards while filming [something]",
                           "Turning the camera downwards while filming [something]"),
    ("obj_cam",   "depth", "Moving [something] towards the camera",
                           "Moving [something] away from the camera"),
    ("obj_obj",   "depth", "Moving [something] closer to [something]",
                           "Moving [something] away from [something]"),
    ("cam_appr",  "depth", "Approaching [something] with your camera",
                           "Moving away from [something] with your camera"),
    ("fall_speed","speed", "[Something] falling like a rock",
                           "[Something] falling like a feather or paper"),
    ("spin_stop", "accel", "Spinning [something] so it continues spinning",
                           "Spinning [something] that quickly stops spinning"),
]


class ItemFrames(SSv2Frames):
    def __init__(self, videos_dir, items):
        self.items = items
        self.dir = Path(videos_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--s_abs", default=150.0, type=float)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--out_csv", default="local_runs/steering/expB2_pair_screen.csv")
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
    print(f"{len(sig)} NFP temporal features | clamp +/-{args.s_abs} | {len(PAIRS)} pairs")

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def probs(cache):
        outs = []
        for pv in cache:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy()

    # --- prepare pairs: verify labels, cache videos, baseline ---
    pairs = []
    for key, axis, pos_l, neg_l in PAIRS:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        np_, nn_ = len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, []))
        if cp < 0 or cn < 0 or min(np_, nn_) < args.per_class:
            print(f"  SKIP {key}: '{pos_l}' / '{neg_l}' (found idx {cp},{cn}; avail {np_},{nn_})")
            continue
        caches = {"pos": cache_of(by_tmpl[pos_l][: args.per_class]),
                  "neg": cache_of(by_tmpl[neg_l][: args.per_class])}
        steer.enabled = False
        Pb = {s: probs(caches[s]) for s in ["pos", "neg"]}
        acc = (float((Pb["pos"][:, cp] > Pb["pos"][:, cn]).mean()),
               float((Pb["neg"][:, cn] > Pb["neg"][:, cp]).mean()))
        print(f"  {key:<10} [{axis:5s}] baseline pair-acc pos={acc[0]:.2f} neg={acc[1]:.2f}")
        pairs.append({"key": key, "axis": axis, "pos": pos_l, "neg": neg_l,
                      "cp": cp, "cn": cn, "caches": caches, "Pb": Pb, "acc": acc})

    S = args.s_abs
    rows = []
    for fi, k in enumerate(sig):
        rec = {"feature": f"feat{k:05d}", "idx": k, "dom_tau": dom[k], "pairs": {}}
        for pr in pairs:
            cp, cn = pr["cp"], pr["cn"]
            P = {}
            for side in ["pos", "neg"]:
                for s in [+S, -S]:
                    steer.enabled = True; steer.k = k; steer.s = s
                    P[(side, s)] = probs(pr["caches"][side])
            steer.enabled = False

            def lo(Pm):
                return np.log(Pm[:, cp] + 1e-12) - np.log(Pm[:, cn] + 1e-12)

            delta = float(np.mean([lo(P[(sd, +S)]).mean() - lo(P[(sd, -S)]).mean()
                                   for sd in ["pos", "neg"]]) / 2.0)
            # orientation A: +S pushes toward pos; orientation B mirrored
            def flip(orient):
                s_pos, s_neg = (+S, -S) if orient == "A" else (-S, +S)
                f_np = float((P[("neg", s_pos)][:, cp] > P[("neg", s_pos)][:, cn]).mean())
                f_pn = float((P[("pos", s_neg)][:, cn] > P[("pos", s_neg)][:, cp]).mean())
                t_np = float((P[("neg", s_pos)].argmax(1) == cp).mean())
                t_pn = float((P[("pos", s_neg)].argmax(1) == cn).mean())
                return (f_np + f_pn) / 2, (t_np + t_pn) / 2
            fA, tA = flip("A"); fB, tB = flip("B")
            best_flip, best_top1 = (fA, tA) if fA >= fB else (fB, tB)
            rec["pairs"][pr["key"]] = {"delta": round(delta, 3),
                                       "best_flip": round(best_flip, 3),
                                       "best_top1": round(best_top1, 3)}
        deltas = [abs(v["delta"]) for v in rec["pairs"].values()]
        flips = [v["best_flip"] for v in rec["pairs"].values()]
        rec["mean_abs_delta"] = round(float(np.mean(deltas)), 3)
        rec["n_pairs_flip25"] = int(sum(f >= 0.25 for f in flips))
        rec["n_pairs_flip50"] = int(sum(f >= 0.50 for f in flips))
        rows.append(rec)
        if (fi + 1) % 10 == 0 or fi == len(sig) - 1:
            print(f"  screened {fi+1}/{len(sig)} features")

    # --- save ---
    out = {"s_abs": S, "per_class": args.per_class,
           "pairs": [{k: pr[k] for k in ["key", "axis", "pos", "neg", "acc"]} for pr in pairs],
           "features": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    pair_keys = [pr["key"] for pr in pairs]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "dom_tau", "mean_abs_delta", "n_pairs_flip25", "n_pairs_flip50"]
                   + [f"{pk}_{m}" for pk in pair_keys for m in ["delta", "flip", "top1"]])
        for r in sorted(rows, key=lambda r: (-r["n_pairs_flip50"], -r["mean_abs_delta"])):
            w.writerow([r["feature"], r["dom_tau"], r["mean_abs_delta"],
                        r["n_pairs_flip25"], r["n_pairs_flip50"]]
                       + [r["pairs"][pk][m] for pk in pair_keys
                          for m in ["delta", "best_flip", "best_top1"]])

    print(f"\n=== top 15 features by (n_pairs_flip50, mean|delta|) ===")
    for r in sorted(rows, key=lambda r: (-r["n_pairs_flip50"], -r["mean_abs_delta"]))[:15]:
        per = "  ".join(f"{pk}:{r['pairs'][pk]['best_flip']:.2f}" for pk in pair_keys)
        print(f"  {r['feature']} [{r['dom_tau']:9s}] mean|d|={r['mean_abs_delta']:5.2f} "
              f"flip50={r['n_pairs_flip50']}  {per}")
    print(f"\nSaved -> {args.out}\n         {args.out_csv}")


if __name__ == "__main__":
    main()
