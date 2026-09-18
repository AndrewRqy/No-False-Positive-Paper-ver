"""
Experiment B — direction-pair flipping: the maximally specific steering demo.

SSv2 contains class pairs with identical spatial content that differ ONLY in motion
direction ("Pushing [something] from left to right" vs "... right to left"; "Moving
[something] up" vs "... down"). NFP tagged SAE features as vel_x / vel_y WITH A SIGN
(the t-stat sign; ball-world +vel_x = screen right, +vel_y = screen up). If clamping a
single vel_x feature on a video the model calls "pushing left to right" flips the
prediction to "pushing right to left", that feature causally controls perceived motion
direction.

Design:
  - ReLU features are one-directional but the clamp is not: s < 0 is well-defined
    (decode is linear in f), so one feature is a two-way knob. Sweep s from -S..+S and
    read the pair log-odds  LO = log P(pos_class) - log P(neg_class)  on real videos of
    BOTH classes. A directional feature gives a monotone LO(s) curve on both input sides.
  - Sign transfer is tested, not assumed: per feature we report (a) NFP t-sign, (b)
    baseline activation preference (mean act on pos-side vs neg-side videos), (c) the
    sign of the induced LO shift — and the agreement between them.
  - Flip rates: pair-restricted (argmax within {pos,neg}) and strict full-174 top-1,
    steering non-preferred-side videos toward the feature's preferred class.
  - Null: random features, same protocol.

Usage (from sae-for-vlm/):
  python analysis/steer_direction_flip.py --per_class 24 --top_feats 8
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
from causal_analysis.common.steer_ssv2_logits import SteerLayer, SSv2Frames, ssv2_collate

PAIRS = [
    {"axis": "vel_x", "pos": "Pushing [something] from left to right",
                      "neg": "Pushing [something] from right to left"},
    {"axis": "vel_x", "pos": "Pulling [something] from left to right",
                      "neg": "Pulling [something] from right to left"},
    {"axis": "vel_y", "pos": "Moving [something] up",
                      "neg": "Moving [something] down"},
]
TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


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
    ap.add_argument("--s_grid", nargs="*", type=float,
                    default=[-150, -100, -40, 0, 40, 100, 150])
    ap.add_argument("--per_class", default=24, type=int)
    ap.add_argument("--top_feats", default=8, type=int, help="candidates per axis by |t|")
    ap.add_argument("--extra_feats", nargs="*", type=int, default=[],
                    help="explicit extra features tested on every pair (e.g. from the "
                         "axis decomposition), regardless of NFP tags")
    ap.add_argument("--skip_nfp", action="store_true",
                    help="test only --extra_feats, not the NFP candidates")
    ap.add_argument("--pair_idx", nargs="*", type=int, default=None,
                    help="subset of PAIRS indices to run (default all)")
    ap.add_argument("--n_random", default=8, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expB_direction_flip.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    label2idx = {v: int(k) for k, v in id2label.items()}

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    # --- candidate features per axis, with NFP sign ---
    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy(); t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = np.where((p_all < bonf).any(1))[0]
    dom = {int(i): TAU[int(np.argmax(np.abs(t_all[i])))] for i in sig}
    cands = {}
    for axis in ["vel_x", "vel_y"]:
        ka = TAU.index(axis)
        feats = [int(i) for i in sig if dom[int(i)] == axis]
        feats.sort(key=lambda i: -abs(t_all[i, ka]))
        cands[axis] = [(i, float(t_all[i, ka])) for i in feats[: args.top_feats]]
        print(f"{axis}: {len(feats)} sig features, testing top {len(cands[axis])}: "
              + ", ".join(f"feat{i:05d}(t={t:+.1f})" for i, t in cands[axis]))
    rng = np.random.RandomState(args.seed)
    rand_feats = rng.choice([k for k in range(sae.dict_size) if k not in set(sig.tolist())],
                            size=args.n_random, replace=False).tolist()

    # --- video caches per pair side ---
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
        return torch.cat(outs, 0).numpy()          # [n, 174]

    s_grid = args.s_grid
    results = {"s_grid": s_grid, "pairs": []}

    run_pairs = PAIRS if args.pair_idx is None else [PAIRS[i] for i in args.pair_idx]
    for pair in run_pairs:
        axis, pos_l, neg_l = pair["axis"], pair["pos"], pair["neg"]
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        n_avail = (len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, [])))
        print(f"\n=== PAIR [{axis}]  {pos_l}  vs  {neg_l}  (avail {n_avail}) ===")
        if cp < 0 or cn < 0 or min(n_avail) < 4:
            print("  skipped (class missing or too few videos)"); continue
        sides = {}
        for side, cls_l in [("pos", pos_l), ("neg", neg_l)]:
            sides[side] = cache_of(by_tmpl[cls_l][: args.per_class])

        # baseline + per-feature activation preference
        steer.enabled = False; steer.record = True; steer.captured = []
        Pb = {side: probs(sides[side]) for side in ["pos", "neg"]}
        A = torch.cat(steer.captured, 0).numpy(); steer.record = False
        n_pos = Pb["pos"].shape[0]
        A_pos, A_neg = A[:n_pos], A[n_pos:]
        for side, cls in [("pos", cp), ("neg", cn)]:
            pr = Pb[side]
            pair_acc = float((pr[:, cp] > pr[:, cn]).mean()) if side == "pos" else \
                       float((pr[:, cn] > pr[:, cp]).mean())
            print(f"  baseline {side}: mean P(own)={pr[:, cp if side=='pos' else cn].mean():.3f} "
                  f"pair-acc={pair_acc:.2f} top1-acc="
                  f"{float((pr.argmax(1) == (cp if side=='pos' else cn)).mean()):.2f}")

        def eval_feature(k, nfp_t=None):
            act_pref = float(A_pos[:, k].mean() - A_neg[:, k].mean())
            curves = {}   # side -> [len(s_grid)] mean pair log-odds LO = logP(pos)-logP(neg)
            flips = {}
            P_by_s = {}
            for side in ["pos", "neg"]:
                lo, Ps = [], []
                for s in s_grid:
                    if s == 0:
                        P = Pb[side]
                    else:
                        steer.enabled = True; steer.k = k; steer.s = s
                        P = probs(sides[side]); steer.enabled = False
                    Ps.append(P)
                    lo.append(float(np.mean(np.log(P[:, cp] + 1e-12)
                                            - np.log(P[:, cn] + 1e-12))))
                curves[side] = lo
                P_by_s[side] = Ps
            # steering-shift sign: dLO/ds via endpoints
            shift = ((curves["pos"][-1] + curves["neg"][-1])
                     - (curves["pos"][0] + curves["neg"][0])) / 2.0
            # flips: steer non-preferred side toward preferred class
            # preferred class by empirical shift: shift>0 -> +s pushes toward pos
            for (src, tgt_cls, tgt_name, s_dir) in [
                    ("neg", cp, "pos", +1 if shift > 0 else -1),
                    ("pos", cn, "neg", -1 if shift > 0 else +1)]:
                # strongest clamp in the pushing direction
                s_idx = (len(s_grid) - 1) if s_dir > 0 else 0
                P = P_by_s[src][s_idx]
                src_cls = cn if src == "neg" else cp
                pair_flip = float((P[:, tgt_cls] > P[:, src_cls]).mean())
                top1_flip = float((P.argmax(1) == tgt_cls).mean())
                base_pair = float((Pb[src][:, tgt_cls] > Pb[src][:, src_cls]).mean())
                flips[f"{src}->{tgt_name}"] = {
                    "s": s_grid[s_idx], "pair_flip": round(pair_flip, 3),
                    "pair_flip_base": round(base_pair, 3),
                    "top1_flip": round(top1_flip, 3)}
            return {"idx": k, "feature": f"feat{k:05d}",
                    "nfp_t": None if nfp_t is None else round(nfp_t, 2),
                    "act_pref": round(act_pref, 4),
                    "lo_pos_side": [round(x, 3) for x in curves["pos"]],
                    "lo_neg_side": [round(x, 3) for x in curves["neg"]],
                    "shift": round(shift, 3), "flips": flips}

        ka = TAU.index(axis)
        rows = []
        test_list = ([] if args.skip_nfp else list(cands[axis])) + \
                    [(k, float(t_all[k, ka])) for k in args.extra_feats]
        for k, t in test_list:
            r = eval_feature(k, t)
            rows.append(r)
            agree_nfp = "Y" if (r["shift"] > 0) == (t > 0) else "n"
            agree_act = "Y" if (r["shift"] > 0) == (r["act_pref"] > 0) else "n"
            f1 = r["flips"]["neg->pos"]; f2 = r["flips"]["pos->neg"]
            print(f"  feat{k:05d} t={t:+6.1f} act_pref={r['act_pref']:+8.3f} "
                  f"shift={r['shift']:+7.2f} sign[nfp:{agree_nfp} act:{agree_act}] "
                  f"flip neg->pos {f1['pair_flip_base']:.2f}->{f1['pair_flip']:.2f} "
                  f"(top1 {f1['top1_flip']:.2f}) | pos->neg {f2['pair_flip_base']:.2f}->"
                  f"{f2['pair_flip']:.2f} (top1 {f2['top1_flip']:.2f})")

        null_rows = []
        for k in rand_feats:
            r = eval_feature(k)
            null_rows.append(r)
        null_shift = np.array([abs(r["shift"]) for r in null_rows]) if null_rows \
            else np.array([0.0])
        if null_rows:
            print(f"  random-null |shift|: mean={null_shift.mean():.2f} "
                  f"sd={null_shift.std():.2f} max={null_shift.max():.2f}")

        results["pairs"].append({
            "axis": axis, "pos": pos_l, "neg": neg_l,
            "n_videos": [int(Pb['pos'].shape[0]), int(Pb['neg'].shape[0])],
            "features": rows, "random": null_rows,
            "null_abs_shift_mean": float(null_shift.mean()),
            "null_abs_shift_sd": float(null_shift.std())})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")

    # cross-pair sign-agreement summary
    n_ag, n_tot = 0, 0
    for pr in results["pairs"]:
        for r in pr["features"]:
            if r["nfp_t"] is not None:
                n_tot += 1
                n_ag += int((r["shift"] > 0) == (r["nfp_t"] > 0))
    if n_tot:
        print(f"\nNFP-sign predicts steering direction: {n_ag}/{n_tot} "
              f"(binomial two-sided vs 0.5)")


if __name__ == "__main__":
    main()
