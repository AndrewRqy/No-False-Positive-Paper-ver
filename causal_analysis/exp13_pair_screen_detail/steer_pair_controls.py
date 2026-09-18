"""
Experiment B2 controls — (1) random static features on the identical pair screen,
(2) ablation: turn temporal features OFF and measure what breaks.

Part 1 (sufficiency comparison): sample N random STATIC features — non-significant in NFP,
finite t, max|t| over taus < 2 (measurably non-temporal on the probe) — and run them through
the exact expB2 protocol (all 10 pairs, clamp +/-S both sides). Compare per-feature
distributions vs the 85 NFP temporal features: mean |LO delta| across pairs, max best_flip,
count reaching >=50% flips. Mann-Whitney one-sided.

Part 2 (necessity / ablation): clamp SETS of features to 0 simultaneously (decode + re-add
error, so only those features' contributions are removed) on the same pair videos:
  - all 85 temporal features
  - the pair-screen hero subset (n_pairs_flip50 >= 1 from expB2)
  - random static sets of matched sizes (2 seeds each)
Measure per pair: pair accuracy (both sides), mean |pair log-odds| (discrimination margin),
and strict top-1-own-class accuracy. If NFP features carry the model's temporal-pair
discrimination, ablating them should hurt temporal pairs more than ablating matched random
sets.

Usage (from sae-for-vlm/):
  python analysis/steer_pair_controls.py --n_static 30 --s_abs 150
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from causal_analysis.common.steer_ssv2_logits import SteerLayer, SSv2Frames, ssv2_collate
from causal_analysis.common.steer_pair_screen import PAIRS, ItemFrames

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--s_abs", default=150.0, type=float)
    ap.add_argument("--n_static", default=30, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expB2_controls.json")
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
    # random STATIC pool: finite t, max|t| < bar, not significant
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    rng = np.random.RandomState(args.seed)
    static_feats = rng.choice(static_pool, size=args.n_static, replace=False).tolist()
    print(f"{len(sig)} temporal | static pool (finite t, max|t|<{args.static_t_bar}): "
          f"{len(static_pool)} -> sampled {len(static_feats)}")

    screen = json.load(open(args.screen_json))
    heroes = [r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1]
    print(f"pair-screen heroes (flip50>=1): {len(heroes)}: "
          + ", ".join(f"feat{k:05d}" for k in heroes))

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

    pairs = []
    for key, axis, pos_l, neg_l in PAIRS:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        if cp < 0 or cn < 0 or min(len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, []))) < args.per_class:
            continue
        caches = {"pos": cache_of(by_tmpl[pos_l][: args.per_class]),
                  "neg": cache_of(by_tmpl[neg_l][: args.per_class])}
        steer.enabled = False
        Pb = {s: probs(caches[s]) for s in ["pos", "neg"]}
        pairs.append({"key": key, "axis": axis, "cp": cp, "cn": cn,
                      "caches": caches, "Pb": Pb})
    print(f"{len(pairs)} pairs cached")
    S = args.s_abs

    # ---------------- Part 1: random static features, identical screen ----------------
    def screen_feature(k):
        rec = {}
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
            def flip(orient):
                s_pos, s_neg = (+S, -S) if orient == "A" else (-S, +S)
                f_np = float((P[("neg", s_pos)][:, cp] > P[("neg", s_pos)][:, cn]).mean())
                f_pn = float((P[("pos", s_neg)][:, cn] > P[("pos", s_neg)][:, cp]).mean())
                return (f_np + f_pn) / 2
            rec[pr["key"]] = {"delta": round(delta, 3),
                              "best_flip": round(max(flip("A"), flip("B")), 3)}
        return rec

    print("\n=== Part 1: random static features on the identical screen ===")
    static_rows = []
    for i, k in enumerate(static_feats):
        rec = screen_feature(k)
        deltas = [abs(v["delta"]) for v in rec.values()]
        flips = [v["best_flip"] for v in rec.values()]
        static_rows.append({"feature": f"feat{k:05d}", "idx": k, "pairs": rec,
                            "mean_abs_delta": round(float(np.mean(deltas)), 3),
                            "max_flip": round(float(np.max(flips)), 3),
                            "n_pairs_flip50": int(sum(f >= 0.5 for f in flips))})
        if (i + 1) % 10 == 0:
            print(f"  screened {i+1}/{len(static_feats)} static features")

    tmp_delta = [r["mean_abs_delta"] for r in screen["features"]]
    tmp_maxflip = [max(v["best_flip"] for v in r["pairs"].values()) for r in screen["features"]]
    tmp_f50 = sum(1 for r in screen["features"] if r["n_pairs_flip50"] >= 1)
    st_delta = [r["mean_abs_delta"] for r in static_rows]
    st_maxflip = [r["max_flip"] for r in static_rows]
    st_f50 = sum(1 for r in static_rows if r["n_pairs_flip50"] >= 1)
    mw_d = stats.mannwhitneyu(tmp_delta, st_delta, alternative="greater").pvalue
    mw_f = stats.mannwhitneyu(tmp_maxflip, st_maxflip, alternative="greater").pvalue
    print(f"\n  mean|delta| across pairs:  temporal {np.mean(tmp_delta):.3f} vs "
          f"static {np.mean(st_delta):.3f}   MW p={mw_d:.2e}")
    print(f"  max best_flip per feature: temporal {np.mean(tmp_maxflip):.3f} vs "
          f"static {np.mean(st_maxflip):.3f}   MW p={mw_f:.2e}")
    print(f"  features with >=50% flip on >=1 pair: temporal {tmp_f50}/{len(tmp_delta)} "
          f"vs static {st_f50}/{len(st_delta)}")

    # ---------------- Part 2: ablation (clamp sets to 0) ----------------
    print("\n=== Part 2: ablation — clamp feature sets to 0 ===")
    rng2 = np.random.RandomState(args.seed + 7)
    ablation_sets = [("none (baseline)", None),
                     ("all 85 temporal", sig),
                     (f"heroes ({len(heroes)})", heroes)]
    for si in range(2):
        ablation_sets.append((f"random static x85 (seed{si})",
                              rng2.choice(static_pool, size=len(sig), replace=False).tolist()))
        ablation_sets.append((f"random static x{len(heroes)} (seed{si})",
                              rng2.choice(static_pool, size=len(heroes), replace=False).tolist()))

    def eval_ablation(feat_set):
        per_pair = {}
        for pr in pairs:
            cp, cn = pr["cp"], pr["cn"]
            accs, lom, top1 = [], [], []
            for side, own, other in [("pos", cp, cn), ("neg", cn, cp)]:
                if feat_set is None:
                    P = pr["Pb"][side]
                else:
                    steer.enabled = True; steer.k = feat_set; steer.s = 0.0
                    P = probs(pr["caches"][side]); steer.enabled = False
                accs.append(float((P[:, own] > P[:, other]).mean()))
                lom.append(float(np.abs(np.log(P[:, cp] + 1e-12)
                                        - np.log(P[:, cn] + 1e-12)).mean()))
                top1.append(float((P.argmax(1) == own).mean()))
            per_pair[pr["key"]] = {"pair_acc": round(float(np.mean(accs)), 3),
                                   "abs_lo": round(float(np.mean(lom)), 3),
                                   "top1": round(float(np.mean(top1)), 3)}
        return per_pair

    abl_results = []
    for name, fs in ablation_sets:
        pp = eval_ablation(fs)
        mean_acc = float(np.mean([v["pair_acc"] for v in pp.values()]))
        mean_lo = float(np.mean([v["abs_lo"] for v in pp.values()]))
        mean_t1 = float(np.mean([v["top1"] for v in pp.values()]))
        abl_results.append({"set": name, "n": 0 if fs is None else len(fs),
                            "per_pair": pp, "mean_pair_acc": round(mean_acc, 3),
                            "mean_abs_lo": round(mean_lo, 3), "mean_top1": round(mean_t1, 3)})
        print(f"  {name:<28} pair-acc={mean_acc:.3f}  |LO|={mean_lo:.3f}  top1={mean_t1:.3f}")

    print("\n  per-pair pair-acc (baseline vs all-85-temporal-off):")
    base_pp = abl_results[0]["per_pair"]; t85_pp = abl_results[1]["per_pair"]
    for k in base_pp:
        d = t85_pp[k]["pair_acc"] - base_pp[k]["pair_acc"]
        print(f"    {k:<11} {base_pp[k]['pair_acc']:.2f} -> {t85_pp[k]['pair_acc']:.2f} ({d:+.2f})"
              f"   |LO| {base_pp[k]['abs_lo']:.2f} -> {t85_pp[k]['abs_lo']:.2f}")

    out = {"s_abs": S, "static_feats": static_feats, "static_rows": static_rows,
           "population": {"temporal_mean_delta": float(np.mean(tmp_delta)),
                          "static_mean_delta": float(np.mean(st_delta)),
                          "mw_p_delta": float(mw_d),
                          "temporal_mean_maxflip": float(np.mean(tmp_maxflip)),
                          "static_mean_maxflip": float(np.mean(st_maxflip)),
                          "mw_p_maxflip": float(mw_f),
                          "temporal_f50": int(tmp_f50), "static_f50": int(st_f50)},
           "ablation": abl_results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
