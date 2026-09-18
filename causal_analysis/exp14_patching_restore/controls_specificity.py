"""
Experiment E — specificity battery: do tonight's positive effects hold ONLY for the
identified features, or for any features?

For every positive result (C1 transplant, C2 shuffle-restore, C4 span erasure, D2 error
repair, D7 composition), rerun the intervention with stronger controls:
  - random-85, two fresh seeds (101, 202; disjoint draws from the non-flagged pool);
  - activation-matched-85: for each NFP feature, the non-flagged feature with the closest
    mean activation over the 4,300-video probe cache (greedy nearest, no replacement).
    This controls for the possibility that effects come from patching ACTIVE features
    rather than TEMPORAL ones;
  - for D7: two random feature pairs steered jointly, and feat01321 + one random feature.

Metrics mirror the original experiments; each part prints NFP's original number next to
the new controls.

Replication: video selections identical to the original experiments (same seeds).
Usage (from sae-for-vlm/):
  python analysis/controls_specificity.py
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
from causal_analysis.exp14_patching_restore.steer_shuffle_restore import shuffle_blocks
from causal_analysis.common.steer_span_erasure import projector

PAIRS7 = [
    ("push_lr", "Pushing [something] from left to right", "Pushing [something] from right to left"),
    ("pull_lr", "Pulling [something] from left to right", "Pulling [something] from right to left"),
    ("cam_lr",  "Turning the camera left while filming [something]",
                "Turning the camera right while filming [something]"),
    ("move_ud", "Moving [something] up", "Moving [something] down"),
    ("cam_ud",  "Turning the camera upwards while filming [something]",
                "Turning the camera downwards while filming [something]"),
    ("fall_speed", "[Something] falling like a rock", "[Something] falling like a feather or paper"),
    ("spin_stop", "Spinning [something] so it continues spinning",
                  "Spinning [something] that quickly stops spinning"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--probe_cache", default="local_runs/steering/expD4_probe_cache.pt")
    ap.add_argument("--c4_json", default="local_runs/steering/expC4_span_erasure.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parts", nargs="*", default=["c1", "c2", "c4", "d2", "d7"])
    ap.add_argument("--out", default="local_runs/steering/expE_specificity.json")
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
    D = sae.dict_size

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy()
    sig = sorted(int(i) for i in np.where((p_all < 0.05 / p_all.shape[0]).any(1))[0])
    nonsig = [k for k in range(D) if k not in set(sig)]

    # control sets
    r1 = sorted(np.random.RandomState(101).choice(nonsig, len(sig), replace=False))
    r2 = sorted(np.random.RandomState(202).choice(nonsig, len(sig), replace=False))
    cache_d4 = torch.load(args.probe_cache, map_location="cpu")
    mean_act = cache_d4["feats"].numpy().mean(0)                     # [6144]
    taken, match = set(), []
    order_pool = sorted(nonsig, key=lambda k: mean_act[k])
    pool_acts = np.array([mean_act[k] for k in order_pool])
    for k in sig:
        j = int(np.argmin(np.abs(pool_acts - mean_act[k]) + 1e9 * np.isin(
            np.arange(len(order_pool)), list(taken))))
        taken.add(j); match.append(order_pool[j])
    match = sorted(match)
    print(f"activation matching: NFP mean act {np.mean(mean_act[sig]):.4f} vs "
          f"matched pool {np.mean(mean_act[match]):.4f} (random {np.mean(mean_act[r1]):.4f})")
    SETS = [("NFP-85", sig), ("rnd85-s101", r1), ("rnd85-s202", r2), ("actmatch-85", match)]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def probs(cache, patch=None, proj=None):
        outs = []
        for bi, pv in enumerate(cache):
            if patch is not None:
                steer.patch_idx = patch[0]; steer.patch_vals = patch[1][bi]
            if proj is not None:
                steer.proj_out = proj
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
            steer.patch_idx = steer.patch_vals = None; steer.proj_out = None
        return torch.cat(outs, 0).numpy()

    def capture_full(cache):
        steer.record_tokens_idx = torch.arange(D); steer.captured_tokens = []
        for pv in cache:
            with torch.no_grad():
                clf(pixel_values=pv.to(device))
        steer.record_tokens_idx = None
        vals, i = [], 0
        allv = torch.cat(steer.captured_tokens, 0)
        for pv in cache:
            vals.append(allv[i:i + pv.shape[0]]); i += pv.shape[0]
        return vals

    res = {}

    # ---------------- C1 transplant, camera pairs ----------------
    if "c1" in args.parts:
        print("\n=== E/C1: transplant specificity (camera pairs) ===")
        res["c1"] = {}
        for key, pos_l, neg_l in [PAIRS7[2], PAIRS7[4]]:
            cp, cn = label2idx[pos_l], label2idx[neg_l]
            caches = {"pos": cache_of(by_tmpl[pos_l][: args.per_class]),
                      "neg": cache_of(by_tmpl[neg_l][: args.per_class])}
            Pb = {s: probs(caches[s]) for s in ["pos", "neg"]}
            donor_full = {s: capture_full(caches[s]) for s in ["pos", "neg"]}
            row = {}
            for name, ks in SETS:
                idx = torch.tensor(ks)
                dls = []
                for recv, donor, r_cls, d_cls in [("pos", "neg", cp, cn), ("neg", "pos", cn, cp)]:
                    vals = [f[..., idx] for f in donor_full[donor]]
                    P = probs(caches[recv], patch=(idx, vals))
                    b = Pb[recv]
                    dls.append(float(np.mean(
                        (np.log(P[:, d_cls] + 1e-12) - np.log(P[:, r_cls] + 1e-12))
                        - (np.log(b[:, d_cls] + 1e-12) - np.log(b[:, r_cls] + 1e-12)))))
                row[name] = round(float(np.mean(dls)), 3)
            res["c1"][key] = row
            print(f"  {key:<8} dLO->donor: " + "  ".join(f"{n}={v:+.2f}" for n, v in row.items()))

    # ---------------- C2 shuffle-restore ----------------
    if "c2" in args.parts:
        print("\n=== E/C2: shuffle-restore specificity (7 pairs) ===")
        prng = np.random.RandomState(args.seed + 99)
        agg = {"clean": [], "shuffled": [], **{n: [] for n, _ in SETS}}
        for key, pos_l, neg_l in PAIRS7:
            cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
            if cp < 0 or cn < 0:
                continue
            for side, cls_l, own, other in [("pos", pos_l, cp, cn), ("neg", neg_l, cn, cp)]:
                cache = cache_of(by_tmpl[cls_l][: args.per_class])
                perm = list(prng.permutation(8))
                while perm == list(range(8)):
                    perm = list(prng.permutation(8))
                cache_sh = [shuffle_blocks(pv, perm) for pv in cache]
                P_c, P_s = probs(cache), probs(cache_sh)

                def lo(P):
                    return float(np.mean(np.log(P[:, own] + 1e-12) - np.log(P[:, other] + 1e-12)))
                agg["clean"].append(lo(P_c)); agg["shuffled"].append(lo(P_s))
                full = capture_full(cache)
                for name, ks in SETS:
                    idx = torch.tensor(ks)
                    vals = [f[..., idx] for f in full]
                    agg[name].append(lo(probs(cache_sh, patch=(idx, vals))))
        c, s = np.mean(agg["clean"]), np.mean(agg["shuffled"])
        res["c2"] = {"clean": round(float(c), 3), "shuffled": round(float(s), 3)}
        print(f"  clean {c:+.3f}  shuffled {s:+.3f}")
        for name, _ in SETS:
            rec = (np.mean(agg[name]) - s) / (c - s + 1e-9)
            res["c2"][name] = round(float(rec), 3)
            print(f"  restore {name:<12} recovery = {100*rec:.0f}%")

    # ---------------- C4 span erasure, family classes ----------------
    if "c4" in args.parts:
        print("\n=== E/C4: span-erasure specificity (11 family classes) ===")
        fam = [r["cls"] for r in json.load(open(args.c4_json))["per_class_top"]
               if r["excess"] >= 0.4]
        Wd = sae.decoder.weight.data.cpu()
        items, labels = [], []
        for c in fam:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][:10]
            items += take; labels += [c] * len(take)
        labels = np.array(labels)
        cache = cache_of(items)
        row = {}
        base = probs(cache).argmax(1)
        row["baseline"] = round(float((base == labels).mean()), 3)
        for name, ks in SETS:
            P = projector(Wd[:, ks])
            pred = probs(cache, proj=P).argmax(1)
            row[name] = round(float((pred == labels).mean()), 3)
        res["c4"] = row
        print("  family-class acc: " + "  ".join(f"{n}={v:.3f}" for n, v in row.items()))

    # ---------------- D2 error repair at alpha=3 ----------------
    if "d2" in args.parts:
        print("\n=== E/D2: repair specificity (alpha=3) ===")
        fam = [r["cls"] for r in json.load(open(args.c4_json))["per_class_top"]
               if r["excess"] >= 0.4]
        wrong_items, wrong_lbls = [], []
        for c in fam:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][:24]
            if not take:
                continue
            cache = cache_of(take)
            pred = probs(cache).argmax(1)
            for i, it in enumerate(take):
                if pred[i] != c:
                    wrong_items.append(it); wrong_lbls.append(c)
        wl = np.array(wrong_lbls); wcache = cache_of(wrong_items)
        print(f"  {len(wrong_items)} errors mined")
        row = {}
        for name, ks in SETS:
            idx = torch.tensor(ks)
            full = capture_full(wcache)
            vals = [3.0 * f[..., idx] for f in full]
            pred = probs(wcache, patch=(idx, vals)).argmax(1)
            row[name] = round(float((pred == wl).mean()), 3)
        res["d2"] = row
        print("  repair rate: " + "  ".join(f"{n}={v:.3f}" for n, v in row.items()))

    # ---------------- D7 composition control ----------------
    if "d7" in args.parts:
        print("\n=== E/D7: composition specificity (s=100, neutral videos) ===")
        CLS = {"right": "Turning the camera right while filming [something]",
               "down":  "Turning the camera downwards while filming [something]"}
        cids = {k: label2idx[v] for k, v in CLS.items()}
        cam_labels = {v for v in label2idx if v.startswith("Turning the camera")}
        items = [it for it in val if it.get("template", "") not in cam_labels][:24]
        cache = cache_of(items)
        rng = np.random.RandomState(303)
        pairs_ctl = [("rnd-pair-1", [int(x) for x in rng.choice(nonsig, 2, replace=False)]),
                     ("rnd-pair-2", [int(x) for x in rng.choice(nonsig, 2, replace=False)]),
                     ("real+rnd", [1321, int(rng.choice(nonsig))]),
                     ("real-pair (orig)", [1321, 4665])]
        row = {}
        for name, ks in pairs_ctl:
            steer.enabled = True; steer.k = ks; steer.s = 100.0
            p = probs(cache).mean(0); steer.enabled = False
            row[name] = {k: round(float(p[c]), 4) for k, c in cids.items()}
            print(f"  {name:<18} P(right)={row[name]['right']:.4f} P(down)={row[name]['down']:.4f}")
        res["d7"] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
