"""
Repair-subsection additions for the paper (VideoMAE, v2 flags):

1. Per-class net accuracy under identified-set amplification across the alpha
   grid: per-class table (baseline vs steered), macro-average, best and worst
   class deltas. Mining protocol identical to steer_error_repair.py (same
   family classes from expC4v2 excess >= 0.4, same seed+1 shuffle, same
   mine_per_class cap).
2. Random-class-selection ablation: the same amplification applied to 13
   classes drawn at random from OUTSIDE the family (seeded), to show the gain
   is specific to the feature-dependent classes.

Usage (from sae-for-vlm/):
  python analysis/steer_repair_addons.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp_v2.pt")
    ap.add_argument("--c4_json", default="local_runs/steering/expC4v2_span_erasure.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--excess_bar", default=0.4, type=float)
    ap.add_argument("--mine_per_class", default=24, type=int)
    ap.add_argument("--alphas", nargs="*", type=float, default=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument("--n_random_classes", default=13, type=int)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD2v2_repair_addons.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    label2idx = {v: int(k) for k, v in id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    D = sae.dict_size

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all, t_all = nfp["p_val"].numpy(), nfp["t_stat"].numpy()
    valid = np.isfinite(t_all).all(1)
    sig = sorted(int(i) for i in np.where((p_all < 0.05 / D).any(1) & valid)[0])
    idx_sig = torch.tensor(sig)
    print(f"identified features: {len(sig)}")

    fam = [r["cls"] for r in json.load(open(args.c4_json))["per_class_top"]
           if r["excess"] >= args.excess_bar]
    print(f"family classes: {len(fam)}")

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_cls = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c >= 0:
            by_cls.setdefault(c, []).append(it)

    def run(cache, patch_sets=None):
        preds = []
        for pv in cache:
            pv = pv.to(device)
            if patch_sets is not None:
                idx, alpha = patch_sets
                steer.record_tokens_idx = idx; steer.captured_tokens = []
                with torch.no_grad():
                    clf(pixel_values=pv)
                steer.record_tokens_idx = None
                fvals = steer.captured_tokens[0]
                steer.patch_idx = idx; steer.patch_vals = alpha * fvals
            with torch.no_grad():
                preds.append(clf(pixel_values=pv).logits.argmax(-1).cpu().numpy())
            steer.patch_idx = steer.patch_vals = None
        return np.concatenate(preds)

    def evaluate(classes, tag):
        items, labels = [], []
        for c in classes:
            take = by_cls.get(c, [])[: args.mine_per_class]
            items += take; labels += [c] * len(take)
        labels = np.array(labels)
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        cache = [b[0]["pixel_values"] for b in dl]
        base = run(cache)
        base_acc = float((base == labels).mean())
        print(f"[{tag}] n={len(labels)} baseline={base_acc:.3f}")
        out = {"n": len(labels), "baseline": round(base_acc, 3), "alphas": {}}
        per_class_base = {int(c): float((base[labels == c] == c).mean()) for c in classes}
        out["per_class_baseline"] = {str(c): round(v, 3) for c, v in per_class_base.items()}
        for a in args.alphas:
            pred = run(cache, patch_sets=(idx_sig, a))
            net = float((pred == labels).mean())
            pcls = {int(c): float((pred[labels == c] == c).mean()) for c in classes}
            macro = float(np.mean(list(pcls.values())))
            deltas = {c: pcls[c] - per_class_base[c] for c in pcls}
            best_c = max(deltas, key=deltas.get); worst_c = min(deltas, key=deltas.get)
            out["alphas"][f"{a:g}"] = {
                "net_acc": round(net, 3), "delta": round(net - base_acc, 3),
                "macro_acc": round(macro, 3),
                "per_class": {str(c): round(v, 3) for c, v in pcls.items()},
                "best_class": {"cls": best_c, "label": id2label[best_c],
                               "base": round(per_class_base[best_c], 3),
                               "steered": round(pcls[best_c], 3)},
                "worst_class": {"cls": worst_c, "label": id2label[worst_c],
                                "base": round(per_class_base[worst_c], 3),
                                "steered": round(pcls[worst_c], 3)}}
            print(f"  a={a:<4g} net={net:.3f} ({net-base_acc:+.3f}) macro={macro:.3f} "
                  f"best {id2label[best_c][:28]} {per_class_base[best_c]:.2f}->{pcls[best_c]:.2f} "
                  f"worst {id2label[worst_c][:28]} {per_class_base[worst_c]:.2f}->{pcls[worst_c]:.2f}")
        return out

    res = {"family": evaluate(fam, "family (13 feature-dependent classes)")}

    pool = [c for c in sorted(by_cls) if c not in set(fam)]
    rnd_classes = sorted(np.random.RandomState(args.seed + 11).choice(
        pool, args.n_random_classes, replace=False).tolist())
    print("random classes:", [id2label[c][:40] for c in rnd_classes])
    res["random_classes"] = evaluate(rnd_classes, "random 13 classes (ablation)")
    res["random_class_ids"] = rnd_classes

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
