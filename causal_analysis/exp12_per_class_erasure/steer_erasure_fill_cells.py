"""
Fill the missing erasure-table cells:
  - family battery (13 classes): static-N span
  - all-class battery (174 classes x 10): random seed-202 span, activation-matched span

Control sets sized to the flagged count. Same batteries as expEv2/expC4v2.

Usage (from sae-for-vlm/):
  python analysis/steer_erasure_fill_cells.py
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
from causal_analysis.common.steer_span_erasure import projector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp_v2.pt")
    ap.add_argument("--c4_json", default="local_runs/steering/expC4v2_span_erasure.json")
    ap.add_argument("--probe_cache", default="local_runs/steering/expD4_probe_cache.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--excess_bar", default=0.4, type=float)
    ap.add_argument("--n_per_class", default=10, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expC4v2_fill_cells.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    D = sae.dict_size

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all, t_all = nfp["p_val"].numpy(), nfp["t_stat"].numpy()
    valid = np.isfinite(t_all).all(1)
    sig_mask = (p_all < 0.05 / D).any(1) & valid
    sig = sorted(int(i) for i in np.where(sig_mask)[0])
    nonsig = [k for k in range(D) if not sig_mask[k]]
    n = len(sig)

    # static-N (finite t, all |t| < bar)
    low_t = valid & (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(low_t)[0] if not sig_mask[i]]
    rng = np.random.RandomState(args.seed + 7)
    static_set = sorted(rng.choice(static_pool, n, replace=False))
    # random seed 202
    r2 = sorted(np.random.RandomState(202).choice(nonsig, n, replace=False))
    # activation-matched
    cache_d4 = torch.load(args.probe_cache, map_location="cpu")
    mean_act = cache_d4["feats"].numpy().mean(0)
    taken, match = set(), []
    order_pool = sorted(nonsig, key=lambda k: mean_act[k])
    pool_acts = np.array([mean_act[k] for k in order_pool])
    for k in sig:
        j = int(np.argmin(np.abs(pool_acts - mean_act[k]) + 1e9 * np.isin(
            np.arange(len(order_pool)), list(taken))))
        taken.add(j); match.append(order_pool[j])
    match = sorted(match)
    print(f"flagged {n}; static pool {len(static_pool)}")

    fam = [r["cls"] for r in json.load(open(args.c4_json))["per_class_top"]
           if r["excess"] >= args.excess_bar]
    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)

    Wd = sae.decoder.weight.data.cpu()

    def battery(classes, tag, sets):
        items, labels = [], []
        for c in classes:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][:args.n_per_class]
            items += take; labels += [c] * len(take)
        # stream per item with skip guard (some val videos decode ragged);
        # forward every set on each decoded batch so nothing is held in RAM
        ds = ItemFrames(args.ssv2_videos, items)
        coll = ssv2_collate(proc)
        projs = [(name, projector(Wd[:, ks])) for name, ks in sets]
        preds = {name: [] for name, _ in projs}
        kept_labels = []
        buf = []
        n_skip = 0

        def flush():
            if not buf:
                return
            pv = torch.cat(buf, 0).to(device)
            for name, P in projs:
                steer.proj_out = P
                with torch.no_grad():
                    preds[name].append(clf(pixel_values=pv).logits.argmax(-1).cpu().numpy())
                steer.proj_out = None
            buf.clear()

        for i in range(len(items)):
            try:
                buf.append(coll([ds[i]])[0]["pixel_values"])
                kept_labels.append(labels[i])
            except Exception as e:
                n_skip += 1
                print(f"  {tag}: skipping video {items[i].get('id','?')} ({type(e).__name__})")
                continue
            if len(buf) >= args.batch_size:
                flush()
            if (i + 1) % 300 == 0:
                print(f"  {tag}: {i+1}/{len(items)}")
        flush()
        kept_labels = np.array(kept_labels)
        print(f"  {tag}: {len(kept_labels)}/{len(items)} videos usable")
        out = {}
        for name, _ in projs:
            pred = np.concatenate(preds[name])
            out[name] = round(float((pred == kept_labels).mean()), 3)
            print(f"  {tag} {name}: {out[name]:.3f}")
        return out

    res = {}
    res["family"] = battery(fam, "family", [("static", static_set)])
    all_classes = sorted(label2idx.values())
    res["all_class"] = battery(all_classes, "all-class",
                               [("rnd_s202", r2), ("actmatch", match)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
