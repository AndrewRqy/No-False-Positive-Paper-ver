"""Erasure Size Sweep - necessity dose curve on the family battery.

Erase the decoder span of the top-k flagged features (ranked by max |t| over
taus) for k in --ks, plus a size-matched random span per k (seed 101), and
measure family-class accuracy (same 13-class battery as expE/expD2: classes
with excess flagged-span dependence >= excess_bar in the C4 json).

Usage (from repo root):
  python -m causal_analysis.exp05_erasure_size_sweep.steer_erasure_size_sweep \
      --nfp_results local_runs/nfp_results/sae_nfp_v2.pt \
      --c4_json local_runs/steering/expC4v2_span_erasure.json \
      --out local_runs/steering/expC4v2_size_sweep.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from causal_analysis.common.steer_ssv2_logits import SteerLayer, ssv2_collate
from causal_analysis.common.steer_pair_screen import ItemFrames
from causal_analysis.common.steer_span_erasure import projector


def main() -> None:
    """Run the erasure span-size sweep over --ks and save the dose curve to JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp_v2.pt")
    ap.add_argument("--c4_json", default="local_runs/steering/expC4v2_span_erasure.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument(
        "--ssv2_val_json",
        default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json",
    )
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--excess_bar", default=0.4, type=float)
    ap.add_argument("--n_per_class", default=10, type=int)
    ap.add_argument("--ks", nargs="*", type=int, default=[12, 25, 50, 109])
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expC4v2_size_sweep.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    D = sae.dict_size

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all, t_all = nfp["p_val"].numpy(), nfp["t_stat"].numpy()
    valid = np.isfinite(t_all).all(1)
    sig_mask = (p_all < 0.05 / D).any(1) & valid
    sig = np.where(sig_mask)[0]
    # rank flagged features by max |t| over taus, strongest first
    strength = np.abs(np.nan_to_num(t_all[sig])).max(1)
    sig_ranked = [int(i) for i in sig[np.argsort(-strength)]]
    nonsig = [k for k in range(D) if not sig_mask[k]]
    print(f"flagged: {len(sig_ranked)}; top-5 by |t|: {sig_ranked[:5]}")

    fam = [
        r["cls"]
        for r in json.load(open(args.c4_json))["per_class_top"]
        if r["excess"] >= args.excess_bar
    ]
    print(f"family battery: {len(fam)} classes")

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1)
    vrng.shuffle(val)
    items, labels = [], []
    for c in fam:
        take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][
            : args.n_per_class
        ]
        items += take
        labels += [c] * len(take)
    labels = np.array(labels)
    print(f"videos: {len(items)}")

    dl = DataLoader(
        ItemFrames(args.ssv2_videos, items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ssv2_collate(proc),
    )
    cache = [b[0]["pixel_values"] for b in dl]

    def acc(proj: Optional[torch.Tensor] = None) -> float:
        """Family-battery accuracy, optionally erasing a decoder span first.

        Args:
            proj: Optional projection matrix applied at the steered layer to erase a
                feature span; None runs the unmodified model.

        Returns:
            Top-1 accuracy over the family battery, rounded to three decimals.
        """
        outs = []
        for pv in cache:
            if proj is not None:
                steer.proj_out = proj
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
            steer.proj_out = None
        pred = torch.cat(outs, 0).numpy().argmax(1)
        return round(float((pred == labels).mean()), 3)

    Wd = sae.decoder.weight.data.cpu()
    res = {"baseline": acc(), "ks": {}}
    print(f"baseline family acc = {res['baseline']:.3f}")
    for k in args.ks:
        k = min(k, len(sig_ranked))
        top = sorted(sig_ranked[:k])
        rnd = sorted(np.random.RandomState(101).choice(nonsig, k, replace=False))
        a_top = acc(projector(Wd[:, top]))
        a_rnd = acc(projector(Wd[:, rnd]))
        res["ks"][k] = {"topk": a_top, "random_k": a_rnd}
        print(f"k={k:3d}: erase top-k span acc={a_top:.3f}   random-k span acc={a_rnd:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
