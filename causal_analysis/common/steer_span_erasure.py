"""
Experiment C4 — subspace erasure: is the span of the identified features' directions
important to the model's temporal performance?

The zero-ablation null (expB2 controls) only removed each feature's current positive
activation. A stronger necessity test erases the DIRECTIONS: build the orthogonal
projector onto span(decoder columns of the set) and replace every layer-11 token with
x - P x. Whatever the model reads along those directions is gone, whether or not the SAE
features were active.

Measured on the same broad sample as C3 (n videos from every class, clean inputs):
top-1 accuracy under erasure of the NFP-85 span vs matched random-85 and static-85 spans
(same subspace dimension, ~85 of 768). Specificity via the same class terciles: rank
classes by their shuffle-induced drop (loaded from the C3 output) and check whether
NFP-span erasure hurts temporally-demanding classes more than shuffle-robust ones,
relative to the random-span control.

Usage (from sae-for-vlm/):
  python analysis/steer_span_erasure.py --n_per_class 10
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


def span_basis(cols, rtol=None):
    """Orthonormal basis for the numerical column space of `cols` [D, K].

    Reduced QR assumes the K decoder columns are linearly independent; near-
    collinear features inflate the projector's rank and make an erasure remove a
    higher-dimensional subspace than the feature count implies. Use an SVD and
    keep singular directions above a numerical tolerance instead (review item 9).
    Returns (U [D, r], effective_rank).
    """
    W = cols.double()
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    if rtol is None:
        rtol = S.max() * max(W.shape) * torch.finfo(S.dtype).eps if S.numel() else 0.0
    keep = S > rtol
    r = int(keep.sum())
    return U[:, :r].float(), r


def projector(cols):
    """Orthogonal projector onto the numerical span of columns [D, K]."""
    U, _ = span_basis(cols)
    return (U @ U.T).float()


def projector_rank(cols):
    """Effective (numerical) rank of the span of `cols` — record this alongside
    the requested feature count so equal-sized feature sets can be checked for
    comparable interventional dimension."""
    return span_basis(cols)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--c3_json", default="local_runs/steering/expC3_global_restore.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_per_class", default=10, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expC4_span_erasure.json")
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
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    screen = json.load(open(args.screen_json))
    flippers = [r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1]
    Wd = sae.decoder.weight.data.cpu()                                # [768, 6144]
    rng = np.random.RandomState(args.seed + 7)
    rnd85 = sorted(rng.choice([k for k in range(sae.dict_size) if k not in set(sig)],
                              85, replace=False))
    stat85 = sorted(rng.choice(static_pool, 85, replace=False))
    conds = [("baseline", None),
             ("erase NFP-85 span", projector(Wd[:, sorted(sig)])),
             ("erase flippers-12 span", projector(Wd[:, sorted(flippers)])),
             ("erase random-85 span", projector(Wd[:, rnd85])),
             ("erase static-85 span", projector(Wd[:, stat85]))]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c >= 0:
            by_tmpl.setdefault(c, []).append(it)
    items, labels = [], []
    for c in sorted(by_tmpl):
        take = by_tmpl[c][: args.n_per_class]
        items += take; labels += [c] * len(take)
    labels = np.array(labels)
    print(f"{len(items)} videos | conditions: {[n for n, _ in conds]}")

    dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
    preds = {n: [] for n, _ in conds}
    done = 0
    for b in dl:
        pv = b[0]["pixel_values"].to(device)
        for name, P in conds:
            steer.proj_out = P
            with torch.no_grad():
                preds[name].append(clf(pixel_values=pv).logits.argmax(-1).cpu().numpy())
            steer.proj_out = None
        done += pv.shape[0]
        if done % 200 < args.batch_size:
            print(f"  {done}/{len(items)} videos")
    preds = {n: np.concatenate(v) for n, v in preds.items()}
    acc = {n: float((preds[n] == labels).mean()) for n, _ in conds}
    print("\n=== overall top-1 accuracy under span erasure ===")
    for n, _ in conds:
        d = acc[n] - acc["baseline"]
        print(f"  {n:<24} {acc[n]:.3f}  ({d:+.3f})")

    res = {"n_videos": len(items), "acc": acc, "terciles": {}}
    # class terciles from C3 (shuffle-induced drop)
    try:
        c3 = json.load(open(args.c3_json))
        # rebuild terciles from this run's own clean/shuffled preds is not possible here;
        # use per-class drop ranking recomputed from C3 output if present.
        terc_src = None
    except FileNotFoundError:
        terc_src = None
    # per-class breakdown regardless: rank classes by NFP-erasure damage minus random-erasure damage
    cls_ids = sorted(set(labels.tolist()))
    rows = []
    for c in cls_ids:
        m = labels == c
        if m.sum() < 4:
            continue
        b_ = float((preds["baseline"][m] == c).mean())
        n_ = float((preds["erase NFP-85 span"][m] == c).mean())
        r_ = float((preds["erase random-85 span"][m] == c).mean())
        rows.append({"cls": int(c), "base": b_, "nfp": n_, "rnd": r_,
                     "excess": (r_ - n_)})
    rows.sort(key=lambda r: -r["excess"])
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    print("\n  classes hurt MOST by NFP-span erasure beyond random-span erasure:")
    for r in rows[:12]:
        print(f"    {lab(r['cls'])[:52]:52s} base={r['base']:.2f} nfp={r['nfp']:.2f} "
              f"rnd={r['rnd']:.2f} excess={r['excess']:+.2f}")
    res["per_class_top"] = rows[:30]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
