"""
Experiment C3 — global temporal-capability restoration, measured on SSv2 top-1 accuracy
across ALL 174 classes.

Question: do the 85 NFP-identified features influence the model's temporal capability at
the level of task performance, not just on hand-picked class pairs?

Design: take n videos from every SSv2 class. Shuffle each video's 8 tubelet blocks (fixed
per-video permutation). The accuracy lost to shuffling is, by construction, the part of the
model's performance that depends on temporal order. Restore only the chosen features' clean
token activations at layer 11 into the shuffled run and measure how much of that lost
accuracy returns.

  recovery(set) = (acc_restored - acc_shuffled) / (acc_clean - acc_shuffled)

Specificity: rank classes by their own shuffle-induced accuracy drop (from the same data),
split into terciles, and report recovery within the temporally-demanding tercile vs the
shuffle-robust tercile. A temporal feature set should recover accuracy where temporal order
matters and be irrelevant where it does not.

Sets: NFP-85, flippers-12, random-85, static-85, all-6144 (ceiling).

Usage (from sae-for-vlm/):
  python analysis/steer_global_restore.py --n_per_class 10
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
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_per_class", default=10, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expC3_global_restore.json")
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
    rng = np.random.RandomState(args.seed + 7)
    sets = [("NFP-85", torch.tensor(sorted(sig))),
            ("flippers-12", torch.tensor(sorted(flippers))),
            ("random-85", torch.tensor(sorted(rng.choice(
                [k for k in range(sae.dict_size) if k not in set(sig)], 85, replace=False)))),
            ("static-85", torch.tensor(sorted(rng.choice(static_pool, 85, replace=False)))),
            ("ALL-6144", None)]   # None = restore the full feature vector

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
    print(f"{len(items)} videos across {len(by_tmpl)} classes | sets: "
          + ", ".join(n for n, _ in sets))

    dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))

    prng = np.random.RandomState(args.seed + 99)
    conds = ["clean", "shuffled"] + [n for n, _ in sets]
    correct = {c: [] for c in conds}

    def top1(pv, patch=None):
        if patch is not None:
            steer.patch_idx, steer.patch_vals = patch
        with torch.no_grad():
            lg = clf(pixel_values=pv.to(device)).logits
        steer.patch_idx = steer.patch_vals = None
        return lg.argmax(-1).cpu().numpy()

    done = 0
    for b in dl:
        pv = b[0]["pixel_values"]                                    # [B,16,3,224,224]
        B = pv.shape[0]
        # per-video non-identity block permutation, fixed across conditions
        from causal_analysis.common.shuffle_util import block_order, random_block_perm
        pv_sh = pv.clone()
        for r in range(B):
            pv_sh[r] = pv[r, block_order(random_block_perm(prng))]

        # clean pass: predictions + full feature capture (transient, this batch only)
        steer.record_tokens_idx = torch.arange(sae.dict_size)
        steer.captured_tokens = []
        pred = top1(pv)
        steer.record_tokens_idx = None
        f_clean = steer.captured_tokens[0]                           # [B,1568,6144] cpu
        correct["clean"].append(pred)

        correct["shuffled"].append(top1(pv_sh))
        for name, idx in sets:
            if idx is None:
                correct[name].append(top1(pv_sh, patch=(torch.arange(sae.dict_size), f_clean)))
            else:
                correct[name].append(top1(pv_sh, patch=(idx, f_clean[..., idx])))
        del f_clean
        done += B
        if done % 200 < args.batch_size:
            print(f"  {done}/{len(items)} videos")

    preds = {c: np.concatenate(v) for c, v in correct.items()}
    acc = {c: float((preds[c] == labels).mean()) for c in conds}
    print("\n=== overall top-1 accuracy ===")
    print(f"  clean    {acc['clean']:.3f}")
    print(f"  shuffled {acc['shuffled']:.3f}   (destroyed: {acc['clean']-acc['shuffled']:.3f})")
    res = {"n_videos": len(items), "acc": acc, "recovery": {}}
    for name, _ in sets:
        rev = (acc[name] - acc["shuffled"]) / (acc["clean"] - acc["shuffled"] + 1e-9)
        res["recovery"][name] = round(float(rev), 3)
        print(f"  restore {name:<12} {acc[name]:.3f}   recovery = {100*rev:.0f}%")

    # class-level specificity: terciles by per-class shuffle drop
    cls_ids = sorted(by_tmpl)
    drop = {}
    for c in cls_ids:
        m = labels == c
        if m.sum() >= 4:
            drop[c] = float((preds["clean"][m] == c).mean() - (preds["shuffled"][m] == c).mean())
    order = sorted(drop, key=lambda c: -drop[c])
    terc = np.array_split(order, 3)
    print("\n=== recovery by class tercile of shuffle-induced drop ===")
    res["terciles"] = {}
    for ti, tc in enumerate(["temporal (largest drop)", "middle", "shuffle-robust (smallest)"]):
        m = np.isin(labels, terc[ti])
        c_, s_ = float((preds["clean"][m] == labels[m]).mean()), float((preds["shuffled"][m] == labels[m]).mean())
        row = {"clean": round(c_, 3), "shuffled": round(s_, 3)}
        line = f"  {tc:<28} clean={c_:.3f} shuf={s_:.3f}"
        for name, _ in sets:
            r_ = float((preds[name][m] == labels[m]).mean())
            rev = (r_ - s_) / (c_ - s_ + 1e-9)
            row[name] = {"acc": round(r_, 3), "recovery": round(float(rev), 3)}
            if name in ("NFP-85", "flippers-12", "random-85", "ALL-6144"):
                line += f"  {name}:{100*rev:.0f}%"
        res["terciles"][tc] = row
        print(line)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
