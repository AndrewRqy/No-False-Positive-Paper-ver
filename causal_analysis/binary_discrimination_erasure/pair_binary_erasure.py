"""Binary-discrimination span-erasure experiment on temporal class pairs.

For each temporally-opposite class pair (A, B), we restrict the model's
pre-softmax logits to the two classes, softmax over just those two, and take
the two-way (binary) accuracy over the pair's validation clips. We then erase
the span of the NFP-identified SAE features (project their decoder columns out
of the layer activations, re-adding the reconstruction error) and measure the
drop in binary accuracy, against three size-matched random feature sets drawn
from all non-identified features. A set of appearance-separable reference
pairs is included to test specificity.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from causal_analysis.common.steer_expansion import (build, forward_logits, cache_frames,
                                       load_flags, FAMILY)  # noqa: E402
from causal_analysis.common.steer_span_erasure import projector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILY.keys()))
    ap.add_argument("--pairs_json", default="analysis/binary_pairs.json")
    ap.add_argument("--ssv2_videos", required=True)
    ap.add_argument("--ssv2_val_json", required=True)
    ap.add_argument("--n_per_class", default=10, type=int)
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--outdir", default="local_runs/expansion")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    fam = args.family
    out = Path(args.outdir)

    clf, proc, steer, sae, cfg = build(fam, device)
    id2label = clf.config.id2label
    label2idx = {v: int(k) for k, v in id2label.items()}
    D, sig, sig_ranked, static_pool, nonsig = load_flags(cfg)
    Wd = sae.decoder.weight.data.cpu()
    val = json.load(open(args.ssv2_val_json))
    print(f"[{fam}] identified={len(sig)} nonsig={len(nonsig)} D={D}")

    pairs = json.load(open(args.pairs_json))
    classes = sorted(set([p["pos_idx"] for p in pairs] + [p["neg_idx"] for p in pairs]))
    cache_by_c = {}
    for c in classes:
        take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][: args.n_per_class]
        cache_by_c[c] = (cache_frames(take, args.ssv2_videos, proc, cfg, cfg["batch"]), len(take))
    nmin = min(v[1] for v in cache_by_c.values())
    print(f"{len(pairs)} pairs over {len(classes)} classes; clips/class {args.n_per_class} (min {nmin})")

    seeds = [int(x) for x in args.seeds.split(",")]
    n = min(len(sig), len(nonsig))
    conds = [("baseline", None), ("identified", sorted(sig))]
    for sd in seeds:
        conds.append((f"rnd{sd}", sorted(np.random.RandomState(sd).choice(nonsig, n, replace=False))))

    # logits[cond][class] = tensor [n_clips, C]
    logits = {name: {} for name, _ in conds}
    for name, ks in conds:
        proj = projector(Wd[:, ks]) if ks is not None else None
        for c in classes:
            caches, _ = cache_by_c[c]
            outs = []
            for inputs in caches:
                steer.proj_out = proj
                outs.append(forward_logits(clf, inputs, device, cfg).cpu())
                steer.proj_out = None
            logits[name][c] = torch.cat(outs, 0) if outs else torch.zeros(0)
        print(f"  logits done: {name}")

    def binary_acc(name, a, b):
        la, lb = logits[name][a], logits[name][b]
        if la.numel() == 0 or lb.numel() == 0:
            return None
        # predict A on A-clips if logit_a > logit_b (softmax over {a,b} is monotone in the diff)
        ca = int((la[:, a] > la[:, b]).sum())
        cb = int((lb[:, b] > lb[:, a]).sum())
        return (ca + cb) / (la.shape[0] + lb.shape[0])

    rows = []
    for p in pairs:
        a, b = p["pos_idx"], p["neg_idx"]
        row = {"key": p["key"], "category": p["category"], "kind": p.get("kind", "temporal")}
        for name, _ in conds:
            row[name] = binary_acc(name, a, b)
        rows.append(row)

    cond_names = [name for name, _ in conds]

    def agg(subset):
        r = [x for x in rows if x in subset]
        d = {}
        for name in cond_names:
            vals = [x[name] for x in r if x[name] is not None]
            d[name] = round(float(np.mean(vals)), 4) if vals else None
        return d, len(r)

    temporal = [x for x in rows if x["kind"] == "temporal"]
    reference = [x for x in rows if x["kind"] == "reference"]
    summary = {"temporal": agg(temporal)[0], "reference": agg(reference)[0]}
    cats = sorted(set(x["category"] for x in temporal))
    summary_cat = {cat: agg([x for x in temporal if x["category"] == cat])[0] for cat in cats}

    result = {"family": fam, "n_identified": len(sig), "conds": cond_names,
              "summary": summary, "summary_by_category": summary_cat, "pairs": rows}
    json.dump(result, open(out / f"{fam}_pair_binary.json", "w"), indent=1)

    print(f"\n{'set':<14}" + "".join(f"{c:>11}" for c in cond_names))
    for label, d in [("temporal", summary["temporal"]), ("reference", summary["reference"])]:
        print(f"{label:<14}" + "".join(f"{(d[c] if d[c] is not None else float('nan')):>11.3f}" for c in cond_names))
    print("-- by category --")
    for cat in cats:
        d = summary_cat[cat]
        print(f"{cat:<14}" + "".join(f"{(d[c] if d[c] is not None else float('nan')):>11.3f}" for c in cond_names))
    print(f"\nsaved -> {fam}_pair_binary.json")


if __name__ == "__main__":
    main()
