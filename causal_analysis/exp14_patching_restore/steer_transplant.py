"""Feature transplant (Experiment C1) - interchange intervention between direction-paired videos.

Every steering experiment so far clamps features to extreme values, which invites the
attractor objection (a hard clamp saturates toward a fixed output). Activation patching
avoids it: swap the NFP features' NATURAL activations between two real videos and see if
the prediction follows. Take a receiver video of class A and a donor video of class B
(same content type, opposite motion). Replace only the chosen features' token-level
activations in the receiver with the donor's (decode + re-add error, so nothing else
changes). If the prediction moves to B, the transplanted features carry the motion percept.

Sets: the 85 NFP temporal features; the 12 pair-screen flippers; matched random and
static-pool sets; all 6144 features (ceiling; = donor's full layer-11 SAE code with the
receiver's reconstruction error).

Metrics per pair and direction: mean pair log-odds shift toward the donor class,
pair-restricted flip rate, strict top-1-to-donor rate. Both transplant directions.

Usage (from repo root):
  python -m causal_analysis.exp14_patching_restore.steer_transplant --per_class 12
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from causal_analysis.common.steer_ssv2_logits import SteerLayer, ssv2_collate
from causal_analysis.common.steer_pair_screen import PAIRS, ItemFrames


def main() -> None:
    """Run the C1 feature-transplant experiment and write per-pair and summary metrics.

    Loads the SSv2 classifier and layer-11 SAE, builds the feature sets (NFP temporal,
    pair-screen flippers, matched random and static pools, and the all-6144 ceiling),
    then for each temporal class pair transplants the donor's natural feature activations
    into the receiver in both directions. Reports the mean log-odds shift toward the donor
    class, the pair-restricted flip rate, and the strict top-1-to-donor rate, and dumps
    the results as JSON.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument(
        "--ssv2_val_json",
        default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json",
    )
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expC1_transplant.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy()
    t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = [int(i) for i in np.where((p_all < bonf).any(1))[0]]
    finite = np.isfinite(t_all).all(1)
    low_t = np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    screen = json.load(open(args.screen_json))
    flippers = [r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1]
    rng = np.random.RandomState(args.seed + 7)
    sets = [
        ("NFP temporal x85", sig),
        (f"flippers x{len(flippers)}", flippers),
        (
            "random x85 (s0)",
            rng.choice(
                [k for k in range(sae.dict_size) if k not in set(sig)], 85, replace=False
            ).tolist(),
        ),
        ("static x85", rng.choice(static_pool, 85, replace=False).tolist()),
        ("ALL 6144 (ceiling)", list(range(sae.dict_size))),
    ]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1)
    vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Batch a list of SSv2 items into cached pixel-value tensors.

        Args:
            items: SSv2 validation records to load and preprocess.

        Returns:
            One pixel-value tensor per batch, in dataloader order.
        """
        dl = DataLoader(
            ItemFrames(args.ssv2_videos, items),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=ssv2_collate(proc),
        )
        return [b[0]["pixel_values"] for b in dl]

    def probs(
        cache: List[torch.Tensor],
        patch: Optional[Tuple[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> np.ndarray:
        """Run the classifier over cached batches and return softmax probabilities.

        Args:
            cache: Per-batch pixel-value tensors.
            patch: Optional (feature-index tensor, per-batch donor values) installed at
                the steered layer for each batch, or None for an unpatched pass.

        Returns:
            A [n_videos, n_classes] array of class probabilities.
        """
        # patch: (idx_tensor, [vals per batch]) or None
        outs = []
        for bi, pv in enumerate(cache):
            if patch is not None:
                steer.patch_idx = patch[0]
                steer.patch_vals = patch[1][bi]
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
            steer.patch_idx = None
            steer.patch_vals = None
        return torch.cat(outs, 0).numpy()

    def capture(cache: List[torch.Tensor], idx: torch.Tensor) -> List[torch.Tensor]:
        """Capture donor token activations for the chosen features, per batch.

        Args:
            cache: Per-batch pixel-value tensors to run through the model.
            idx: Feature indices to record at the steered layer.

        Returns:
            Per-batch tensors of recorded token activations, aligned to `cache`.
        """
        steer.record_tokens_idx = idx
        steer.captured_tokens = []
        for pv in cache:
            with torch.no_grad():
                clf(pixel_values=pv.to(device))
        steer.record_tokens_idx = None
        # keep per-batch structure for patching
        vals, i = [], 0
        for pv in cache:
            vals.append(torch.cat(steer.captured_tokens, 0)[i : i + pv.shape[0]])
            i += pv.shape[0]
        return vals

    temporal_pairs = [p for p in PAIRS if p[1] != "depth"]
    results = {"per_class": args.per_class, "pairs": []}
    for key, axis_tag, pos_l, neg_l in temporal_pairs:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        if (
            cp < 0
            or cn < 0
            or min(len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, []))) < args.per_class
        ):
            continue
        caches = {
            "pos": cache_of(by_tmpl[pos_l][: args.per_class]),
            "neg": cache_of(by_tmpl[neg_l][: args.per_class]),
        }
        steer.enabled = False
        Pb = {s: probs(caches[s]) for s in ["pos", "neg"]}
        print(
            f"\n=== PAIR {key} [{axis_tag}]  base pair-acc "
            f"pos={float((Pb['pos'][:, cp] > Pb['pos'][:, cn]).mean()):.2f} "
            f"neg={float((Pb['neg'][:, cn] > Pb['neg'][:, cp]).mean()):.2f} ==="
        )
        rec = {"key": key, "axis": axis_tag, "sets": {}}
        for name, ks in sets:
            idx = torch.tensor(sorted(set(ks)))
            r = {}
            for recv, donor, r_cls, d_cls in [("pos", "neg", cp, cn), ("neg", "pos", cn, cp)]:
                dvals = capture(caches[donor], idx)
                P = probs(caches[recv], patch=(idx, dvals))
                b = Pb[recv]
                dlo = float(
                    np.mean(
                        (np.log(P[:, d_cls] + 1e-12) - np.log(P[:, r_cls] + 1e-12))
                        - (np.log(b[:, d_cls] + 1e-12) - np.log(b[:, r_cls] + 1e-12))
                    )
                )
                flip = float((P[:, d_cls] > P[:, r_cls]).mean())
                base_flip = float((b[:, d_cls] > b[:, r_cls]).mean())
                top1 = float((P.argmax(1) == d_cls).mean())
                r[f"{recv}<-{donor}"] = {
                    "dLO_to_donor": round(dlo, 3),
                    "flip": round(flip, 3),
                    "flip_base": round(base_flip, 3),
                    "top1_donor": round(top1, 3),
                }
            rec["sets"][name] = r
            m = np.mean([v["dLO_to_donor"] for v in r.values()])
            fl = np.mean([v["flip"] for v in r.values()])
            t1 = np.mean([v["top1_donor"] for v in r.values()])
            print(f"  {name:<22} dLO->donor={m:+7.2f}  pair-flip={fl:.2f}  top1-donor={t1:.2f}")
        results["pairs"].append(rec)

    # summary across pairs
    print("\n=== MEAN ACROSS PAIRS ===")
    summary = {}
    for name, _ in sets:
        dl = [v["dLO_to_donor"] for pr in results["pairs"] for v in pr["sets"][name].values()]
        fl = [v["flip"] for pr in results["pairs"] for v in pr["sets"][name].values()]
        t1 = [v["top1_donor"] for pr in results["pairs"] for v in pr["sets"][name].values()]
        summary[name] = {
            "dLO": round(float(np.mean(dl)), 3),
            "flip": round(float(np.mean(fl)), 3),
            "top1": round(float(np.mean(t1)), 3),
        }
        print(
            f"  {name:<22} dLO->donor={np.mean(dl):+7.2f}  pair-flip={np.mean(fl):.2f}  "
            f"top1-donor={np.mean(t1):.2f}"
        )
    results["summary"] = summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
