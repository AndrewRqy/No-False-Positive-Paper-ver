"""Direction nulls (Experiment E2) - closing the two control gaps in the valid-evidence list.

Gap 1 (F1b): the c_bar[vel_x] camera-steering claim was controlled by only two random
unit directions, and uniform-random 768-d directions are a weak null (they are nearly
orthogonal to the activation manifold, so they barely perturb anything meaningful).
Fix: on the two camera pairs, steer (a) 20 random unit directions and (b) 10
manifold-matched directions - random unit combinations of the top-50 principal
components of the ball-token activations, i.e. directions that look like real activity
- at delta +/-150, and compare the null distribution of |pair log-odds shift| and flip
rate against c_bar[vel_x] (shift -4.08, flips 0.42 on cam_lr).

Gap 2 (D3): the reversal-restoration result (NFP-85 recovers 13%, one random seed and
static both 0%) lacked extra seeds and the activation-matched control. Fix: rerun the
restoration on the 5 direction pairs with random-85 x2 fresh seeds (101, 202) and
activation-matched-85.

Usage (from repo root):
  python -m causal_analysis.exp14_patching_restore.controls_direction_nulls
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
from causal_analysis.common.steer_pair_screen import ItemFrames
from causal_analysis.exp14_patching_restore.steer_reverse_play import DIR_PAIRS

CAM_PAIRS = [
    (
        "cam_lr",
        "Turning the camera left while filming [something]",
        "Turning the camera right while filming [something]",
    ),
    (
        "cam_ud",
        "Turning the camera upwards while filming [something]",
        "Turning the camera downwards while filming [something]",
    ),
]


def main() -> None:
    """Run the E2 direction-null controls and write both parts as JSON.

    Part 1 builds a null battery on the two camera pairs: it steers the c_bar[vel_x] and
    c_bar[vel_y] directions against 20 random unit directions and 10 manifold-matched
    directions (random combinations of the top-50 activation PCs) at delta +/-150, and
    reports each direction's pair log-odds shift and flip rate plus an empirical p-value.
    Part 2 reruns the reversal-restoration on the 5 direction pairs with two fresh
    random-85 seeds and an activation-matched-85 control.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--ball_acts_v2", default="local_runs/nfp_results/ball_raw_acts_v2.pt")
    ap.add_argument("--probe_cache", default="local_runs/steering/expD4_probe_cache.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument(
        "--ssv2_val_json",
        default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json",
    )
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_rand_dirs", default=20, type=int)
    ap.add_argument("--n_manifold_dirs", default=10, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expE2_direction_nulls.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

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
        vec: Optional[torch.Tensor] = None,
        patch: Optional[Tuple[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> np.ndarray:
        """Run the classifier over cached batches and return softmax probabilities.

        Args:
            cache: Per-batch pixel-value tensors.
            vec: Optional steering vector added at the steered layer for every batch.
            patch: Optional (feature-index tensor, per-batch values) installed at the
                steered layer for each batch, or None for no patch.

        Returns:
            A [n_videos, n_classes] array of class probabilities.
        """
        outs = []
        for bi, pv in enumerate(cache):
            steer.add_vec = vec
            if patch is not None:
                steer.patch_idx = patch[0]
                steer.patch_vals = patch[1][bi]
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
            steer.add_vec = None
            steer.patch_idx = steer.patch_vals = None
        return torch.cat(outs, 0).numpy()

    out = {}

    # ---------------- Part 1: direction nulls for the c_bar claim ----------------
    d2 = torch.load(args.ball_acts_v2, map_location="cpu")
    ball, tau = d2["ball"].float(), d2["tau"].float()
    hc = ball - ball.mean(1, keepdim=True)
    tc = tau - tau.mean(1, keepdim=True)
    cbar = (torch.einsum("btd,btk->bdk", hc, tc) / ball.shape[1]).mean(0)  # [768,5]
    cbar_vx = cbar[:, 1] / cbar[:, 1].norm()
    cbar_vy = cbar[:, 2] / cbar[:, 2].norm()

    # manifold basis: top-50 PCs of ball activations
    X = ball.reshape(-1, 768)
    X = X[X.norm(dim=1) > 1e-6]
    Xc = X - X.mean(0)
    U, S, Vt = torch.linalg.svd(Xc[::7], full_matrices=False)  # subsample rows for speed
    PCs = Vt[:50].T  # [768, 50]

    g = torch.Generator().manual_seed(1234)
    rand_dirs = [torch.randn(768, generator=g) for _ in range(args.n_rand_dirs)]
    rand_dirs = [(v / v.norm(), f"rand{i:02d}") for i, v in enumerate(rand_dirs)]
    mani = []
    for i in range(args.n_manifold_dirs):
        c = torch.randn(50, generator=g)
        v = PCs @ c
        mani.append((v / v.norm(), f"manifold{i:02d}"))

    print("=== Part 1: direction-null battery on camera pairs (delta +/-150) ===")
    out["part1"] = {}
    for key, pl, nl in CAM_PAIRS:
        cp, cn = label2idx[pl], label2idx[nl]
        caches = {
            "pos": cache_of(by_tmpl[pl][: args.per_class]),
            "neg": cache_of(by_tmpl[nl][: args.per_class]),
        }

        def eval_dir(u: torch.Tensor) -> Tuple[float, float]:
            """Steer one unit direction on the current camera pair.

            Args:
                u: A unit direction in the 768-d activation space.

            Returns:
                A tuple of the pair log-odds shift (half the +/-150 difference) and the
                mean flip rate at the shift-aligned polarity.
            """
            los = {}
            for dl_ in (150.0, -150.0):
                vec = (dl_ * u).float()
                lo_s = []
                for side in ("pos", "neg"):
                    P = probs(caches[side], vec=vec)
                    lo_s.append(float(np.mean(np.log(P[:, cp] + 1e-12) - np.log(P[:, cn] + 1e-12))))
                los[dl_] = np.mean(lo_s)
            shift = (los[150.0] - los[-150.0]) / 2
            s_pos = 150.0 if shift > 0 else -150.0
            Pn = probs(caches["neg"], vec=(s_pos * u).float())
            Pp = probs(caches["pos"], vec=(-s_pos * u).float())
            flip = (
                float((Pn[:, cp] > Pn[:, cn]).mean()) + float((Pp[:, cn] > Pp[:, cp]).mean())
            ) / 2
            return shift, flip

        target = cbar_vx if key == "cam_lr" else cbar_vy
        t_shift, t_flip = eval_dir(target)
        null_sh, null_fl = [], []
        for u, name in rand_dirs + mani:
            s_, f_ = eval_dir(u)
            null_sh.append(abs(s_))
            null_fl.append(f_)
        null_sh, null_fl = np.array(null_sh), np.array(null_fl)
        n_r = args.n_rand_dirs
        p_emp = float((null_sh >= abs(t_shift)).mean())
        print(f"  {key}: c_bar |shift|={abs(t_shift):.2f} flip={t_flip:.2f}")
        print(
            f"    null (n={len(null_sh)}: {n_r} random + {len(mani)} manifold): "
            f"|shift| mean={null_sh.mean():.2f} max={null_sh.max():.2f}; "
            f"flip mean={null_fl.mean():.2f} max={null_fl.max():.2f}"
        )
        print(
            f"    manifold-only |shift| mean={null_sh[n_r:].mean():.2f} "
            f"max={null_sh[n_r:].max():.2f}"
        )
        print(f"    empirical p(|shift|_null >= c_bar) = {p_emp:.3f}")
        out["part1"][key] = {
            "cbar_shift": round(t_shift, 3),
            "cbar_flip": round(t_flip, 3),
            "null_shift_mean": round(float(null_sh.mean()), 3),
            "null_shift_max": round(float(null_sh.max()), 3),
            "manifold_shift_max": round(float(null_sh[n_r:].max()), 3),
            "null_flip_max": round(float(null_fl.max()), 3),
            "p_emp": p_emp,
        }

    # ---------------- Part 2: extra controls for reversal restoration ----------------
    print("\n=== Part 2: reversal-restore extra controls ===")
    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy()
    t_all = nfp["t_stat"].numpy()
    sig = sorted(int(i) for i in np.where((p_all < 0.05 / p_all.shape[0]).any(1))[0])
    nonsig = [k for k in range(sae.dict_size) if k not in set(sig)]
    r1 = sorted(np.random.RandomState(101).choice(nonsig, 85, replace=False))
    r2 = sorted(np.random.RandomState(202).choice(nonsig, 85, replace=False))
    mean_act = torch.load(args.probe_cache, map_location="cpu")["feats"].numpy().mean(0)
    taken, match = set(), []
    pool = sorted(nonsig, key=lambda k: mean_act[k])
    pool_acts = np.array([mean_act[k] for k in pool])
    for k in sig:
        j = int(
            np.argmin(
                np.abs(pool_acts - mean_act[k]) + 1e9 * np.isin(np.arange(len(pool)), list(taken))
            )
        )
        taken.add(j)
        match.append(pool[j])
    SETS = [("NFP-85", sig), ("rnd85-s101", r1), ("rnd85-s202", r2), ("actmatch-85", sorted(match))]

    def capture(cache: List[torch.Tensor], idx: torch.Tensor) -> List[torch.Tensor]:
        """Capture forward-run token activations for the chosen features, per batch.

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
        vals, i = [], 0
        allv = torch.cat(steer.captured_tokens, 0)
        for pv in cache:
            vals.append(allv[i : i + pv.shape[0]])
            i += pv.shape[0]
        return vals

    agg = {"fwd": [], "rev": [], **{n: [] for n, _ in SETS}}
    for key, pl, nl in DIR_PAIRS:
        cp, cn = label2idx.get(pl, -1), label2idx.get(nl, -1)
        if cp < 0 or cn < 0:
            continue
        for side, cls_l, own, other in [("pos", pl, cp, cn), ("neg", nl, cn, cp)]:
            cache = cache_of(by_tmpl[cls_l][: args.per_class])
            cache_rev = [pv.flip(1) for pv in cache]
            P_f, P_r = probs(cache), probs(cache_rev)

            def lo(P: np.ndarray) -> float:
                """Own-vs-other log-odds averaged over the side's clips."""
                return float(np.mean(np.log(P[:, own] + 1e-12) - np.log(P[:, other] + 1e-12)))

            agg["fwd"].append(lo(P_f))
            agg["rev"].append(lo(P_r))
            for name, ks in SETS:
                idx = torch.tensor(ks)
                vals = capture(cache, idx)
                agg[name].append(lo(probs(cache_rev, patch=(idx, vals))))
    f, r = np.mean(agg["fwd"]), np.mean(agg["rev"])
    out["part2"] = {"fwd": round(float(f), 3), "rev": round(float(r), 3)}
    print(f"  forward {f:+.3f}  reversed {r:+.3f}")
    for name, _ in SETS:
        rec = (np.mean(agg[name]) - r) / (f - r + 1e-9)
        out["part2"][name] = round(float(rec), 3)
        print(f"  restore {name:<12} recovery = {100*rec:.0f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
