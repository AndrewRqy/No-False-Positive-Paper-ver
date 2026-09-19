"""NFP Cross-Dictionary Replication - Experiment D6.

Runs the NFP statistic on a second dictionary (cross-dictionary replication)
from cached raw ball-token activations.

Loads ball_raw_acts.pt (dumped once by dump_ball_raw_acts.py: raw layer-11 ball-token
activations for the 3000 NFP videos) and runs the NFP statistic through any dictionary:
encode -> off-screen zeroing -> within-video covariance with tau -> t-test, Bonferroni
0.05/dict_size. Then compares the flagged set with the main SAE's 85:
  - counts and per-tau breakdown;
  - decoder-space overlap: for each new flagged feature, max cosine between its decoder
    column and the main SAE's 85 flagged columns (are the same DIRECTIONS found?);
  - concept overlap: top-activating SSv2-val videos are not recomputed here; direction
    cosine is the primary bridge.

Usage:
  python -m nfp_testing.nfp_on_dict --dict_path local_runs/sae_btk/train_acts_batch_top_k_32_x8/trainer_0/ae.pt --dict_class batch_top_k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def main() -> None:
    """Run the NFP statistic for one dictionary and compare it to the main SAE.

    Loads cached raw ball-token activations, encodes them through the chosen
    dictionary, computes the within-video covariance with tau, runs a one-sample
    t-test with Bonferroni correction, and (for dictionaries with decoder
    columns) reports decoder-direction overlap with the main SAE's flagged set.
    Writes a JSON summary to the output path.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball_acts", default="local_runs/nfp_results/ball_raw_acts.pt")
    ap.add_argument("--dict_path", required=True)
    ap.add_argument(
        "--dict_class",
        default="batch_top_k",
        choices=["standard", "batch_top_k", "pca", "ica", "identity"],
    )
    ap.add_argument("--main_sae", default="local_runs/sae/ae.pt")
    ap.add_argument("--main_nfp", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--alpha", default=0.05, type=float)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD6_nfp_btk.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.dict_class == "standard":
        dic = AutoEncoder.from_pretrained(args.dict_path, device=device)
        dic.eval()
    elif args.dict_class == "batch_top_k":
        from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

        dic = BatchTopKSAE.from_pretrained(args.dict_path, device=device)
        dic.eval()
    elif args.dict_class in ("pca", "ica"):
        from dictionary_learning import PCADict, ICADict

        cls = PCADict if args.dict_class == "pca" else ICADict
        dic = cls.from_pretrained(args.dict_path, device=device, mode="sign_split")
        dic.eval()
    else:  # identity: the raw 768 residual dimensions as "features"

        class _Identity:
            """Identity dictionary exposing the raw 768 residual dims as features."""

            dict_size = 768

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                """Return the input unchanged (identity encode)."""
                return x

        dic = _Identity()
    D = dic.dict_size
    print(f"dictionary: {args.dict_class}, {D} features")

    d = torch.load(args.ball_acts, map_location="cpu")
    ball, tau, mask = d["ball"], d["tau"], d["mask"]  # [N,8,768],[N,8,5],[N,8]
    N, T, _ = ball.shape
    with torch.no_grad():
        feats = dic.encode(ball.reshape(N * T, -1).to(device).float()).cpu().reshape(N, T, -1)
    feats = feats * mask.unsqueeze(-1).float()
    psi_c = feats - feats.mean(1, keepdim=True)
    tau_c = tau - tau.mean(1, keepdim=True)
    C = torch.einsum("btd,btk->bdk", psi_c, tau_c).numpy() / T
    t = np.zeros((D, 5), np.float32)
    p = np.ones_like(t)
    for k in range(5):
        t[:, k], p[:, k] = stats.ttest_1samp(C[:, :, k], 0.0)
    bonf = args.alpha / D
    sig_new = [int(i) for i in np.where((p < bonf).any(1))[0]]
    dom_new = {i: TAU[int(np.argmax(np.abs(np.nan_to_num(t[i]))))] for i in sig_new}
    print(f"\nflagged: {len(sig_new)} ({100*len(sig_new)/D:.2f}%)")
    for name in TAU:
        n = sum(1 for i in sig_new if dom_new[i] == name)
        print(f"  dom {name:<10} {n}")

    # decoder-direction overlap with the main SAE's 85 (only for dictionary types
    # that have decoder columns)
    if args.dict_class not in ("standard", "batch_top_k"):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res = {
            "dict": args.dict_path,
            "dict_class": args.dict_class,
            "n_flagged": len(sig_new),
            "pct": round(100 * len(sig_new) / D, 3),
            "per_tau": {name: sum(1 for i in sig_new if dom_new[i] == name) for name in TAU},
        }
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nSaved -> {args.out}")
        return

    main = AutoEncoder.from_pretrained(args.main_sae, device="cpu")
    nfp = torch.load(args.main_nfp, map_location="cpu")
    p_m = nfp["p_val"].numpy()
    sig_main = [int(i) for i in np.where((p_m < 0.05 / p_m.shape[0]).any(1))[0]]
    Wm = main.decoder.weight.data.cpu()[:, sig_main]  # [768, 85]
    Wm = Wm / Wm.norm(dim=0, keepdim=True)
    Wn = dic.decoder.weight.data.cpu().float()[:, sig_new] if sig_new else None
    res = {
        "dict": args.dict_path,
        "n_flagged": len(sig_new),
        "pct": round(100 * len(sig_new) / D, 3),
        "per_tau": {name: sum(1 for i in sig_new if dom_new[i] == name) for name in TAU},
    }
    if Wn is not None and Wn.shape[1] > 0:
        Wn = Wn / Wn.norm(dim=0, keepdim=True)
        cos = (Wn.T @ Wm).abs()  # [n_new, 85]
        best = cos.max(1).values.numpy()
        print(f"\ndecoder-direction match to the main SAE's 85 flagged columns:")
        print(
            f"  max |cos| per new feature: median {np.median(best):.3f}, "
            f">=0.5: {(best >= 0.5).sum()}/{len(best)}, >=0.7: {(best >= 0.7).sum()}"
        )
        # and the reverse: how many of the main 85 have a close partner among new flags
        rbest = cos.max(0).values.numpy()
        print(
            f"  main-85 with a new-flag partner >=0.5: {(rbest >= 0.5).sum()}/85, "
            f">=0.7: {(rbest >= 0.7).sum()}"
        )
        res["new_match_main"] = {
            "median": round(float(np.median(best)), 3),
            "ge05": int((best >= 0.5).sum()),
            "ge07": int((best >= 0.7).sum()),
        }
        res["main_match_new"] = {
            "ge05": int((rbest >= 0.5).sum()),
            "ge07": int((rbest >= 0.7).sum()),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
