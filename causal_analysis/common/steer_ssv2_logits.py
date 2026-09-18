"""
Experiment A — steer a temporal SAE feature and read VideoMAE's own SSv2 logits.

Recipe (clamp + re-add reconstruction error, on the layer-11 post-MLP residual where the
SAE was trained):
    f = Enc(x); e = x - Dec(f); f[...,k] = s (all 1568 tokens); x_out = Dec(f) + e
This sets feature k to value s along its decoder direction and pays no reconstruction tax
(equivalently x_out = x + n_k*(s - f_k)). We then finish the forward pass through the
174-way Something-Something-v2 classifier and compare the steered class distribution to the
unsteered one.

Hypothesis: clamping the ACCELERATION feature (feat05087) shifts probability mass toward
forceful/fast classes (throwing, hitting, "falls off/over") and away from gentle/slight
classes ("slightly moves", "almost doesn't move", "pretending to"). Summary readout:
    force_shift = sum_{c in FAST} dP_c  -  sum_{c in SLOW} dP_c

Controls: sign-flip (clamp to 0), N random-feature clamps (null distribution), and the
speed feature (feat02818) for an accel-vs-speed comparison.

Usage (from sae-for-vlm/):
  python analysis/steer_ssv2_logits.py --n_videos 64 --features 5087 2818
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder

N_FRAMES = 16

# Publication default: a decode failure must NOT be silently turned into a black
# video with the real label (that corrupts every metric computed over it). Fail
# closed. Entry points may set this to "zero" for diagnostics (review item 8).
DEFAULT_DECODE_ERROR = "raise"   # {"raise", "zero"}


class SSv2Frames(Dataset):
    """Real SSv2 validation clips: decode webm -> 16 uniformly-sampled RGB frames.

    decode_error:
      "raise" (default) -- re-raise with the offending video path; a corrupt clip
                           stops the run rather than silently biasing results;
      "zero"            -- substitute a gray clip AND record the id in
                           `self.failed_ids` so the count can be saved and
                           compared across conditions.
    """
    def __init__(self, videos_dir, val_json, n, seed=0, decode_error=None):
        val = json.load(open(val_json))
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(val), size=min(n, len(val)), replace=False)
        self.items = [val[i] for i in sel]
        self.dir = Path(videos_dir)
        self.decode_error = decode_error or DEFAULT_DECODE_ERROR
        self.failed_ids = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import av
        it = self.items[i]
        path = self.dir / f"{it['id']}.webm"
        try:
            cont = av.open(str(path))
            frames = [f.to_ndarray(format="rgb24") for f in cont.decode(video=0)]
            cont.close()
            idx = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
            imgs = [Image.fromarray(frames[j]) for j in idx]
        except Exception as e:
            if self.decode_error == "raise":
                raise RuntimeError(f"failed to decode video {path}: {e}") from e
            self.failed_ids.append(it["id"])   # "zero": gray fallback, but recorded
            imgs = [Image.new("RGB", (224, 224)) for _ in range(N_FRAMES)]
        return imgs, it["id"], it.get("template", "")


def ssv2_collate(processor):
    def c(batch):
        inputs = processor(images=[b[0] for b in batch], return_tensors="pt")
        return inputs, [b[1] for b in batch], [b[2] for b in batch]
    return c

# SSv2 class keyword sets (matched case-insensitively against id2label).
# A label counts FAST only if it hits a FAST kw and no SLOW kw (so "push so it slightly
# moves" is SLOW, not FAST).
FAST_KW = ["throwing", "hitting", "falls off", "falls over", "falls down", "tipping",
           "so that it falls", "until it breaks", "tearing", "kicking", "dropping",
           "quickly stops spinning", "knocking", "slamming", "collides"]
SLOW_KW = ["slightly moves", "almost doesn't move", "doesn't move", "so lightly",
           "pretending", "without letting", "continues spinning", "letting something roll",
           "barely", "a little"]


def build_fast_slow(id2label):
    fast, slow = [], []
    for i, lab in id2label.items():
        L = lab.lower()
        has_slow = any(k in L for k in SLOW_KW)
        has_fast = any(k in L for k in FAST_KW)
        if has_slow:
            slow.append(int(i))
        elif has_fast:
            fast.append(int(i))
    return fast, slow


class SteerLayer(nn.Module):
    """Wrap encoder.layer[L]; optionally clamp one SAE feature in the residual output."""
    def __init__(self, base, sae):
        super().__init__()
        self.base, self.sae = base, sae
        self.enabled = False
        self.k = None
        self.s = 0.0
        self.record = False          # capture per-clip mean-pooled feature activations
        self.record_raw = False      # capture per-clip mean-pooled RAW residual (768-d)
        self.add_vec = None          # if set, add this [768] vector to the residual (all tokens)
        self.captured = []
        self.captured_raw = []
        # token-level patching (interchange interventions): record_tokens_idx captures the
        # FULL token grid of activations for a feature subset; patch_idx/patch_vals replace
        # those features' activations with donor values (natural range, unlike clamping).
        self.record_tokens_idx = None   # LongTensor [K] -> capture f[..., idx] as [B,1568,K]
        self.captured_tokens = []
        self.patch_idx = None           # LongTensor [K]
        self.patch_vals = None          # tensor [B, 1568, K] (donor activations)
        # subspace erasure: if set to a [768,768] projector P onto a subspace,
        # every token is replaced by x - P x (the subspace is projected out).
        self.proj_out = None
        # reconstruction-only mode: replace every token by its SAE
        # reconstruction decode(encode(x)) with no error re-add.
        self.reconstruct = False
        # when True, feature interventions output decode(f') WITHOUT re-adding
        # the reconstruction error (pure-SAE pipeline)
        self.no_readd = False

    def forward(self, hidden_states, *args, **kwargs):
        kwargs.pop("head_mask", None)
        out = self.base(hidden_states, *args, **kwargs)
        is_tuple = isinstance(out, tuple)
        acts = out[0] if is_tuple else out
        rest = out[1:] if is_tuple else ()
        if self.record:
            with torch.no_grad():
                self.captured.append(self.sae.encode(acts).mean(1).detach().cpu())  # [B, dict]
        if self.record_raw:
            self.captured_raw.append(acts.mean(1).detach().cpu())                   # [B, 768]
        if self.add_vec is not None:
            acts = acts + self.add_vec.to(acts.dtype).to(acts.device)               # diff-of-means steer
        if self.reconstruct:
            acts = self.sae.decode(self.sae.encode(acts))
        if self.proj_out is not None:
            P = self.proj_out.to(acts.dtype).to(acts.device)
            acts = acts - acts @ P.T
        if self.record_tokens_idx is not None:
            with torch.no_grad():
                f = self.sae.encode(acts)
                self.captured_tokens.append(f[..., self.record_tokens_idx].detach().cpu())
        if self.patch_idx is not None and self.patch_vals is not None:
            x = acts
            f = self.sae.encode(x)
            e = x - self.sae.decode(f)
            f = f.clone()
            f[..., self.patch_idx] = self.patch_vals.to(f.dtype).to(f.device)
            acts = self.sae.decode(f) if self.no_readd else self.sae.decode(f) + e
        if self.enabled and self.k is not None:
            x = acts
            f = self.sae.encode(x)
            e = x - self.sae.decode(f)
            f = f.clone()
            f[..., self.k] = self.s
            acts = self.sae.decode(f) if self.no_readd else self.sae.decode(f) + e
        # preserve the base layer's return type (bare tensor vs tuple) — the classifier
        # head does sequence_output.mean(1), so a stray 1-tuple would break it.
        return ((acts,) + rest) if is_tuple else acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--feat_acts", default="local_runs/nfp_results/sae_feat_acts.pt")
    ap.add_argument("--input", default="ssv2", choices=["ssv2", "balls"],
                    help="ssv2 = real SSv2-val clips (in-distribution); balls = synthetic NFP set")
    ap.add_argument("--dataset_dir", default="data/output/nfp", help="for --input balls")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--labels_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/labels.json")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--features", nargs="*", type=int, default=[5087, 2818])
    ap.add_argument("--s_mults", nargs="*", type=float, default=[1.0, 3.0, 10.0])
    ap.add_argument("--n_random", default=10, type=int)
    ap.add_argument("--n_videos", default=64, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_results.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    fast, slow = build_fast_slow(id2label)
    print(f"SSv2 classes: {len(id2label)} | FAST set: {len(fast)} | SLOW set: {len(slow)}")
    print("  FAST e.g.:", [id2label[str(i)] if str(i) in id2label else id2label[i] for i in fast[:5]])
    print("  SLOW e.g.:", [id2label[str(i)] if str(i) in id2label else id2label[i] for i in slow[:5]])

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer

    # steering scale per feature = 99th percentile of its per-video peak activation
    fa = torch.load(args.feat_acts, map_location="cpu")
    fmax = fa["feat_max"].numpy()
    scale = {k: float(np.percentile(fmax[:, k], 99)) for k in args.features}
    print("feature p99 peak-activation scale:", {k: round(v, 2) for k, v in scale.items()})

    # load a sample of input videos
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    if args.input == "ssv2":
        ds = SSv2Frames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=ssv2_collate(proc))
        print(f"Input: {len(ds)} real SSv2-val clips")
    else:
        ds = Subset(NFPDataset(Path(args.dataset_dir)), list(range(min(args.n_videos, 3000))))
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=make_collate(proc))
        print(f"Input: {len(ds)} synthetic ball clips (OOD for SSv2 classifier)")

    fast_t = torch.tensor(fast, device=device)
    slow_t = torch.tensor(slow, device=device)

    # decode/process inputs ONCE and cache pixel_values: SSv2 webm decoding is the
    # bottleneck and every steering condition re-runs the forward pass over the same clips.
    cached, tmpl_all = [], []
    for batch in dl:
        cached.append(batch[0]["pixel_values"])
        if args.input == "ssv2":
            tmpl_all += batch[2]
    print(f"cached pixel_values for {sum(b.shape[0] for b in cached)} clips")

    def all_probs():
        outs = []
        for pv in cached:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, dim=-1).cpu())
        return torch.cat(outs, 0)

    def force_shift(dP):
        return float(dP[:, fast_t.cpu()].sum(1).mean() - dP[:, slow_t.cpu()].sum(1).mean())

    results = {"input": args.input, "scale": scale, "conditions": [],
               "fast_classes": [id2label.get(str(i), id2label.get(i)) for i in fast],
               "slow_classes": [id2label.get(str(i), id2label.get(i)) for i in slow]}

    steer.enabled = False
    base = all_probs()
    results["n_videos"] = int(base.shape[0])
    print(f"\nbaseline computed on {base.shape[0]} videos. "
          f"baseline FAST mass={base[:, fast].sum(1).mean():.4f} "
          f"SLOW mass={base[:, slow].sum(1).mean():.4f}")
    if args.input == "ssv2":
        label2idx = {v: int(k) for k, v in id2label.items()}   # classifier's own bracketed labels
        gt = [label2idx.get(t, -1) for t in tmpl_all]
        pred = base.argmax(1).numpy()
        ok = [(int(p) == g) for p, g in zip(pred, gt) if g >= 0]
        acc = float(np.mean(ok)) if ok else float("nan")
        print(f"  sanity: baseline top-1 acc on this SSv2-val sample = {acc:.3f} (n={len(ok)})")
        results["baseline_top1_acc"] = acc

    rng = np.random.RandomState(0)
    rand_feats = rng.choice([k for k in range(sae.dict_size) if k not in args.features],
                            size=args.n_random, replace=False).tolist()

    def run(label, k, s):
        steer.enabled = True; steer.k = k; steer.s = s
        P = all_probs()
        steer.enabled = False
        dP = P - base
        fs = force_shift(dP)
        top = (dP.mean(0)).numpy()
        up = np.argsort(-top)[:5]; dn = np.argsort(top)[:5]
        rec = {"label": label, "feature": k, "s": round(s, 3), "force_shift": round(fs, 5),
               "top_up": [(id2label.get(str(i), id2label.get(i)), round(float(top[i]), 4)) for i in up],
               "top_down": [(id2label.get(str(i), id2label.get(i)), round(float(top[i]), 4)) for i in dn]}
        results["conditions"].append(rec)
        print(f"  {label:28s} force_shift={fs:+.5f}  up:{rec['top_up'][0]}  down:{rec['top_down'][0]}")
        return fs

    print("\n=== TARGET FEATURES (s-sweep) ===")
    for k in args.features:
        for m in args.s_mults:
            run(f"feat{k}_s{m:g}x", k, scale[k] * m)
        run(f"feat{k}_signflip0", k, 0.0)   # clamp to 0 -> expect opposite shift

    print("\n=== RANDOM-FEATURE CONTROLS (at 3x scale of feat0) ===")
    s_ctrl = scale[args.features[0]] * 3.0
    null = [run(f"rand_feat{k}", k, s_ctrl) for k in rand_feats]
    null = np.array(null)
    nm, ns = float(null.mean()), float(null.std())
    print(f"\nrandom-feature force_shift null: mean={nm:+.5f} std={ns:.5f} "
          f"range=[{null.min():+.5f},{null.max():+.5f}]")
    results["null"] = {"mean": nm, "std": ns, "n": len(null),
                       "features": rand_feats, "values": null.tolist()}

    print("\n=== TARGET force_shift as z-score vs random-feature null ===")
    for rec in results["conditions"]:
        if rec["label"].startswith("rand_feat"):
            continue
        z = (rec["force_shift"] - nm) / ns if ns > 0 else float("nan")
        rec["z_vs_null"] = round(z, 2)
        flag = "  <-- exceeds null" if abs(z) >= 2 else ""
        print(f"  {rec['label']:28s} force_shift={rec['force_shift']:+.5f}  z={z:+5.2f}{flag}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
