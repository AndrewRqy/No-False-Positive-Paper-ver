"""
Cross-model causal battery: the paper's three steering experiments + appendix
ablations, generalized over model families (timesformer, vjepa2).

Protocols mirror the VideoMAE originals:
  cache   - mean SAE feature activations over N SSv2-val clips (for
            activation-matched controls), as in expD4's probe cache.
  erasure - span erasure x - QQ^T x on the all-class battery (174 x
            n_per_class) and the family battery (classes with excess
            flagged-span dependence >= 0.4), with controls {random s101,
            random s202, static, activation-matched}; plus the top-k size
            sweep (appendix).
  repair  - amplification dose sweep (capture own features, patch back
            alpha x) on the family classes; net accuracy over wrong+right.
  pairs   - per-feature clamp screen on the 10 class pairs, strict top-1
            flip + rank-based rate, plus the size-matched static control.

The SAE is loaded unnormalized and then manually normalized (zero-guarded)
so clamp magnitudes are in decoder-unit-norm units, comparable to the
VideoMAE runs' s_abs = 150.

Usage (from sae-for-vlm/):
  python analysis/steer_expansion.py --family timesformer --stage cache
  python analysis/steer_expansion.py --family timesformer --stage erasure
  python analysis/steer_expansion.py --family timesformer --stage repair
  python analysis/steer_expansion.py --family timesformer --stage pairs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from causal_analysis.common.steer_ssv2_logits import SteerLayer
from causal_analysis.common.steer_pair_screen import ItemFrames
from causal_analysis.common.steer_span_erasure import projector
from utils.config import add_config_arg, apply_config
from utils.provenance import run_metadata

FAMILY = {
    "videomae_base": dict(
        model="MCG-NJU/videomae-base-ssv2", layer=11,
        frame_step=1, video_proc=False,
        sae="local_runs/expansion/sae_videomae_pretrain.pt",
        nfp="local_runs/expansion/videomae_pretrain_sae.pt",
        probe="local_runs/expansion/vm_base_probe.pt",
        batch=6, autocast=False),
    "videomae": dict(
        model="MCG-NJU/videomae-base-finetuned-ssv2", layer=11,
        frame_step=1, video_proc=False,
        sae="local_runs/sae/ae.pt",
        nfp="local_runs/nfp_results/sae_nfp_v2.pt",
        batch=6, autocast=False),
    "timesformer": dict(
        model="facebook/timesformer-base-finetuned-ssv2", layer=11,
        frame_step=2, video_proc=False,
        sae="local_runs/expansion/sae_timesformer.pt",
        nfp="local_runs/expansion/timesformer_sae_l11.pt",
        batch=6, autocast=False),
    "timesformer_l10": dict(
        model="facebook/timesformer-base-finetuned-ssv2", layer=10,
        frame_step=2, video_proc=False,
        sae="local_runs/expansion/sae_timesformer_l10.pt",
        nfp="local_runs/expansion/timesformer_sae_l10.pt",
        batch=6, autocast=False),
    "vjepa2_l11": dict(
        model="facebook/vjepa2-vitl-fpc16-256-ssv2", layer=11,
        frame_step=1, video_proc=True,
        sae="local_runs/expansion/sae_vjepa2_l11.pt",
        nfp="local_runs/expansion/vjepa2_sae_l11.pt",
        batch=3, autocast=True),
    "vjepa2_ft3": dict(
        model="/net/scratch2/renqy/ft_vj2_top_v3/classifier_export", layer=23,
        frame_step=1, video_proc=True,
        sae="local_runs/expansion/sae_vjepa2_ft3.pt",
        nfp="local_runs/expansion/vjepa2_ft3_sae.pt",
        batch=3, autocast=True),
    "vjepa2_ft": dict(
        model="/net/scratch2/renqy/ft_vj2_top/classifier_export", layer=23,
        frame_step=1, video_proc=True,
        sae="local_runs/expansion/sae_vjepa2_ft.pt",
        nfp="local_runs/expansion/vjepa2_ft_sae.pt",
        batch=3, autocast=True),
    "vjepa2_probe": dict(
        model="facebook/vjepa2-vitl-fpc16-256-ssv2", layer=2, attach="pooler",
        frame_step=1, video_proc=True,
        sae="local_runs/expansion/sae_vjepa2_probe.pt",
        nfp="local_runs/expansion/vjepa2_probe_sae.pt",
        batch=3, autocast=True),
    "vjepa2_raw": dict(
        model="facebook/vjepa2-vitl-fpc16-256-ssv2", layer=23,
        frame_step=1, video_proc=True,
        sae="identity", dim=1024,
        nfp="local_runs/expansion/vjepa2_raw_l23.pt",
        batch=3, autocast=True),
    "vjepa2_raw_l11": dict(
        model="facebook/vjepa2-vitl-fpc16-256-ssv2", layer=11,
        frame_step=1, video_proc=True,
        sae="identity", dim=1024, max_identified=128,
        nfp="local_runs/expansion/vjepa2_raw_l11.pt",
        batch=3, autocast=True),
    "vjepa2": dict(
        model="facebook/vjepa2-vitl-fpc16-256-ssv2", layer=23,
        frame_step=1, video_proc=True,
        sae="local_runs/expansion/sae_vjepa2.pt",
        nfp="local_runs/expansion/vjepa2_sae_l23.pt",
        batch=3, autocast=True),
}

PAIRS10 = [
    ("push_lr", "vel_x", "Pushing [something] from left to right", "Pushing [something] from right to left"),
    ("pull_lr", "vel_x", "Pulling [something] from left to right", "Pulling [something] from right to left"),
    ("cam_lr", "vel_x", "Turning the camera left while filming [something]",
     "Turning the camera right while filming [something]"),
    ("move_ud", "vel_y", "Moving [something] up", "Moving [something] down"),
    ("cam_ud", "vel_y", "Turning the camera upwards while filming [something]",
     "Turning the camera downwards while filming [something]"),
    ("obj_cam", "depth", "Moving [something] towards the camera", "Moving [something] away from the camera"),
    ("obj_obj", "depth", "Moving [something] closer to [something]", "Moving [something] away from [something]"),
    ("cam_appr", "depth", "Approaching [something] with your camera",
     "Moving away from [something] with your camera"),
    ("fall_speed", "speed", "[Something] falling like a rock", "[Something] falling like a feather or paper"),
    ("spin_stop", "accel_mag", "Spinning [something] so it continues spinning",
     "Spinning [something] that quickly stops spinning"),
]


class IdentitySAE(torch.nn.Module):
    """Identity dictionary: interventions act directly on residual-stream coordinates."""
    def __init__(self, d, device):
        super().__init__()
        self.decoder = torch.nn.Linear(d, d, bias=False)
        self.decoder.weight.data = torch.eye(d)
        self.to(device)
        self.eval()

    def encode(self, x):
        return x

    def decode(self, f):
        return f


def load_sae_normalized(path, device):
    ae = AutoEncoder.from_pretrained(path, device=device, normalize_decoder=False)
    with torch.no_grad():
        norms = ae.decoder.weight.norm(dim=0)
        safe = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
        ae.decoder.weight.data /= safe
        ae.encoder.weight.data *= safe[:, None]
        ae.encoder.bias.data *= safe
    ae.eval()
    return ae


class VideoMAEBaseClf(torch.nn.Module):
    """VideoMAE without label finetuning plus a linear probe on the
    mean-pooled final-layer activations (trained by vm_base_probe.py)."""
    def __init__(self, model_name, probe_path, device):
        super().__init__()
        from transformers import VideoMAEModel, AutoConfig
        self.videomae = VideoMAEModel.from_pretrained(model_name)
        # label mapping from the finetuned checkpoint, the same mapping the
        # probe was trained with
        self.config = AutoConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        pw = torch.load(probe_path, map_location="cpu", weights_only=False)
        self.classifier = torch.nn.Linear(pw["weight"].shape[1], pw["weight"].shape[0])
        self.classifier.weight.data = pw["weight"].clone()
        self.classifier.bias.data = pw["bias"].clone()
        self.to(device)
        self.eval()

    def forward(self, pixel_values=None, **kw):
        import types
        h = self.videomae(pixel_values=pixel_values).last_hidden_state
        return types.SimpleNamespace(logits=self.classifier(h.mean(1)))


def build(family, device):
    cfg = FAMILY[family]
    if family == "videomae_base":
        from transformers import VideoMAEImageProcessor
        clf = VideoMAEBaseClf(cfg["model"], cfg["probe"], device)
        proc = VideoMAEImageProcessor.from_pretrained(cfg["model"])
        layers = clf.videomae.encoder.layer
    elif family.startswith("videomae"):
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        clf = VideoMAEForVideoClassification.from_pretrained(cfg["model"]).to(device).eval()
        proc = VideoMAEImageProcessor.from_pretrained(cfg["model"])
        layers = clf.videomae.encoder.layer
    elif family.startswith("timesformer"):
        from transformers import TimesformerForVideoClassification, AutoImageProcessor
        clf = TimesformerForVideoClassification.from_pretrained(cfg["model"]).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(cfg["model"])
        layers = clf.timesformer.encoder.layer
    else:
        from transformers import VJEPA2ForVideoClassification, AutoVideoProcessor
        clf = VJEPA2ForVideoClassification.from_pretrained(cfg["model"]).to(device).eval()
        proc = AutoVideoProcessor.from_pretrained(cfg["model"])
        if cfg.get("attach") == "pooler":
            layers = clf.pooler.self_attention_layers
        else:
            layers = clf.vjepa2.encoder.layer
    if cfg["sae"] == "identity":
        sae = IdentitySAE(cfg["dim"], device)
    else:
        sae = load_sae_normalized(cfg["sae"], device)
    steer = SteerLayer(layers[cfg["layer"]], sae).to(device)
    layers[cfg["layer"]] = steer
    return clf, proc, steer, sae, cfg


def make_collate(proc, cfg):
    def c(batch):
        frames = [b[0][::cfg["frame_step"]] for b in batch]
        if cfg["video_proc"]:
            inputs = proc([[np.asarray(f) for f in fl] for fl in frames], return_tensors="pt")
        else:
            inputs = proc(images=frames, return_tensors="pt")
        return inputs, [b[1] for b in batch], [b[2] for b in batch]
    return c


def forward_logits(clf, inputs, device, cfg):
    key = "pixel_values_videos" if cfg["video_proc"] else "pixel_values"
    pv = inputs[key].to(device)
    with torch.no_grad():
        if cfg["autocast"]:
            with torch.autocast("cuda", dtype=torch.float16):
                out = clf(**{key: pv}).logits.float()
        else:
            out = clf(**{key: pv}).logits
    return out


def load_flags(cfg):
    d = torch.load(cfg["nfp"], map_location="cpu")
    p, t = d["p_val"].numpy(), d["t_stat"].numpy()
    D = p.shape[0]
    valid = np.isfinite(t).all(1)
    sig_mask = (p < 0.05 / D).any(1) & valid
    sig = sorted(int(i) for i in np.where(sig_mask)[0])
    strength = np.abs(np.nan_to_num(t))[sig].max(1)
    sig_ranked = [int(i) for i in np.array(sig)[np.argsort(-strength)]]
    static_pool = sorted(int(i) for i in np.where(
        valid & (np.abs(np.nan_to_num(t, nan=1e9)).max(1) < 2.0) & ~sig_mask)[0])
    nonsig = sorted(int(i) for i in np.where(~sig_mask)[0])
    cap = cfg.get("max_identified")
    if cap and len(sig) > cap:
        sig_ranked = sig_ranked[:cap]
        sig = sorted(sig_ranked)
        print(f"identified capped to top {cap} by max |t| (controls stay size-matched)")
    return D, sig, sig_ranked, static_pool, nonsig


def cache_frames(items, videos_dir, proc, cfg, batch):
    dl = DataLoader(ItemFrames(videos_dir, items), batch_size=batch, shuffle=False,
                    num_workers=0, collate_fn=make_collate(proc, cfg))
    return [b[0] for b in dl]


def preds_for(clf, caches, device, cfg, steer, proj=None, patch=None):
    outs = []
    if getattr(steer, "no_readd", False):
        # pure-SAE pipeline: plain passes and erasure run on the reconstruction;
        # patch passes already output decode(f') themselves
        steer.reconstruct = (patch is None)
    for inputs in caches:
        if patch is not None:
            idx, alpha = patch
            steer.record_tokens_idx = idx
            steer.captured_tokens = []
            forward_logits(clf, inputs, device, cfg)
            steer.record_tokens_idx = None
            fvals = steer.captured_tokens[0]
            steer.patch_idx = idx
            steer.patch_vals = alpha * fvals
        if proj is not None:
            steer.proj_out = proj
        outs.append(forward_logits(clf, inputs, device, cfg).argmax(-1).cpu().numpy())
        steer.proj_out = None
        steer.patch_idx = steer.patch_vals = None
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    ap.add_argument("--family", required=True, choices=list(FAMILY.keys()))
    ap.add_argument("--stage", required=True, choices=["cache", "erasure", "repair", "pairs", "repair_addons", "amp_allclass", "erasure_allclass", "recon_acc"])
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--n_cache", default=400, type=int)
    ap.add_argument("--n_per_class", default=10, type=int)
    ap.add_argument("--mine_per_class", default=24, type=int)
    ap.add_argument("--per_class_pairs", default=12, type=int)
    ap.add_argument("--alphas", nargs="*", type=float, default=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument("--s_abs", default=150.0, type=float)
    ap.add_argument("--excess_bar", default=0.4, type=float)
    ap.add_argument("--relaxed_bar", default=0.2, type=float,
                    help="repair_addons: fallback excess bar when the family is empty")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--limit_classes", default=-1, type=int, help="debug: cap class count")
    ap.add_argument("--static_seed", default=-1, type=int,
                    help="pairs: alternative RandomState seed for the static control draw; "
                         "-1 keeps the default (seed+7). Output gets a _s<seed> suffix.")
    ap.add_argument("--static_only", action="store_true",
                    help="pairs: screen only the static control set")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pairs_json", default="",
                    help="JSON with {'pairs': [[key, axis, pos_label, neg_label], ...]} replacing PAIRS10")
    ap.add_argument("--pairs_tag", default="",
                    help="suffix for the pairs output filename, e.g. rnd0")
    ap.add_argument("--pairs_topk", type=int, default=0,
                    help="cap pair screening to top-K identified by max |t| (and K statics)")
    ap.add_argument("--no_error_readd", action="store_true",
                    help="pure-SAE pipeline: interventions output decode(f') with no error "
                         "re-add; baseline is the plain reconstruction. Outputs get _noreadd.")
    ap.add_argument("--outdir", default="local_runs/expansion")
    args = ap.parse_args()
    apply_config(ap, args)
    device = torch.device(args.device)
    fam = args.family
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # provenance stamped into every result this run writes (review item 3)
    cfg_sae = FAMILY[fam].get("sae") if isinstance(FAMILY.get(fam), dict) else None
    prov = run_metadata(
        args, checkpoints={"sae": cfg_sae if isinstance(cfg_sae, str) else None},
        family=fam, stage=args.stage)

    clf, proc, steer, sae, cfg = build(fam, device)
    if args.no_error_readd:
        steer.no_readd = True
    nrsfx = "_noreadd" if args.no_error_readd else ""
    id2label = clf.config.id2label
    label2idx = {v: int(k) for k, v in id2label.items()}
    D, sig, sig_ranked, static_pool, nonsig = load_flags(cfg)
    N = len(sig)
    print(f"[{fam}] flagged={N}  static-pool={len(static_pool)}  D={D}")

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1)
    vrng.shuffle(val)
    Wd = sae.decoder.weight.data.cpu()

    n_r = min(N, len(nonsig))
    r101 = sorted(np.random.RandomState(101).choice(nonsig, n_r, replace=False))
    r202 = sorted(np.random.RandomState(202).choice(nonsig, n_r, replace=False))
    st_n = min(N, len(static_pool))
    st_seed = args.static_seed if args.static_seed >= 0 else args.seed + 7
    static_set = sorted(np.random.RandomState(st_seed).choice(static_pool, st_n, replace=False))

    # ---------------- cache ----------------
    if args.stage == "cache":
        items = val[:args.n_cache]
        caches = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
        steer.record = True
        steer.captured = []
        for inputs in caches:
            forward_logits(clf, inputs, device, cfg)
        steer.record = False
        feats = torch.cat(steer.captured, 0)     # [n, D] mean-pooled
        torch.save({"feats": feats}, out / f"{fam}_probe_cache.pt")
        print(f"cache: {feats.shape} -> {fam}_probe_cache.pt")
        return

    # activation matching (needs cache)
    cache_path = out / f"{fam}_probe_cache.pt"
    actmatch = None
    if cache_path.exists():
        mean_act = torch.load(cache_path, map_location="cpu")["feats"].numpy().mean(0)
        taken, match = set(), []
        order_pool = sorted(nonsig, key=lambda k: mean_act[k])
        pool_acts = np.array([mean_act[k] for k in order_pool])
        for k in sig:
            j = int(np.argmin(np.abs(pool_acts - mean_act[k]) + 1e9 * np.isin(
                np.arange(len(order_pool)), list(taken))))
            taken.add(j)
            match.append(order_pool[j])
        actmatch = sorted(match)
        print(f"actmatch: flagged mean act {mean_act[sig].mean():.4f} vs matched {mean_act[actmatch].mean():.4f}")

    def battery(classes, n_per, sets, tag):
        items, labels = [], []
        for c in classes:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][:n_per]
            items += take
            labels += [c] * len(take)
        labels = np.array(labels)
        caches = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
        res = {}
        base = preds_for(clf, caches, device, cfg, steer)
        res["baseline"] = round(float((base == labels).mean()), 3)
        per_class = {"base": base}
        for name, ks in sets:
            pred = preds_for(clf, caches, device, cfg, steer, proj=projector(Wd[:, ks]))
            res[name] = round(float((pred == labels).mean()), 3)
            per_class[name] = pred
            print(f"  [{tag}] {name}: {res[name]:.3f} (baseline {res['baseline']:.3f})")
        return res, labels, per_class

    # ---------------- erasure ----------------
    if args.stage == "erasure":
        all_classes = sorted(label2idx.values())
        if args.limit_classes > 0:
            all_classes = all_classes[:args.limit_classes]
        sets = [("identified", sig), ("rnd101", r101), ("rnd202", r202), ("static", static_set)]
        if actmatch:
            sets.append(("actmatch", actmatch))
        res_all, labels, pc = battery(all_classes, args.n_per_class, sets, "all-class")

        # per-class excess table -> family classes
        rows = []
        for c in sorted(set(labels.tolist())):
            m = labels == c
            base_a = float((pc["base"][m] == c).mean())
            nfp_a = float((pc["identified"][m] == c).mean())
            rnd_a = float((pc["rnd101"][m] == c).mean())
            rows.append({"cls": int(c), "base": round(base_a, 2), "nfp": round(nfp_a, 2),
                         "rnd": round(rnd_a, 2), "excess": round(rnd_a - nfp_a, 2)})
        rows.sort(key=lambda r: -r["excess"])
        fam_classes = [r["cls"] for r in rows if r["excess"] >= args.excess_bar]
        print(f"family classes (excess >= {args.excess_bar}): {len(fam_classes)}")
        for r in rows[:15]:
            print(f"   cls {r['cls']:3d} {id2label[r['cls']][:48]:<50} base={r['base']} nfp={r['nfp']} rnd={r['rnd']} excess={r['excess']}")

        res_fam = {}
        sweep = {}
        if fam_classes:
            res_fam, _, _ = battery(fam_classes, args.n_per_class, sets, "family")
            # size sweep (appendix): top-k by max |t| vs random-k
            ks_list = sorted(set([max(1, N // 4), max(1, N // 2), N]))
            items, labels_f = [], []
            for c in fam_classes:
                take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][:args.n_per_class]
                items += take
                labels_f += [c] * len(take)
            labels_f = np.array(labels_f)
            caches_f = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
            for k in ks_list:
                top = sorted(sig_ranked[:k])
                rndk = sorted(np.random.RandomState(101).choice(nonsig, k, replace=False))
                a_top = float((preds_for(clf, caches_f, device, cfg, steer,
                                         proj=projector(Wd[:, top])) == labels_f).mean())
                a_rnd = float((preds_for(clf, caches_f, device, cfg, steer,
                                         proj=projector(Wd[:, rndk])) == labels_f).mean())
                sweep[k] = {"topk": round(a_top, 3), "random_k": round(a_rnd, 3)}
                print(f"  [sweep] k={k}: topk={a_top:.3f} random={a_rnd:.3f}")

        json.dump({"all_class": res_all, "per_class_top": rows[:40],
                   "family_classes": fam_classes, "family": res_fam,
                   "size_sweep": sweep, "provenance": prov},
                  open(out / f"{fam}_erasure.json", "w"), indent=2)
        print(f"saved -> {fam}_erasure.json")
        return

    # ---------------- repair ----------------
    if args.stage == "repair":
        er = json.load(open(out / f"{fam}_erasure.json"))
        fam_classes = er["family_classes"]
        if not fam_classes:
            print("no family classes; skip repair")
            return
        wrong_items, wrong_lbls, right_items, right_lbls = [], [], [], []
        for c in fam_classes:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][:args.mine_per_class]
            if not take:
                continue
            caches = cache_frames(take, args.ssv2_videos, proc, cfg, cfg["batch"])
            pred = preds_for(clf, caches, device, cfg, steer)
            for i, it in enumerate(take):
                (wrong_items if pred[i] != c else right_items).append(it)
                (wrong_lbls if pred[i] != c else right_lbls).append(c)
        wl, rl = np.array(wrong_lbls), np.array(right_lbls)
        print(f"mined: wrong={len(wl)} right={len(rl)} over {len(fam_classes)} classes")
        wcache = cache_frames(wrong_items, args.ssv2_videos, proc, cfg, cfg["batch"])
        rcache = cache_frames(right_items, args.ssv2_videos, proc, cfg, cfg["batch"])
        base_acc = len(rl) / (len(rl) + len(wl))
        res = {"n_wrong": len(wl), "n_right": len(rl), "baseline": round(base_acc, 3), "net_acc": {}}
        sets = [("identified", torch.tensor(sig)),
                ("random", torch.tensor(r101)),
                ("static", torch.tensor(static_set))]
        for name, idx in sets:
            for a in args.alphas:
                pw = preds_for(clf, wcache, device, cfg, steer, patch=(idx, a)) if len(wl) else np.array([])
                pr = preds_for(clf, rcache, device, cfg, steer, patch=(idx, a)) if len(rl) else np.array([])
                repaired = int((pw == wl).sum())
                kept = int((pr == rl).sum())
                net = (repaired + kept) / (len(wl) + len(rl))
                res["net_acc"][f"{name}@a{a:g}"] = {
                    "repaired": repaired, "kept": kept, "net_acc": round(net, 3),
                    "delta": round(net - base_acc, 3)}
                print(f"  {name:<11} a={a:<4g} repaired={repaired}/{len(wl)} kept={kept}/{len(rl)} net={net:.3f} ({net-base_acc:+.3f})")
        res["provenance"] = prov
        json.dump(res, open(out / f"{fam}_repair.json", "w"), indent=2)
        print(f"saved -> {fam}_repair.json")
        return

    # ---------------- erasure over all classes (full per-class saving) --------
    if args.stage == "recon_acc":
        all_classes = sorted(label2idx.values())
        if args.limit_classes > 0:
            all_classes = all_classes[:args.limit_classes]
        items, labels = [], []
        for c in all_classes:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][: args.n_per_class]
            items += take; labels += [c] * len(take)
        labels = np.array(labels)
        caches = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
        base = preds_for(clf, caches, device, cfg, steer)
        steer.reconstruct = True
        rec = preds_for(clf, caches, device, cfg, steer)
        steer.reconstruct = False
        res = {"n": len(labels),
               "net_baseline": round(float((base == labels).mean()), 3),
               "net_recon": round(float((rec == labels).mean()), 3),
               "agree_with_baseline": round(float((rec == base).mean()), 3),
               "per_class_baseline": {str(c): round(float((base[labels == c] == c).mean()), 3)
                                       for c in all_classes},
               "per_class_recon": {str(c): round(float((rec[labels == c] == c).mean()), 3)
                                    for c in all_classes}}
        print(f"baseline={res['net_baseline']:.3f} recon={res['net_recon']:.3f} "
              f"agree={res['agree_with_baseline']:.3f}")
        res["provenance"] = prov
        json.dump(res, open(out / f"{fam}_recon_acc.json", "w"), indent=2)
        print(f"saved -> {fam}_recon_acc.json")
        return

    if args.stage == "erasure_allclass":
        all_classes = sorted(label2idx.values())
        if args.limit_classes > 0:
            all_classes = all_classes[:args.limit_classes]
        items, labels = [], []
        for c in all_classes:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][: args.n_per_class]
            items += take; labels += [c] * len(take)
        labels = np.array(labels)
        caches = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
        base = preds_for(clf, caches, device, cfg, steer)
        res = {"n": len(labels),
               "per_class_baseline": {str(c): round(float((base[labels == c] == c).mean()), 3)
                                       for c in all_classes},
               "net_baseline": round(float((base == labels).mean()), 3),
               "conds": {}}
        print(f"all-class n={len(labels)} baseline={res['net_baseline']:.3f}")
        ks_list = sorted(set([max(1, N // 4), max(1, N // 2)]))
        conds = [("identified", sig), ("rnd101", r101), ("rnd202", r202),
                 ("static", static_set)]
        if actmatch:
            conds.append(("actmatch", actmatch))
        for k in ks_list:
            conds.append((f"top{k}", sorted(sig_ranked[:k])))
            conds.append((f"rndk{k}", sorted(np.random.RandomState(101).choice(
                nonsig, k, replace=False))))
        for name, ks in conds:
            pred = preds_for(clf, caches, device, cfg, steer, proj=projector(Wd[:, ks]))
            res["conds"][name] = {
                "net": round(float((pred == labels).mean()), 3),
                "per_class": {str(c): round(float((pred[labels == c] == c).mean()), 3)
                              for c in all_classes}}
            print(f"  {name:<12} net={res['conds'][name]['net']:.3f}")
        res["provenance"] = prov
        json.dump(res, open(out / f"{fam}_erasure_allclass{nrsfx}.json", "w"), indent=2)
        print(f"saved -> {fam}_erasure_allclass{nrsfx}.json")
        return

    # ---------------- amplification over all classes ----------------
    if args.stage == "amp_allclass":
        all_classes = sorted(label2idx.values())
        if args.limit_classes > 0:
            all_classes = all_classes[:args.limit_classes]
        items, labels = [], []
        for c in all_classes:
            take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][: args.n_per_class]
            items += take; labels += [c] * len(take)
        labels = np.array(labels)
        caches = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
        base = preds_for(clf, caches, device, cfg, steer)
        pcb = {int(c): float((base[labels == c] == c).mean()) for c in all_classes}
        res = {"n": len(labels), "baseline_net": round(float((base == labels).mean()), 3),
               "per_class_baseline": {str(c): round(v, 3) for c, v in pcb.items()},
               "sets": {}}
        print(f"all-class n={len(labels)} baseline={res['baseline_net']:.3f}")
        set_list = [("identified", torch.tensor(sig)),
                    ("random", torch.tensor(r101)),
                    ("static", torch.tensor(static_set))]
        for name, idx in set_list:
            res["sets"][name] = {}
            for a in args.alphas:
                pred = preds_for(clf, caches, device, cfg, steer, patch=(idx, a))
                pcls = {int(c): float((pred[labels == c] == c).mean()) for c in all_classes}
                res["sets"][name][f"{a:g}"] = {
                    "net_acc": round(float((pred == labels).mean()), 3),
                    "per_class": {str(c): round(v, 3) for c, v in pcls.items()}}
                print(f"  {name:<11} a={a:<4g} net={res['sets'][name][f'{a:g}']['net_acc']:.3f}")
        res["provenance"] = prov
        json.dump(res, open(out / f"{fam}_amp_allclass{nrsfx}.json", "w"), indent=2)
        print(f"saved -> {fam}_amp_allclass{nrsfx}.json")
        return

    # ---------------- repair addons (macro, per-class, random-class ablation) ----
    if args.stage == "repair_addons":
        er = json.load(open(out / f"{fam}_erasure.json"))
        fam_classes = er["family_classes"]
        relaxed = False
        if not fam_classes:
            fam_classes = [r["cls"] for r in er["per_class_top"]
                           if r["excess"] >= args.relaxed_bar]
            relaxed = True
            print(f"family empty; relaxed bar {args.relaxed_bar} -> {len(fam_classes)} classes")
        idx_sig = torch.tensor(sig)

        def evaluate(classes, tag):
            items, labels = [], []
            for c in classes:
                take = [it for it in val if label2idx.get(it.get("template", ""), -1) == c][: args.mine_per_class]
                items += take; labels += [c] * len(take)
            labels = np.array(labels)
            caches = cache_frames(items, args.ssv2_videos, proc, cfg, cfg["batch"])
            base = preds_for(clf, caches, device, cfg, steer)
            base_acc = float((base == labels).mean())
            pcb = {int(c): float((base[labels == c] == c).mean()) for c in classes}
            print(f"[{tag}] n={len(labels)} baseline={base_acc:.3f}")
            outr = {"n": len(labels), "baseline": round(base_acc, 3),
                    "per_class_baseline": {str(c): round(v, 3) for c, v in pcb.items()},
                    "alphas": {}}
            for a in args.alphas:
                pred = preds_for(clf, caches, device, cfg, steer, patch=(idx_sig, a))
                net = float((pred == labels).mean())
                pcls = {int(c): float((pred[labels == c] == c).mean()) for c in classes}
                macro = float(np.mean(list(pcls.values())))
                deltas = {c: pcls[c] - pcb[c] for c in pcls}
                bc = max(deltas, key=deltas.get); wc = min(deltas, key=deltas.get)
                outr["alphas"][f"{a:g}"] = {
                    "net_acc": round(net, 3), "macro_acc": round(macro, 3),
                    "per_class": {str(c): round(v, 3) for c, v in pcls.items()},
                    "best_class": {"cls": bc, "label": id2label[bc],
                                   "base": round(pcb[bc], 3), "steered": round(pcls[bc], 3)},
                    "worst_class": {"cls": wc, "label": id2label[wc],
                                    "base": round(pcb[wc], 3), "steered": round(pcls[wc], 3)}}
                print(f"  a={a:<4g} net={net:.3f} macro={macro:.3f} "
                      f"best {id2label[bc][:24]} {pcb[bc]:.2f}->{pcls[bc]:.2f} "
                      f"worst {id2label[wc][:24]} {pcb[wc]:.2f}->{pcls[wc]:.2f}")
            return outr

        res = {"relaxed": relaxed, "family_classes": fam_classes,
               "family": evaluate(fam_classes, "family")}
        by_cls_ids = sorted(set(label2idx.get(it.get("template", ""), -1) for it in val) - {-1})
        pool = [c for c in by_cls_ids if c not in set(fam_classes)]
        rnd_classes = sorted(np.random.RandomState(args.seed + 11).choice(
            pool, len(fam_classes), replace=False).tolist())
        res["random_classes"] = evaluate(rnd_classes, "random classes")
        res["random_class_ids"] = rnd_classes
        res["provenance"] = prov
        json.dump(res, open(out / f"{fam}_repair_addons.json", "w"), indent=2)
        print(f"saved -> {fam}_repair_addons.json")
        return

    # ---------------- pairs ----------------
    if args.stage == "pairs":
        pair_list = PAIRS10
        if args.pairs_json:
            pair_list = [tuple(x) for x in json.load(open(args.pairs_json))["pairs"]]
            print(f"pairs from {args.pairs_json}: {[x[0] for x in pair_list]}")
        pair_data = []
        for key, axis, pos_l, neg_l in pair_list:
            cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
            if cp < 0 or cn < 0:
                print(f"pair {key}: label missing, skipped")
                continue
            pos_items = [it for it in val if label2idx.get(it.get("template", ""), -1) == cp][:args.per_class_pairs]
            neg_items = [it for it in val if label2idx.get(it.get("template", ""), -1) == cn][:args.per_class_pairs]
            pair_data.append(dict(
                key=key, axis=axis, cp=cp, cn=cn,
                pos=cache_frames(pos_items, args.ssv2_videos, proc, cfg, cfg["batch"]),
                neg=cache_frames(neg_items, args.ssv2_videos, proc, cfg, cfg["batch"]),
                n_pos=len(pos_items), n_neg=len(neg_items)))
        print(f"pairs ready: {[p['key'] for p in pair_data]}")

        def screen(feature_ids, tag):
            rows = []
            for fi, k in enumerate(feature_ids):
                rec = {"feature": f"feat{k:05d}", "idx": int(k), "pairs": {}}
                for p in pair_data:
                    P = {}
                    for side, cache in [("pos", p["pos"]), ("neg", p["neg"])]:
                        for s in (+args.s_abs, -args.s_abs):
                            steer.enabled, steer.k, steer.s = True, k, s
                            outs = []
                            for inputs in cache:
                                outs.append(forward_logits(clf, inputs, device, cfg).cpu())
                            steer.enabled = False
                            P[(side, s)] = torch.cat(outs, 0)
                    def rates(orient):
                        sp, sn = (+args.s_abs, -args.s_abs) if orient == "A" else (-args.s_abs, +args.s_abs)
                        lp, ln = P[("neg", sp)], P[("pos", sn)]
                        f_np = float((lp[:, p["cp"]] > lp[:, p["cn"]]).float().mean())
                        f_pn = float((ln[:, p["cn"]] > ln[:, p["cp"]]).float().mean())
                        t_np = float((lp.argmax(1) == p["cp"]).float().mean())
                        t_pn = float((ln.argmax(1) == p["cn"]).float().mean())
                        return (f_np + f_pn) / 2, (t_np + t_pn) / 2
                    fA, tA = rates("A")
                    fB, tB = rates("B")
                    rec["pairs"][p["key"]] = {"rank_flip": round(max(fA, fB), 3),
                                              "top1_flip": round(max(tA, tB), 3)}
                rows.append(rec)
                best = max(rec["pairs"].items(), key=lambda kv: kv[1]["top1_flip"])
                print(f"  [{tag}] {rec['feature']} ({fi+1}/{len(feature_ids)}) best {best[0]} top1={best[1]['top1_flip']} rank={best[1]['rank_flip']}")
            return rows

        sig_screen = sig_ranked[:args.pairs_topk] if args.pairs_topk else sig
        static_screen = list(static_set)[:args.pairs_topk] if args.pairs_topk else static_set
        rows_sig = [] if args.static_only else screen(sig_screen, "identified")
        rows_static = screen(static_screen, "static")
        suffix = f"_s{st_seed}" if args.static_seed >= 0 else ""
        if args.pairs_tag:
            suffix = f"_{args.pairs_tag}" + suffix
        json.dump({"s_abs": args.s_abs, "per_class": args.per_class_pairs,
                   "static_seed": st_seed,
                   "pairs": [{k: p[k] for k in ["key", "axis", "cp", "cn", "n_pos", "n_neg"]} for p in pair_data],
                   "identified": rows_sig, "static": rows_static, "provenance": prov},
                  open(out / f"{fam}_pairs{suffix}.json", "w"), indent=2)
        print(f"saved -> {fam}_pairs{suffix}.json")
        return


if __name__ == "__main__":
    main()
