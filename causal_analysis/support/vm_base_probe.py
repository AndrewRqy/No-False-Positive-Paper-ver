"""Linear probe for VideoMAE without label finetuning.

The base checkpoint has no classification head, so causal analysis needs a
readout. Following the V-JEPA2 pattern (frozen encoder, trained readout), we
train a linear probe on the mean-pooled final-layer activations, which
matches the architecture of the finetuned model's own head.

Modes:
  extract  decode SSv2 clips, mean-pool VideoMAEModel last_hidden_state,
           save float32 feature chunks with labels
  train    fit a 174-way softmax linear probe on the saved features
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from causal_analysis.common.steer_pair_screen import ItemFrames  # noqa: E402


class Collate:
    """Module-level and picklable so Windows dataloader workers can spawn."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, batch):
        imgs = [b[0] for b in batch]
        inputs = self.proc(images=imgs, return_tensors="pt")
        return inputs, [b[2] for b in batch]


def extract(args, device):
    from transformers import VideoMAEModel, VideoMAEImageProcessor

    model = VideoMAEModel.from_pretrained(args.model_name).to(device).eval()
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    items = json.load(open(args.items_json, encoding="utf-8"))
    if args.max_clips > 0:
        items = items[: args.max_clips]
    print(f"extracting {len(items)} clips")
    dl = DataLoader(
        ItemFrames(args.ssv2_videos, items),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collate(proc),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats, tpls = [], []
    done = 0
    for inputs, batch_tpls in dl:
        pv = inputs["pixel_values"].to(device)
        with torch.no_grad():
            h = model(pixel_values=pv).last_hidden_state
            feats.append(h.mean(1).float().cpu())
        tpls += batch_tpls
        done += pv.shape[0]
        if done % 4000 < args.batch:
            print(f"  {done}/{len(items)}", flush=True)
    torch.save({"feats": torch.cat(feats, 0), "templates": tpls}, out)
    print(f"saved {done} features -> {out}")


def train(args, device):
    from transformers import VideoMAEForVideoClassification

    ref = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
    label2idx = {v: int(k) for k, v in ref.config.id2label.items()}
    del ref
    tr = torch.load(args.train_feats, map_location="cpu", weights_only=False)
    X = tr["feats"]
    y = torch.tensor([label2idx[t] for t in tr["templates"]])
    n_val = max(2000, int(0.02 * len(y)))
    Xv, yv = X[:n_val].to(device), y[:n_val].to(device)
    Xt, yt = X[n_val:].to(device), y[n_val:].to(device)
    lin = torch.nn.Linear(X.shape[1], 174).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), 8192):
            idx = perm[i : i + 8192]
            loss = torch.nn.functional.cross_entropy(lin(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (lin(Xv).argmax(-1) == yv).float().mean().item()
        print(f"epoch {ep + 1}: heldout acc {acc:.4f}")
    torch.save(
        {"weight": lin.weight.detach().cpu(), "bias": lin.bias.detach().cpu(), "heldout_acc": acc},
        args.probe_out,
    )
    print(f"probe saved -> {args.probe_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["extract", "train"])
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-ssv2")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--items_json", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max_clips", default=0, type=int)
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--num_workers", default=6, type=int)
    ap.add_argument("--train_feats", default="")
    ap.add_argument("--probe_out", default="local_runs/expansion/vm_base_probe.pt")
    ap.add_argument("--epochs", default=30, type=int)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.mode == "extract":
        extract(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
