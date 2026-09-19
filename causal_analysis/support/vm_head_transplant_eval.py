"""Evaluate the finetuned VideoMAE head transplanted onto frozen base features.

The finetuned classifier applies fc_norm (LayerNorm) then a linear layer to
the mean-pooled encoder output. The extracted base features are the
mean-pooled encoder output of the checkpoint without label finetuning, so the
head can be applied directly. Compares against the trained linear probe on
the same held-out split (the first rows, matching vm_base_probe.py).
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_feats", default="D:/vmpre_work/vm_base_train_feats.pt")
    ap.add_argument("--probe", default="local_runs/expansion/vm_base_probe.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    from transformers import VideoMAEForVideoClassification

    ft = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
    label2idx = {v: int(k) for k, v in ft.config.id2label.items()}
    fc_norm = ft.fc_norm.to(device).eval() if ft.fc_norm is not None else None
    head = ft.classifier.to(device).eval()

    tr = torch.load(args.train_feats, map_location="cpu", weights_only=False)
    X = tr["feats"]
    y = torch.tensor([label2idx[t] for t in tr["templates"]])
    n_val = max(2000, int(0.02 * len(y)))
    Xv, yv = X[:n_val].to(device), y[:n_val].to(device)

    with torch.no_grad():
        z = fc_norm(Xv) if fc_norm is not None else Xv
        acc_t = (head(z).argmax(-1) == yv).float().mean().item()
    print(
        f"transplanted finetuned head on base features: {acc_t:.4f} (n={n_val}, chance {1/174:.4f})"
    )

    try:
        pw = torch.load(args.probe, map_location="cpu", weights_only=False)
        lin = torch.nn.Linear(pw["weight"].shape[1], pw["weight"].shape[0]).to(device)
        lin.weight.data = pw["weight"].to(device)
        lin.bias.data = pw["bias"].to(device)
        with torch.no_grad():
            acc_p = (lin(Xv).argmax(-1) == yv).float().mean().item()
        print(f"trained linear probe on base features:        {acc_p:.4f}")
    except FileNotFoundError:
        print("probe not trained yet")


if __name__ == "__main__":
    main()
