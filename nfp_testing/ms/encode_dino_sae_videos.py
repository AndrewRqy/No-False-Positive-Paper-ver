"""Extract video-level DINOv2 SAE activations (mean-pooled over frames) for monosemanticity metric."""
import torch
import os
import argparse
import tqdm
from pathlib import Path
from utils.datasets.ssv2 import SSv2Dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Dinov2Model
from dictionary_learning import AutoEncoder


def get_args_parser():
    parser = argparse.ArgumentParser("Encode DINOv2 SAE activations at video level")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sae_path", required=True)
    parser.add_argument("--model_name", default="dinov2-base")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--save_every", default=5000, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(f"facebook/{args.model_name}")
    model = Dinov2Model.from_pretrained(f"facebook/{args.model_name}").to(args.device)
    model.eval()
    sae = AutoEncoder.from_pretrained(args.sae_path).to(args.device)
    sae.eval()

    ds = SSv2Dataset(args.data_path, split=args.split)

    def collate_fn(batch):
        all_frames, n_frames = [], []
        for frames_list, _ in batch:
            all_frames.extend(frames_list)
            n_frames.append(len(frames_list))
        inputs = processor(images=all_frames, return_tensors="pt")
        return inputs, n_frames

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    buffer, save_count, total = [], 0, 0

    for inputs, n_frames in tqdm.tqdm(dl, desc="Encoding videos"):
        with torch.no_grad():
            frame_embeds = model(**{k: v.to(args.device) for k, v in inputs.items()}).pooler_output
            sae_feats = sae.encode(frame_embeds)  # [B*T, n_features]

        # Mean-pool SAE features over frames to get one vector per video
        video_feats = []
        offset = 0
        for nf in n_frames:
            video_feats.append(sae_feats[offset:offset + nf].mean(dim=0))
            offset += nf
        buffer.append(torch.stack(video_feats).cpu())
        total += len(n_frames)

        if total >= args.save_every * (save_count + 1):
            chunk = torch.cat(buffer, dim=0)
            fname = f"ssv2_{args.split}_activations_dino_sae_part{save_count + 1}.pt"
            torch.save(chunk, os.path.join(args.output_dir, fname))
            print(f"Saved {total} videos → {fname}")
            buffer, save_count = [], save_count + 1

    if buffer:
        chunk = torch.cat(buffer, dim=0)
        fname = f"ssv2_{args.split}_activations_dino_sae_part{save_count + 1}.pt"
        torch.save(chunk, os.path.join(args.output_dir, fname))
        print(f"Saved {total} videos total")
