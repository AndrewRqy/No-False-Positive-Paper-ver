"""Encode video clips using DINOv2 (max-pooled over frames) for monosemanticity metric."""
import torch
import os
import argparse
import tqdm
from utils.datasets.ssv2 import SSv2Dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, Dinov2Model


def get_args_parser():
    parser = argparse.ArgumentParser("Encode video clips with DINOv2", add_help=False)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--model_name", default="dinov2-base", type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--max_clips", default=-1, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if os.path.exists(args.embeddings_path):
        print(f"Embeddings already saved at {args.embeddings_path}")
        exit(0)

    processor = AutoImageProcessor.from_pretrained(f"facebook/{args.model_name}")
    model = Dinov2Model.from_pretrained(f"facebook/{args.model_name}").to(args.device)
    model.eval()

    ds = SSv2Dataset(args.data_path, split=args.split)
    if args.max_clips > 0:
        ds = Subset(ds, list(range(min(args.max_clips, len(ds)))))

    def collate_fn(batch):
        all_frames = []
        clip_sizes = []
        for frames_list, _ in batch:
            all_frames.extend(frames_list)
            clip_sizes.append(len(frames_list))
        inputs = processor(images=all_frames, return_tensors="pt")
        return inputs, clip_sizes

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    embeddings = []
    for inputs, clip_sizes in tqdm.tqdm(dl, desc="Encoding clips"):
        with torch.no_grad():
            outputs = model(**{k: v.to(args.device) for k, v in inputs.items()})
            frame_embeds = outputs.pooler_output  # [B*T, embed_dim]

        split_embeds = torch.split(frame_embeds, clip_sizes)
        clip_embeds = torch.stack([e.max(dim=0).values for e in split_embeds])
        embeddings.append(clip_embeds.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.embeddings_path)), exist_ok=True)
    torch.save(embeddings, args.embeddings_path)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to {args.embeddings_path}")
