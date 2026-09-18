"""
Extract DINOv2 spatial patch token activations from SSv2 for SAE training.

For each SSv2 video, processes each of the 16 sampled frames independently
through DINOv2 and saves all 196 spatial patch tokens (CLS skipped) as
flat 768-dim vectors. Output files match the format expected by sae_train.py.

Usage:
    python training/extract_dino_patch_activations.py \
        --data_path /net/scratch/renqy/SSv2 \
        --output_dir /net/scratch2/renqy/dino_patch_activations/train \
        --split train \
        --max_videos 4000 \
        --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Dinov2Model
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.datasets.ssv2 import SSv2Dataset

FRAMES_PER_VIDEO = 16
PATCHES_PER_FRAME = 196  # 14×14 grid for 224×224 input with 16×16 patches
HIDDEN_DIM = 768


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True, help="SSv2 dataset root")
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--split",       default="train", choices=["train", "val"])
    p.add_argument("--max_videos",  default=-1, type=int,
                   help="Limit number of videos (-1 = all)")
    p.add_argument("--batch_size",  default=8, type=int,
                   help="Videos per GPU batch (each gives 16 frames)")
    p.add_argument("--save_every",  default=50000, type=int,
                   help="Number of patch tokens per output file")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--device",      default="cuda:0")
    return p.parse_args()


def collate_fn(processor):
    def _collate(batch):
        # batch: list of (frames_list, label)
        all_frames = []
        for frames, _ in batch:
            all_frames.extend(frames)
        inputs = processor(images=all_frames, return_tensors="pt")
        return inputs, len(batch)
    return _collate


def main():
    args = get_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv2 (facebook/dinov2-base)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    print(f"Loading SSv2 {args.split} split from {args.data_path}")
    ds = SSv2Dataset(args.data_path, split=args.split,
                     num_frames=FRAMES_PER_VIDEO)
    if args.max_videos > 0:
        ds.samples = ds.samples[:args.max_videos]
    print(f"Using {len(ds)} videos")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_fn(processor))

    accumulated = []
    n_accumulated = 0
    chunk_idx = 0
    total_videos = 0

    for inputs, batch_n_videos in tqdm(dl, desc="Extracting patch tokens"):
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in inputs.items()})
            # last_hidden_state: [B*16, 197, 768]
            # Index 0 = CLS; 1:197 = spatial patch tokens
            patch_tokens = out.last_hidden_state[:, 1:, :]  # [B*16, 196, 768]
            flat = patch_tokens.reshape(-1, HIDDEN_DIM).cpu()  # [B*16*196, 768]

        accumulated.append(flat)
        n_accumulated += flat.shape[0]
        total_videos += batch_n_videos

        # Flush when we have enough tokens
        while n_accumulated >= args.save_every:
            buf = torch.cat(accumulated, dim=0)
            chunk = buf[:args.save_every]
            torch.save(chunk, out_dir / f"activations_part{chunk_idx}.pt")
            print(f"  Saved chunk {chunk_idx}: {chunk.shape}  "
                  f"(videos processed so far: {total_videos})")
            chunk_idx += 1
            remainder = buf[args.save_every:]
            accumulated = [remainder] if remainder.shape[0] > 0 else []
            n_accumulated = remainder.shape[0] if remainder.shape[0] > 0 else 0

    # Save final partial chunk
    if n_accumulated > 0:
        buf = torch.cat(accumulated, dim=0)
        torch.save(buf, out_dir / f"activations_part{chunk_idx}.pt")
        print(f"  Saved final chunk {chunk_idx}: {buf.shape}")
        chunk_idx += 1

    print(f"\nDone. Processed {total_videos} videos → "
          f"{total_videos * FRAMES_PER_VIDEO * PATCHES_PER_FRAME:,} patch tokens "
          f"in {chunk_idx} chunk(s).")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
