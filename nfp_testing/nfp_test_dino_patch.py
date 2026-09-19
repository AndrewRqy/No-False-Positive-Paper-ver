"""NFP Temporal Covariance Test - DINOv2 SAE negative control.

Runs the NFP temporal covariance test on a DINOv2 SAE as a (corrected) negative
control against the VideoMAE SAE.

Uses the same ball-tracking extraction as nfp_test.py:
  psi_i(V, t) = SAE feature i applied to the DINOv2 spatial patch token
                at the ball-containing grid position at temporal step t,
                where DINOv2 processes each frame independently.

This makes the comparison with VideoMAE SAE methodologically clean:
  - VideoMAE: spatial ball token from cross-frame representation
  - DINOv2: spatial ball token from single-frame representation
Both use the same grid (14x14, same as VideoMAE's spatial layout).

Expected result: 0 significant features, since DINOv2 has no temporal context.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder, PCADict, ICADict, IdentityDict

TAU_KEYS = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_TEMPORAL = 8  # 8 temporal steps (one frame each = frame 2t)
N_FRAMES = 16


# ---------------------------------------------------------------------------
# Dataset - returns frames, tau, and ball spatial token index per step
# ---------------------------------------------------------------------------


class NFPDataset(Dataset):
    """Dataset of NFP ball videos for the per-frame DINOv2 control.

    Each item exposes one representative frame per temporal step (the first frame
    of each 2-frame tubelet), the tau sequence, and the ball spatial token index
    per step, so DINOv2 can process each frame independently.

    Attributes:
        video_dirs: Sorted list of per-video directories under ``root_dir``.
    """

    def __init__(self, root_dir: Path) -> None:
        """Index the video directories under the dataset root.

        Args:
            root_dir: Directory containing the per-video subdirectories ("v*").

        Raises:
            FileNotFoundError: If no video directories are found under root_dir.
        """
        self.video_dirs = sorted(root_dir.glob("v*"))
        if not self.video_dirs:
            raise FileNotFoundError(f"No video dirs found in {root_dir}")

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.video_dirs)

    def __getitem__(self, idx: int) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor, str]:
        """Load one video's representative frames, tau, and ball tokens.

        Args:
            idx: Index into the sorted list of video directories.

        Returns:
            A tuple of (rep_frames, tau_steps, ball_tok_steps, video_id) where
            rep_frames is a list of N_TEMPORAL PIL images (one per step),
            tau_steps is a [8, 5] float tensor, ball_tok_steps is an [8] long
            tensor (-1 off-screen), and video_id is the video identifier string.
        """
        vdir = self.video_dirs[idx]
        # One frame per temporal step (first frame of each 2-frame tubelet)
        rep_frames = [
            Image.open(vdir / f"rgba_{step * 2:05d}.png").convert("RGB")
            for step in range(N_TEMPORAL)
        ]
        with open(vdir / "metadata.json") as f:
            meta = json.load(f)

        traj = meta["trajectory"]
        tau_steps = []
        ball_tok_steps = []
        for step in range(N_TEMPORAL):
            frame_rec = traj[step * 2]
            tau_steps.append([frame_rec["tau"][k] for k in TAU_KEYS])
            ball_tok_steps.append(frame_rec["spatial_token"])  # -1 if off-screen

        return (
            rep_frames,
            torch.tensor(tau_steps, dtype=torch.float32),  # [8, 5]
            torch.tensor(ball_tok_steps, dtype=torch.long),  # [8]
            meta["video_id"],
        )


def make_collate(
    processor: Any,
) -> Callable[[List[Any]], Tuple[Any, torch.Tensor, torch.Tensor, List[str]]]:
    """Build a DataLoader collate function for the DINOv2 control.

    The per-video frame lists are flattened into a single image batch (DINOv2
    processes each frame independently) while tau and ball-token tensors are
    stacked per video.

    Args:
        processor: The DINOv2 image processor used to tensorize frames.

    Returns:
        A collate function mapping a batch to (inputs, tau, ball_tokens,
        video_ids).
    """

    def collate(
        batch: List[Any],
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, List[str]]:
        all_frames = []
        tau_lst = []
        ball_tok_lst = []
        video_ids = []
        for frames, tau, ball_toks, vid_id in batch:
            all_frames.extend(frames)
            tau_lst.append(tau)
            ball_tok_lst.append(ball_toks)
            video_ids.append(vid_id)
        inputs = processor(images=all_frames, return_tensors="pt")
        tau_tensor = torch.stack(tau_lst)  # [B, 8, 5]
        ball_toks_tensor = torch.stack(ball_tok_lst)  # [B, 8]
        return inputs, tau_tensor, ball_toks_tensor, video_ids

    return collate


# ---------------------------------------------------------------------------
# Within-video covariance
# ---------------------------------------------------------------------------


def within_video_covariance_all(feats: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Compute the within-video covariance of each feature with each tau.

    Args:
        feats: Ball-tracking SAE activations of shape [B, T, D].
        tau: All tau variables of shape [B, T, K].

    Returns:
        The within-video covariance tensor C of shape [B, D, K].
    """
    psi_c = feats - feats.mean(dim=1, keepdim=True)
    tau_c = tau - tau.mean(dim=1, keepdim=True)
    return torch.einsum("btd,btk->bdk", psi_c, tau_c) / feats.shape[1]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(
    t_stat: np.ndarray, p_val: np.ndarray, C_mean: np.ndarray, alpha: float = 0.05
) -> None:
    """Print the DINOv2-control NFP report.

    Args:
        t_stat: Per-feature, per-tau t-statistics of shape [D, K].
        p_val: Per-feature, per-tau p-values of shape [D, K].
        C_mean: Per-feature, per-tau mean covariance of shape [D, K].
        alpha: Family-wise significance level before Bonferroni correction.
    """
    D, K = t_stat.shape
    bonf = alpha / D
    print(f"\n{'='*68}")
    print(f"NFP Test — DINOv2 SAE, spatial ball token  " f"(Bonferroni p < {bonf:.2e})")
    print(f"Expected: 0% significant features for all tau variables")
    print(f"{'='*68}")
    print(f"{'Tau':<12} {'Sig+':>6} {'Sig-':>6} {'Total%':>8} {'Mean|t|':>9}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*9}")

    sig_any = np.zeros(D, dtype=bool)
    for k, name in enumerate(TAU_KEYS):
        t_k = t_stat[:, k]
        p_k = p_val[:, k]
        sp = ((p_k < bonf) & (t_k > 0)).sum()
        sn = ((p_k < bonf) & (t_k < 0)).sum()
        sig = (p_k < bonf).sum()
        sig_any |= p_k < bonf
        print(f"{name:<12} {sp:>6} {sn:>6} {100*sig/D:>7.2f}%" f" {np.abs(t_k).mean():>9.4f}")

    print(
        f"\nFeatures significant for at least one tau : "
        f"{sig_any.sum()} ({100*sig_any.mean():.2f}%)"
    )
    print(
        f"Features non-significant for all taus     : "
        f"{(~sig_any).sum()} ({100*(~sig_any).mean():.2f}%)"
    )
    print(f"Overall mean |C_i|                        : " f"{np.abs(C_mean).mean():.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the DINOv2 control NFP test.

    Returns:
        The parsed argparse namespace.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument(
        "--sae_path",
        default=None,
        help="Dictionary checkpoint. Not needed for --sae_model identity.",
    )
    p.add_argument(
        "--sae_model",
        default="standard",
        choices=["standard", "pca", "ica", "identity"],
        help="standard=ReLU SAE; pca/ica=linear decomposition fit on DINO "
        "patch activations; identity=raw DINO patch dims (no transform).",
    )
    p.add_argument(
        "--decomp_mode",
        default="sign_split",
        choices=["sign_split", "abs", "signed"],
        help="For pca/ica: how signed components map to features.",
    )
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_name", default="dinov2-base")
    p.add_argument(
        "--layer",
        default=-1,
        type=int,
        help="encoder layer to read (-1 = final, i.e. last_hidden_state)",
    )
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--alpha", default=0.05, type=float)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    """Run the DINOv2-control NFP test and save the result tensors.

    Loads DINOv2 and the selected dictionary, extracts the ball-containing
    spatial patch token per independently-processed frame, computes the
    within-video covariance with each tau, runs a one-sample t-test with
    Bonferroni correction, prints the report, and saves the statistics.
    """
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading DINOv2: facebook/{args.model_name}")
    processor = AutoImageProcessor.from_pretrained(f"facebook/{args.model_name}")
    model = Dinov2Model.from_pretrained(f"facebook/{args.model_name}").to(device)
    model.eval()

    print(f"Loading dictionary ({args.sae_model}): {args.sae_path}")
    if args.sae_model == "standard":
        sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    elif args.sae_model == "pca":
        sae = PCADict.from_pretrained(args.sae_path, device=device, mode=args.decomp_mode)
    elif args.sae_model == "ica":
        sae = ICADict.from_pretrained(args.sae_path, device=device, mode=args.decomp_mode)
    elif args.sae_model == "identity":
        sae = IdentityDict.from_pretrained(None)
    else:
        raise ValueError(f"Unknown sae_model: {args.sae_model}")
    sae.eval()

    ds = NFPDataset(Path(args.dataset_dir))
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(processor),
    )

    print(f"Dataset: {len(ds)} videos | Testing all {len(TAU_KEYS)} tau variables")
    print("Each temporal step uses one independently-processed frame (no temporal context).")
    print("Activation extracted at the ball-containing spatial patch token.")

    all_C = []
    all_tau = []
    all_ids = []

    for inputs, tau, ball_tokens, video_ids in tqdm(dl, desc="Extracting"):
        B = tau.shape[0]
        with torch.no_grad():
            out = model(
                **{k: v.to(device) for k, v in inputs.items()}, output_hidden_states=args.layer >= 0
            )
            if args.layer >= 0:
                # hidden_states[0] is the embedding output, so layer i of the
                # encoder is hidden_states[i + 1]
                hidden = out.hidden_states[args.layer + 1]
            else:
                hidden = out.last_hidden_state  # [B*8, 197, 768]

        # ball_tokens: [B, 8], flatten to [B*8]
        flat_toks = ball_tokens.reshape(-1).to(device)  # [B*8]
        mask = flat_toks >= 0  # True = on-screen
        safe_toks = flat_toks.clamp(min=0)  # clamp -1 → 0 (off-screen fallback)

        # CLS token is at index 0; spatial patch tokens start at index 1
        patch_indices = safe_toks + 1  # [B*8]
        batch_range = torch.arange(B * N_TEMPORAL, device=device)

        with torch.no_grad():
            ball_embeds = hidden[batch_range, patch_indices, :]  # [B*8, 768]
            ball_embeds = ball_embeds * mask.float().unsqueeze(-1)  # zero off-screen
            sae_out = sae.encode(ball_embeds)  # [B*8, D]

        feats = sae_out.reshape(B, N_TEMPORAL, -1).cpu()  # [B, 8, D]
        mask_2d = mask.reshape(B, N_TEMPORAL).float().cpu()
        feats = feats * mask_2d.unsqueeze(-1)  # zero off-screen

        C = within_video_covariance_all(feats, tau)  # [B, D, 5]
        all_C.append(C)
        all_tau.append(tau)
        all_ids.extend(video_ids)

    C_all = torch.cat(all_C, dim=0)  # [N, D, 5]
    tau_all = torch.cat(all_tau, dim=0)  # [N, 8, 5]

    print(f"C tensor: {C_all.shape}  Running t-tests...")
    C_np = C_all.numpy()
    t_stat = np.zeros((C_np.shape[1], len(TAU_KEYS)), dtype=np.float32)
    p_val = np.ones_like(t_stat)
    for k in range(len(TAU_KEYS)):
        t_stat[:, k], p_val[:, k] = stats.ttest_1samp(C_np[:, :, k], 0.0)

    C_mean = C_np.mean(axis=0)
    print_report(t_stat, p_val, C_mean, alpha=args.alpha)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "C": C_all,
            "tau": tau_all,
            "t_stat": torch.from_numpy(t_stat),
            "p_val": torch.from_numpy(p_val),
            "C_mean": torch.from_numpy(C_mean),
            "video_ids": all_ids,
            "tau_keys": TAU_KEYS,
            "model": f"dinov2_{args.model_name}_patch_tokens",
        },
        out_path,
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
