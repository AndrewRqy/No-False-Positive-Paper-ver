import os
import json
import io
import tarfile
import av
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


def _sample_frames(source, num_frames):
    """Decode num_frames evenly-spaced frames from a video file path or bytes."""
    if isinstance(source, (str, os.PathLike)):
        container = av.open(str(source))
    else:
        container = av.open(io.BytesIO(source))

    stream = container.streams.video[0]
    total = stream.frames

    frames = []
    if total > 0:
        indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_image())
            if len(frames) == num_frames:
                break
    else:
        # Frame count not reported — decode everything then sample
        all_frames = [f.to_image() for f in container.decode(video=0)]
        if all_frames:
            indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
            frames = [all_frames[i] for i in indices]

    container.close()

    # Pad to num_frames with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new("RGB", (224, 224)))

    return frames[:num_frames]


class SSv2Dataset(Dataset):
    """
    Fast map-style dataset reading from individually extracted .webm files.
    Requires running extract.py first to unpack the tar.gz.

    Expected layout:
        data_path/
            videos/          <-- extracted .webm files
            raw/20bn-something-something-download-package-labels/labels/
                train.json
                validation.json
    """

    def __init__(self, data_path, split="train", num_frames=16):
        self.videos_dir = os.path.join(data_path, "videos")
        self.num_frames = num_frames

        # SSv2 uses "validation.json" for the val split
        label_split = "validation" if split == "val" else split
        label_file = os.path.join(
            data_path,
            "raw",
            "20bn-something-something-download-package-labels",
            "labels",
            f"{label_split}.json",
        )
        with open(label_file, encoding="utf-8") as f:
            annotations = json.load(f)

        self.samples = [
            ann["id"]
            for ann in annotations
            if os.path.exists(os.path.join(self.videos_dir, f"{ann['id']}.webm"))
        ]

        missing = len(annotations) - len(self.samples)
        if missing:
            print(f"[SSv2Dataset] Warning: {missing} videos missing from {self.videos_dir}")
        print(f"[SSv2Dataset] {split}: {len(self.samples)} videos loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id = self.samples[idx]
        path = os.path.join(self.videos_dir, f"{video_id}.webm")
        try:
            frames = _sample_frames(path, self.num_frames)
        except Exception as e:
            print(f"[SSv2Dataset] Failed to decode {path}: {e}")
            frames = [Image.new("RGB", (224, 224))] * self.num_frames
        return frames, 0


class SSv2TarDataset(IterableDataset):
    """
    Slower fallback that streams directly from the tar.gz without extracting.
    Use SSv2Dataset (above) when possible — it's significantly faster.
    Only supports shuffle=False and num_workers=0.
    """

    def __init__(self, tar_path, annotation_file, num_frames=16):
        self.tar_path = tar_path
        self.num_frames = num_frames

        with open(annotation_file, encoding="utf-8") as f:
            annotations = json.load(f)
        self.valid_ids = {ann["id"] for ann in annotations}

    def __iter__(self):
        with tarfile.open(self.tar_path, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                video_id = os.path.splitext(os.path.basename(member.name))[0]
                if video_id not in self.valid_ids:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    frames = _sample_frames(f.read(), self.num_frames)
                    yield frames, 0
                except Exception:
                    continue
