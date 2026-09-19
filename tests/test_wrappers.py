"""Smoke test: every advertised model wrapper imports and (optionally) instantiates.

Import-only by default (fast, no network) -- this is what catches the V-JEPA2 /
transformers version mismatch (review item 2). Pass --instantiate to also build
each model from its default HuggingFace checkpoint (downloads weights).

    python -m tests.test_wrappers
    python -m tests.test_wrappers --instantiate --device cpu
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (module, class, default checkpoint)
WRAPPERS = [
    ("utils.models.videomae", "VideoMAE", "MCG-NJU/videomae-base-finetuned-ssv2"),
    ("utils.models.timesformer", "Timesformer", "facebook/timesformer-base-finetuned-ssv2"),
    ("utils.models.vjepa2", "VJEPA2", "facebook/vjepa2-vitl-fpc16-256-ssv2"),
    ("utils.models.dino", "Dino", "facebook/dinov2-base"),
]


def check_imports():
    results = {}
    for mod, cls, _ in WRAPPERS:
        try:
            m = importlib.import_module(mod)
            assert hasattr(m, cls), f"{mod} has no class {cls}"
            results[cls] = "import OK"
        except Exception as e:  # noqa: BLE001 -- report, don't abort other wrappers
            results[cls] = f"IMPORT FAILED: {type(e).__name__}: {e}"
    return results


def check_instantiate(device):
    results = {}
    for mod, cls, ckpt in WRAPPERS:
        try:
            m = importlib.import_module(mod)
            getattr(m, cls)(ckpt, device)
            results[cls] = f"instantiate OK ({ckpt})"
        except Exception as e:  # noqa: BLE001
            results[cls] = f"INSTANTIATE FAILED: {type(e).__name__}: {e}"
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instantiate", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    results = check_imports()
    if args.instantiate:
        results = {
            **results,
            **{f"{k} (instantiate)": v for k, v in check_instantiate(args.device).items()},
        }

    ok = True
    for name, status in results.items():
        flag = "ok " if "OK" in status else "ERR"
        if "OK" not in status:
            ok = False
        print(f"[{flag}] {name}: {status}")
    print("\n" + ("ALL WRAPPERS OK" if ok else "SOME WRAPPERS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
