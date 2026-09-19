"""Run-provenance metadata for reproducible result files (review item 3).

`run_metadata(args, **extra)` returns a JSON-serializable dict recording exactly
how a result was produced: parsed args, resolved model, checkpoint hashes, git
commit, package versions, seed, and anything passed in `extra`. Attach it under a
`"provenance"` key in every saved NFP / causal result so a table can be traced to
the exact repository state and command, not guessed from metrics alone.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "einops",
    "nnsight",
]


def sha256_file(path, chunk=1 << 20):
    """Hex SHA-256 of a file, or None if the path is missing/unreadable."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def git_commit():
    """Current commit hash of the repo, with a '-dirty' suffix if the tree has
    uncommitted changes; None outside a git checkout."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        if rev.returncode != 0:
            return None
        commit = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions(names=_TRACKED_PACKAGES):
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def run_metadata(args=None, checkpoints=None, **extra):
    """Build the provenance dict.

    args         : the argparse.Namespace (stored as a plain dict).
    checkpoints  : {name: path} whose SHA-256 is recorded under `<name>_sha256`.
    extra        : any additional key/values (model_name, model_revision,
                   dataset_size, video_ids, seed, ...).
    """
    meta = {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "package_versions": package_versions(),
        "command": " ".join(sys.argv),
    }
    if args is not None:
        meta["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    for name, path in (checkpoints or {}).items():
        meta[f"{name}_path"] = str(path) if path is not None else None
        meta[f"{name}_sha256"] = sha256_file(path)
    meta.update(extra)
    return meta
