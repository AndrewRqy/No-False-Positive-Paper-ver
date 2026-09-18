"""Self-describing SAE checkpoint load/save (review item 5).

Training scripts write a sidecar `<checkpoint>.meta.json` recording the
architecture (`sae_type`, dims, and the model/layer/attachment the SAE was
trained on). `load_sae` reads it and instantiates the correct dictionary class
automatically; if the metadata is missing it falls back to the caller-supplied
`sae_type` and, on a state-dict mismatch, raises a clear architecture-mismatch
error instead of an opaque missing-parameter exception.
"""
import json
from pathlib import Path

_META_KEYS = ("sae_type", "activation_dim", "dict_size", "expansion_factor",
              "model_name", "layer", "attachment_point")


def meta_path(ckpt_path):
    return Path(str(ckpt_path) + ".meta.json")


def save_sae_meta(ckpt_path, sae_type, activation_dim=None, dict_size=None,
                  expansion_factor=None, model_name=None, layer=None,
                  attachment_point=None, **extra):
    meta = {"sae_type": sae_type, "activation_dim": activation_dim,
            "dict_size": dict_size, "expansion_factor": expansion_factor,
            "model_name": model_name, "layer": layer,
            "attachment_point": attachment_point, **extra}
    p = meta_path(ckpt_path)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def read_sae_meta(ckpt_path):
    p = meta_path(ckpt_path)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def load_sae(path, sae_type=None, device="cpu", mode="sign_split", **kw):
    """Load a dictionary, dispatching on checkpoint metadata (falling back to
    `sae_type`). Raises a clear error on architecture mismatch.

    sae_type in {standard, jumprelu, topk, batch_top_k, pca, ica, identity}.
    """
    meta = read_sae_meta(path)
    resolved = (meta or {}).get("sae_type") or sae_type or "standard"

    def _load():
        if resolved in ("pca", "ica"):
            from dictionary_learning import PCADict, ICADict
            cls = PCADict if resolved == "pca" else ICADict
            return cls.from_pretrained(path, device=device, mode=mode)
        if resolved == "identity":
            from dictionary_learning import IdentityDict
            return IdentityDict.from_pretrained(None)
        if resolved == "jumprelu":
            from dictionary_learning import JumpReluAutoEncoder
            return JumpReluAutoEncoder.from_pretrained(path, device=device, **kw)
        if resolved in ("topk", "batch_top_k", "batchtopk"):
            try:
                from dictionary_learning.dictionary import AutoEncoderTopK
            except ImportError as e:
                raise ImportError(
                    f"checkpoint metadata says sae_type='{resolved}' but the "
                    f"TopK autoencoder class is unavailable in this build") from e
            return AutoEncoderTopK.from_pretrained(path, device=device, **kw)
        # standard L1 ReLU SAE
        from dictionary_learning import AutoEncoder
        kw.setdefault("normalize_decoder", False)
        return AutoEncoder.from_pretrained(path, device=device, **kw)

    try:
        return _load()
    except (RuntimeError, KeyError, TypeError) as e:
        hint = ("checkpoint has no sidecar metadata; pass the correct sae_type"
                if meta is None else f"sidecar metadata says sae_type='{resolved}'")
        raise RuntimeError(
            f"Failed to load SAE '{path}' as architecture '{resolved}'. "
            f"This usually means an architecture mismatch ({hint}). "
            f"Underlying error: {type(e).__name__}: {e}") from e
