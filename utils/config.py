"""Config-file support for the experiment entry points (review item 10).

A config is a flat YAML (or JSON) mapping of argparse dest -> value. Call
`add_config_arg(parser)` to add `--config`, then `apply_config(parser, args)`
right after `parse_args()`: values from the file fill in any argument the user
did NOT pass on the command line, so an explicit CLI flag always wins.

This lets every reported table ship as one checked-in, fully-specified config
(configs/*.yaml) run with, e.g.:

    python -m nfp_testing.nfp_test --config configs/nfp_videomae_l11.yaml
"""
import argparse
import json
import sys
from pathlib import Path


def load_config(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pyyaml is required for YAML configs "
                              "(`pip install pyyaml`), or use a .json config") from e
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a flat mapping, got {type(cfg)}")
    return cfg


def add_config_arg(parser):
    parser.add_argument("--config", default=None,
                        help="YAML/JSON file of argument defaults; explicit CLI "
                             "flags override it.")
    return parser


def _explicit_cli_dests(parser):
    """Dests the user actually passed on the command line (so config never
    overrides an explicit flag)."""
    seen = set()
    tokens = set(sys.argv[1:])
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in tokens or any(t.startswith(opt + "=") for t in tokens):
                seen.add(action.dest)
    return seen


def apply_config(parser, args):
    """Merge args.config into args: config fills only args left at their default
    and not given on the CLI. Returns the same namespace. Unknown keys raise."""
    if getattr(args, "config", None) is None:
        return args
    cfg = load_config(args.config)
    valid = {a.dest for a in parser._actions}
    unknown = set(cfg) - valid
    if unknown:
        raise ValueError(f"config {args.config} has unknown keys: {sorted(unknown)}")
    explicit = _explicit_cli_dests(parser)
    for key, val in cfg.items():
        if key not in explicit:
            setattr(args, key, val)
    return args
