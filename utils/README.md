# utils

Cross-section shared code (base layer, imports nothing from the experiment folders).

- `models/` -- encoder wrappers: videomae, vjepa2, timesformer, dino (+clip, siglip).
- `datasets/` -- ssv2 video dataset, chunked activation dataset.
- `utils.py` -- get_model / get_dataset / collate helpers, IdentitySAE.
- `sweep_common.py` -- dependency-light PCA/ICA sweep helpers shared by nfp_testing and synthetic_control.
