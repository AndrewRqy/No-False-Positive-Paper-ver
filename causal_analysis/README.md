# causal_analysis

The paper's causal experiments. `common/` holds the shared engine
(`steer_expansion.py` cross-model battery, plus the `steer_ssv2_logits` SteerLayer
primitive, `steer_pair_screen` pair defs, `steer_span_erasure` projector). Each
`expNN_*/` folder is one paper experiment with its own driver / inputs / README.

Run everything as modules from the repo root, e.g.
`python -m causal_analysis.common.steer_expansion --family videomae --stage erasure_allclass`.

Common inputs: the SAE checkpoint, NFP flags `sae_nfp_v2.pt` (from
`nfp_testing/save_nfp_stats_v2.py`), and SSv2 videos + `validation.json`. Per-model
checkpoint paths live in the `FAMILY` dict in `common/steer_expansion.py`.

| folder | paper | what |
|---|---|---|
| exp01_reconstruction_faithfulness | tab:recon | reconstruction-only baseline |
| exp02_amplification_dose | tab:repair_dose | amplification alpha sweep |
| exp03_amplification_alpha3 | tab:repair | alpha=3, four views (slice of exp02/amp_allclass) |
| exp04_span_erasure_allclass | tab:erasure | all-class span erasure |
| exp05_erasure_size_sweep | tab:erasure_size | top-k span-size dose |
| exp06_flip_handpicked_pairs | tab:flippers | per-feature flipping, hand-picked pairs |
| exp07_flip_random_pairs | tab:flippers_rnd | flipping, random pairs (control) |
| exp08_vjepa2_layer11 | app:depth | V-JEPA2 layer-11 SAE causal |
| exp09_vjepa2_probe | app:depth | V-JEPA2 probe-layer SAE causal |
| exp10_vjepa2_alt_erasure | tab:vj2_alt_erasure | aggregate of exp08+exp09 erasure |
| exp11_raw_dictionary | app:depth | raw dims as identity dictionary |
| exp12_per_class_erasure | tab:erasure_perclass | per-class erasure detail |
| exp13_pair_screen_detail | tab:pairdetail_vm | full flip-rate matrix + controls |
| exp14_patching_restore | tab:recovery | transplant + restoration (Patching) |
| binary_discrimination_erasure | (not in paper) | supplementary, kept per request |
| pair_catalogs | -- | enumerators + pair JSONs |
| support | -- | VideoMAE-base linear probe + head-transplant eval |
