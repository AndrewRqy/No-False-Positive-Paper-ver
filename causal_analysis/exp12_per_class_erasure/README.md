# Per-class span-erasure detail

Paper: tab:erasure_perclass

Per-class accuracy under identified vs random(seed 101) erasure (VideoMAE-ft 13
classes, V-JEPA2 5 classes), plus missing static / activation-matched control spans.

```
python -m causal_analysis.exp12_per_class_erasure.steer_erasure_fill_cells
```
Needs SAE, `sae_nfp_v2.pt`, `videomae_erasure.json`, `<fam>_probe_cache.pt` (activation-matched), SSv2.
