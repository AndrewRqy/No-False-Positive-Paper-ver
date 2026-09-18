# Amplification dose sweep

Paper: tab:repair_dose

Scale flagged vs control features by alpha and measure all-class SSv2 accuracy across the alpha grid.

```
python -m causal_analysis.common.steer_expansion --family videomae --stage repair_addons --alphas 1.5 2 3 4 5 6 8 10
# VideoMAE-native equivalent:
python -m causal_analysis.exp02_amplification_dose.steer_repair_addons
```
Output: `<fam>_repair_addons.json`. Needs the SAE, NFP flags `sae_nfp_v2.pt`, and
`<fam>_erasure.json` (defines the feature-dependent classes).
