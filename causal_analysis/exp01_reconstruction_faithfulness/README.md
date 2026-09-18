# Reconstruction-only faithfulness

Paper: tab:recon

Replace every token by its SAE reconstruction with NO error re-add, and measure accuracy.
Establishes the re-add-error baseline every other causal experiment uses.

```
python -m causal_analysis.common.steer_expansion --family videomae      --stage recon_acc
python -m causal_analysis.common.steer_expansion --family videomae_base --stage recon_acc
python -m causal_analysis.common.steer_expansion --family timesformer_l10 --stage recon_acc
python -m causal_analysis.common.steer_expansion --family vjepa2         --stage recon_acc
```
Output: `<fam>_recon_acc.json` (baseline vs full-reconstruction accuracy). The
`*_noreadd` variants elsewhere are this same no-re-add mode at the intervention step.
