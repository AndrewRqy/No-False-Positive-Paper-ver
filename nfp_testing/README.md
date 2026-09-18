# nfp_testing

The Next-Frame-Prediction (NFP) temporal-feature IDENTIFICATION test and its controls.

- `nfp_test.py` -- core driver: within-video-covariance one-sample t-test, Bonferroni.
  `--model_family {videomae,timesformer,vjepa2} --layer L`, with SAE / PCA / ICA / Identity dicts.
  Depth sweeps, dictionary-type comparison, and VideoMAE-before-finetuning all run through this.
- `nfp_on_dict.py`, `save_nfp_stats_v2.py`, `nfp_v2_analysis.py`, `nfp_analyze.py` -- dictionary
  comparison, flag-file creation (`sae_nfp_v2.pt`), decorrelated-vs-entangled analysis.
- `nfp_test_dino_patch.py` -- DINOv2 spatial-patch negative control.
- `plot_tscore_dist.py`, `cutoff_distribution.py` -- the t-score distribution figure (fig:tdist).
- `ms/` -- appearance-based monosemanticity score across SAE/PCA/ICA/raw (tab:ms).
- `sweeps/` -- PCA/ICA and DINOv2 dictionary-dimension sweeps.
- `synthetic_control/` -- projection-fraction control, synthetic positive control, oblique variant.
