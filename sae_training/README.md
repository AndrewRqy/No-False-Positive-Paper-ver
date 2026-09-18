# sae_training

Activation extraction and SAE training (the vendored library is `dictionary_learning/`).

- `save_activations.py`, `extract_activations.py`, `extract_dino_patch_activations.py` -- cache model
  activations (video / DINOv2 patch) for SAE training.
- `train_sae.py` -- trains the SAE (standard L1, TopK, BatchTopK, JumpReLU, Matryoshka).
- `fit_pca_ica.py` -- PCA / ICA dictionaries used as monosemanticity + NFP baselines.

Hyperparameters are in the paper appendix (tab:sae_params).
