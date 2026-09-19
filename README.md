# Identifying and Steering Temporal Features in a VideoMAE Sparse Autoencoder

<!-- agent-summary: Reproduction code for a paper on identifying and steering temporal features in video-model SAEs. Pipeline: dataset_creation (Kubric ball videos + NFP stimulus designs), sae_training (activation extraction + SAE/PCA/ICA), nfp_testing (the NFP identification test + controls + synthetic control + monosemanticity), causal_analysis (14 paper experiments: amplification, span erasure, flipping, patching). Models: VideoMAE, V-JEPA2, TimeSformer, DINOv2. Run modules with `python -m`. Datasets and checkpoints hosted on HuggingFace. -->

**No False Positive** is the reproduction code for our study of temporal features
in video sparse autoencoders. It identifies the SAE features that track a motion
concept (speed, velocity, acceleration, direction) with the Next-Frame-Prediction
(NFP) covariance test, then verifies those features causally through amplification,
span erasure, and activation patching, across VideoMAE, V-JEPA2, TimeSformer, and
a DINOv2 negative control.

This repository ships execution files only. Rendered datasets and trained
checkpoints are hosted externally (see [External resources](#external-resources)).

## Repository layout

| Directory | Contents |
| --- | --- |
| `dataset_creation/` | Kubric ball-video generator and the NFP stimulus designs (decorrelated v2, correlated v3) |
| `sae_training/` | Activation extraction and SAE / PCA / ICA training |
| `nfp_testing/` | The NFP identification test, its controls, synthetic control, and monosemanticity |
| `causal_analysis/` | The paper's causal experiments (`common/` engine plus one folder per experiment) |
| `utils/` | Model wrappers (`models/`), datasets, and shared helpers |
| `dictionary_learning/` | Vendored SAE library (third-party, unmodified in style) |
| `configs/` | Per-table run configs (`*.yaml`) consumed by the entry points |
| `tests/` | Wrapper import smoke test |

## Requirements

- Python 3.10+ and PyTorch 2.1.2 (CUDA 12.1 build)
- A HuggingFace account for the pretrained models and the SSv2 dataset
- Docker Desktop with the `kubricdockerhub/kubruntu` image, for rendering stimuli

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt            # Windows / RTX 50xx: requirements-win.txt
python -m tests.test_wrappers              # verify every model wrapper imports
```

V-JEPA2 needs a newer `transformers` than VideoMAE and TimeSformer; install
`requirements-vjepa2.txt` in a separate environment for V-JEPA2 runs.

## Quick start

Every entry point is a package module, run from the repository root:

```bash
# NFP identification on the VideoMAE SAE at layer 11
python -m nfp_testing.nfp_test --config configs/nfp_videomae_l11.yaml

# A causal span-erasure experiment
python -m causal_analysis.common.steer_expansion --family videomae --stage erasure_allclass
```

A config file fills in argument defaults, and an explicit CLI flag overrides it.
Every saved NFP or causal result embeds a `provenance` block (git commit, resolved
model, checkpoint SHA-256, package versions, and args) so a reported number traces
to the exact run that produced it.

## External resources

Not shipped in this repository; download separately.

**Models** (HuggingFace):

| Role | Identifier |
| --- | --- |
| VideoMAE, finetuned on SSv2 (primary) | `MCG-NJU/videomae-base-finetuned-ssv2` |
| VideoMAE, before label finetuning | `MCG-NJU/videomae-base-ssv2` |
| V-JEPA2 | `facebook/vjepa2-vitl-fpc16-256-ssv2` |
| TimeSformer | `facebook/timesformer-base-finetuned-ssv2` |
| DINOv2 (negative control) | `facebook/dinov2-base` |

**Datasets:**

- Something-Something-v2 videos and labels: the scripts expect a video directory and a `.../labels/validation.json`.
- Rendered NFP ball videos and SAE / flag checkpoints: HuggingFace `AndrewRqy/temporal-sae-videomae`. The v2 and v3 stimulus sets render locally from the specs in `dataset_creation/specs/`.

## Reproducing the experiments

Each table below gives the command and the file (or files) responsible for the
experiment, for review. Shared causal inputs are the SAE checkpoint, the NFP flag
file `sae_nfp_v2.pt` (written by `nfp_testing/save_nfp_stats_v2.py`), and the SSv2
videos with `validation.json`. Per-model checkpoint paths live in the `FAMILY`
dictionary in `causal_analysis/common/steer_expansion.py`.

### Dataset creation

| Step | Command | Responsible files |
| --- | --- | --- |
| Decorrelated (v2) stimulus design | `python -m dataset_creation.design_decorrelated_stimulus --joint --N 3000` | `dataset_creation/design_decorrelated_stimulus.py` |
| Correlated (v3) stimulus design | `python -m dataset_creation.design_correlated_stimulus --N 3000` | `dataset_creation/design_correlated_stimulus.py` |
| Render a spec to videos | `dataset_creation/render/run_nfp_v3_dataset.ps1 -Variant S` | `dataset_creation/render/*.ps1`, `dataset_creation/nfp_ball_dataset.py` |

### SAE training

| Step | Command | Responsible files |
| --- | --- | --- |
| Cache activations | `python -m sae_training.save_activations ...` | `sae_training/save_activations.py`, `sae_training/extract_activations.py`, `sae_training/extract_dino_patch_activations.py` |
| Train the SAE | `python -m sae_training.train_sae ...` | `sae_training/train_sae.py` |
| PCA / ICA baselines | `python -m sae_training.fit_pca_ica ...` | `sae_training/fit_pca_ica.py` |

### NFP identification (`nfp_testing/`)

| Experiment | Paper | Command | Responsible files |
| --- | --- | --- | --- |
| NFP test and depth sweep | `tab:master`, depth ablations | `python -m nfp_testing.nfp_test --model_family videomae --layer 11` | `nfp_testing/nfp_test.py` |
| VideoMAE before finetuning | `tab:vm_pretrain` | `python -m nfp_testing.nfp_test --model_name MCG-NJU/videomae-base-ssv2` | `nfp_testing/nfp_test.py` |
| Dictionary-type comparison | `tab:dictionaries` | `python -m nfp_testing.nfp_on_dict ...` | `nfp_testing/nfp_on_dict.py`, `nfp_testing/dump_ball_raw_acts.py` |
| DINOv2 negative control | `tab:master` | `python -m nfp_testing.nfp_test_dino_patch ...` | `nfp_testing/nfp_test_dino_patch.py` |
| Strongly entangled set | `app:entangled` | `python -m nfp_testing.nfp_test --dataset_dir <v3-S> ...` | `nfp_testing/nfp_test.py`, `dataset_creation/design_correlated_stimulus.py` |
| Build the v2 flag file | — | `python -m nfp_testing.save_nfp_stats_v2 ...` | `nfp_testing/save_nfp_stats_v2.py` |
| t-score distribution figure | `fig:tdist` | `python -m nfp_testing.plot_tscore_dist` | `nfp_testing/plot_tscore_dist.py` |
| Monosemanticity score | `tab:ms` | `python -m nfp_testing.ms.metric ...` | `nfp_testing/ms/metric.py`, `nfp_testing/ms/encode_dino_sae_videos.py` |
| Projection-fraction control | `tab:projfrac` | `python -m nfp_testing.synthetic_control.projfrac_sweep` | `nfp_testing/synthetic_control/projfrac_sweep.py` |
| Synthetic positive control | `tab:synth_*` | `python -m nfp_testing.synthetic_control.nfp_test_synthetic` | `nfp_testing/synthetic_control/gen_synthetic_activations.py`, `nfp_testing/synthetic_control/nfp_test_synthetic.py` |

### Causal analysis (`causal_analysis/`)

The cross-model battery `causal_analysis/common/steer_expansion.py` produces most
causal results through `--family` and `--stage`. It builds on the shared primitives
`steer_ssv2_logits.py` (the `SteerLayer` intervention), `steer_pair_screen.py` (pair
definitions), and `steer_span_erasure.py` (the span projector), all in `common/`.

| Experiment | Paper | Command | Responsible files |
| --- | --- | --- | --- |
| exp01 reconstruction faithfulness | `tab:recon` | `steer_expansion --family <fam> --stage recon_acc` | `common/steer_expansion.py` |
| exp02 amplification dose | `tab:repair_dose` | `steer_expansion --family videomae --stage repair_addons` | `common/steer_expansion.py`, `exp02_amplification_dose/steer_repair_addons.py` |
| exp03 amplification at alpha=3 | `tab:repair` | `steer_expansion --family videomae --stage amp_allclass` | `common/steer_expansion.py` |
| exp04 span erasure, all-class | `tab:erasure` | `steer_expansion --family videomae --stage erasure_allclass` | `common/steer_expansion.py`, `common/steer_span_erasure.py` |
| exp05 top-k erasure size sweep | `tab:erasure_size` | `python -m causal_analysis.exp05_erasure_size_sweep.steer_erasure_size_sweep --ks 12 25 50 109` | `exp05_erasure_size_sweep/steer_erasure_size_sweep.py` |
| exp06 flipping, hand-picked pairs | `tab:flippers` | `python -m causal_analysis.exp06_flip_handpicked_pairs.steer_direction_flip` | `exp06_flip_handpicked_pairs/steer_direction_flip.py` |
| exp07 flipping, random pairs | `tab:flippers_rnd` | `gen_random_pairs`, then `steer_expansion --stage pairs --pairs_json ...` | `exp07_flip_random_pairs/gen_random_pairs.py`, `common/steer_expansion.py` |
| exp08 V-JEPA2 layer-11 SAE | `app:depth` | `steer_expansion --family vjepa2_l11 --stage {cache,amp_allclass,erasure_allclass,pairs}` | `common/steer_expansion.py` |
| exp09 V-JEPA2 probe SAE | `app:depth` | `steer_expansion --family vjepa2_probe --stage ...` | `common/steer_expansion.py` |
| exp10 alt-V-JEPA2 erasure table | `tab:vj2_alt_erasure` | aggregate of exp08 and exp09 `erasure_allclass` outputs | `common/steer_expansion.py` |
| exp11 raw dims as dictionary | `app:depth` | `steer_expansion --family vjepa2_raw --stage erasure_allclass` | `common/steer_expansion.py` |
| exp12 per-class erasure detail | `tab:erasure_perclass` | `python -m causal_analysis.exp12_per_class_erasure.steer_erasure_fill_cells` | `exp12_per_class_erasure/steer_erasure_fill_cells.py` |
| exp13 full pair-screen detail | `tab:pairdetail_vm` | `common.steer_pair_screen`, then `exp13_pair_screen_detail.steer_pair_controls` | `common/steer_pair_screen.py`, `exp13_pair_screen_detail/steer_pair_controls.py` |
| exp14 patching and restoration | `tab:recovery` | drivers in `exp14_patching_restore/` | `exp14_patching_restore/steer_transplant.py`, `steer_shuffle_restore.py`, `steer_reverse_play.py`, `steer_global_restore.py`, `controls_*.py` |
| binary-discrimination erasure | (not in paper) | `python -m causal_analysis.binary_discrimination_erasure.pair_binary_erasure --family videomae` | `binary_discrimination_erasure/pair_binary_erasure.py` |

## Notes

Working notes and exploratory scripts are kept in the private development
repository; this release contains only what the paper and appendix report, plus
the NFP-v3 stimulus design and the binary-discrimination erasure.
