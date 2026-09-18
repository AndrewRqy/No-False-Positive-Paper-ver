# Identifying and Steering Temporal Features in a VideoMAE Sparse Autoencoder

Code to reproduce the paper. Execution files only: rendered datasets and trained
checkpoints are hosted externally (see [External resources](#external-resources)).

---

## Architecture

```
sae-for-vlm-release/
  dataset_creation/     Kubric ball-video generator + NFP stimulus designs (v2, v3) + render scripts
  sae_training/         activation extraction + SAE / PCA / ICA training
  nfp_testing/          the NFP identification test, controls, synthetic control, monosemanticity
      ms/  sweeps/  synthetic_control/
  causal_analysis/      the paper's causal experiments
      common/           shared engine: steer_expansion (cross-model battery) + SteerLayer / pair /
                        span-erasure primitives, imported by every experiment
      expNN_*/          one folder per experiment: driver(s) + inputs + a README
      pair_catalogs/    class-pair enumerators + JSONs        support/  base-model probe
  utils/                model wrappers (models/), datasets/, helpers, sweep_common
  dictionary_learning/  vendored SAE library
```

**How to run.** Everything is a package; run from this directory with `-m`:

```
python -m nfp_testing.nfp_test --model_family videomae --layer 11
python -m causal_analysis.common.steer_expansion --family videomae --stage erasure_allclass
```

Common causal inputs: the SAE checkpoint, the NFP flag file `sae_nfp_v2.pt`
(produced by `nfp_testing/save_nfp_stats_v2.py`), and the SSv2 videos +
`validation.json`. Per-model checkpoint paths are in the `FAMILY` dict in
`causal_analysis/common/steer_expansion.py`.

## Setup

```
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # Windows / RTX 50xx: see requirements-win.txt
python -m tests.test_wrappers            # verify all model wrappers import
```

V-JEPA2 needs a newer transformers than VideoMAE/TimeSformer; use a separate env
from `requirements-vjepa2.txt` for V-JEPA2 runs.

**Configs.** Each reported table can be run from a checked-in config instead of
long flag lists (`configs/*.yaml`); an explicit CLI flag overrides the file:

```
python -m nfp_testing.nfp_test --config configs/nfp_videomae_l11.yaml
```

Every saved NFP/causal result embeds a `provenance` block (git commit, resolved
model, checkpoint SHA-256, package versions, args) so a table traces to the exact
run. `--model_name` may be omitted; it resolves to the family default.

## External resources

Not shipped in this repo; download separately.

**Models** (HuggingFace):
| role | id |
|---|---|
| VideoMAE, finetuned on SSv2 (primary) | `MCG-NJU/videomae-base-finetuned-ssv2` |
| VideoMAE, before label finetuning | `MCG-NJU/videomae-base-ssv2` |
| V-JEPA2 | `facebook/vjepa2-vitl-fpc16-256-ssv2` |
| TimeSformer | `facebook/timesformer-base-finetuned-ssv2` |
| DINOv2 (negative control) | `facebook/dinov2-base` |

**Datasets:**
- Something-Something-v2 (SSv2) videos + label json: https://www.qualcomm.com/developer/software/something-something-v-2-dataset (registration required). Scripts expect a video directory and `.../labels/validation.json`.
- Rendered NFP ball videos (v1) + SAE / flag checkpoints: HuggingFace `AndrewRqy/temporal-sae-videomae`. The v2/v3 sets are rendered locally from the specs in `dataset_creation/specs/`.

---

## Reproducing each experiment

### 1. Dataset creation (`dataset_creation/`)
| step | requires | command |
|---|---|---|
| Design the decorrelated (v2) stimulus | — | `python -m dataset_creation.design_decorrelated_stimulus --joint --N 3000` |
| Design the correlated (v3) stimulus | — | `python -m dataset_creation.design_correlated_stimulus --N 3000` |
| Render a spec to videos | Docker Desktop + `kubricdockerhub/kubruntu` image, a spec from `specs/` | `dataset_creation/render/run_nfp_v3_dataset.ps1 -Variant S` |

### 2. SAE training (`sae_training/`)
| step | requires | command |
|---|---|---|
| Cache activations | a model id, SSv2 (or ball) videos | `python -m sae_training.save_activations ...` (video: `sae_training.extract_activations`; DINO patches: `sae_training.extract_dino_patch_activations`) |
| Train the SAE | cached activations | `python -m sae_training.train_sae ...` (hyperparameters: paper `tab:sae_params`) |
| PCA / ICA baselines | cached activations | `python -m sae_training.fit_pca_ica ...` |

### 3. NFP identification (`nfp_testing/`)
| experiment | paper | requires | command |
|---|---|---|---|
| NFP test / depth sweep (all models) | `tab:raw_layers`, `tab:master`, depth ablations | model + SAE, rendered NFP videos | `python -m nfp_testing.nfp_test --model_family {videomae,timesformer,vjepa2} --layer L` |
| VideoMAE before finetuning | `tab:vm_pretrain` | `MCG-NJU/videomae-base-ssv2` | `python -m nfp_testing.nfp_test --model_name MCG-NJU/videomae-base-ssv2` |
| Dictionary-type comparison (SAE/BatchTopK/PCA/ICA/raw) | `tab:dictionaries` | cached raw acts + dicts | `python -m nfp_testing.nfp_on_dict ...` (cache first: `nfp_testing.dump_ball_raw_acts`) |
| DINOv2 negative control | `tab:master` | DINOv2 SAE | `python -m nfp_testing.nfp_test_dino_patch ...` |
| Build the v2 flag file | SAE + raw acts | `python -m nfp_testing.save_nfp_stats_v2 ...` (writes `sae_nfp_v2.pt`) |
| t-score distribution figure | `fig:tdist` | per-model SAEs | `python -m nfp_testing.plot_tscore_dist` |
| Monosemanticity score | `tab:ms` | DINOv2 embeddings of SSv2-val | `python -m nfp_testing.ms.metric` (encode with `nfp_testing.ms.encode_dino_sae_videos`) |
| Projection-fraction control | `tab:projfrac` | — | `python -m nfp_testing.synthetic_control.projfrac_sweep` |
| Synthetic positive control (+ oblique) | `tab:synth_*`, `app:synth_oblique` | — | `python -m nfp_testing.synthetic_control.gen_synthetic_activations` then `nfp_testing.synthetic_control.nfp_test_synthetic` |

### 4. Causal analysis (`causal_analysis/`)
Each folder's `README.md` has the full command, inputs, and the table it produces.

| experiment | paper | reproduce (see folder README) |
|---|---|---|
| exp01 reconstruction faithfulness | `tab:recon` | `steer_expansion --family <fam> --stage recon_acc` |
| exp02 amplification dose | `tab:repair_dose` | `steer_expansion --family videomae --stage repair_addons` |
| exp03 amplification α=3, four views | `tab:repair` | `steer_expansion --family videomae --stage amp_allclass` (slice α=3) |
| exp04 span erasure, all-class | `tab:erasure` | `steer_expansion --family videomae --stage erasure_allclass` |
| exp05 top-k erasure size sweep | `tab:erasure_size` | `exp05_erasure_size_sweep.steer_erasure_size_sweep --ks 12 25 50 109` |
| exp06 flipping, hand-picked pairs | `tab:flippers` | `exp06_flip_handpicked_pairs.steer_direction_flip` |
| exp07 flipping, random pairs | `tab:flippers_rnd` | `gen_random_pairs` then `steer_expansion --stage pairs --pairs_json ...` |
| exp08 V-JEPA2 layer-11 SAE | `app:depth` | `steer_expansion --family vjepa2_l11 --stage {cache,amp_allclass,erasure_allclass,pairs}` |
| exp09 V-JEPA2 probe SAE | `app:depth` | `steer_expansion --family vjepa2_probe --stage ...` |
| exp10 alt-V-JEPA2 erasure table | `tab:vj2_alt_erasure` | aggregate of exp08 + exp09 `erasure_allclass` |
| exp11 raw dims as dictionary | `app:depth` | `steer_expansion --family vjepa2_raw[_l11] --stage erasure_allclass` |
| exp12 per-class erasure detail | `tab:erasure_perclass` | `exp12_per_class_erasure.steer_erasure_fill_cells` |
| exp13 full pair-screen detail | `tab:pairdetail_vm` | `common.steer_pair_screen` + `exp13_pair_screen_detail.steer_pair_controls` |
| exp14 patching + restoration | `tab:recovery` | `steer_transplant`, `steer_shuffle_restore`, `steer_reverse_play` (+ controls) in `exp14_patching_restore` |
| binary-discrimination erasure | (not in paper) | `binary_discrimination_erasure.pair_binary_erasure --family videomae` |

---

Working notes and exploratory scripts are kept in the private development repo; this
release contains only what the paper and appendix report (plus the NFP-v3 design and
the binary-discrimination erasure).
