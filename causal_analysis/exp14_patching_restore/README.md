# Activation patching: transplant + restoration

Paper: tab:recovery

Fraction-recovered (log-odds) causal experiments using the Patching primitive.

```
# transplant donor activations between opposite classes (camera L/R and U/D):
python -m causal_analysis.exp14_patching_restore.steer_transplant
# destroy motion by frame shuffle, patch clean feature values back:
python -m causal_analysis.exp14_patching_restore.steer_shuffle_restore
python -m causal_analysis.exp14_patching_restore.steer_global_restore    # all-class variant
# destroy motion by temporal reversal, patch back (+ by-tag-group breakdown):
python -m causal_analysis.exp14_patching_restore.steer_reverse_play
python -m causal_analysis.exp14_patching_restore.controls_direction_nulls
python -m causal_analysis.exp14_patching_restore.controls_specificity    # shared controls
```
Needs SAE, `sae_nfp_v2.pt`, the pair-screen output `expB2_pair_screen.json`, SSv2.
