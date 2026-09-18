# Span erasure, all-class

Paper: tab:erasure

Project the decoder span of the identified features out of the activations
(x -> x - QQ^T x, re-adding reconstruction error), vs random / static /
activation-matched control sets, over all 174 classes.

```
python -m causal_analysis.common.steer_expansion --family videomae --stage erasure_allclass
# VideoMAE-native (Experiment C4):
python -m causal_analysis.common.steer_span_erasure
```
Output: `<fam>_erasure_allclass.json`. Needs SAE, `sae_nfp_v2.pt`, SSv2.
