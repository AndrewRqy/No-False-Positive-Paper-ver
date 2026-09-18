# Individual-feature flipping, hand-picked pairs

Paper: tab:flippers, tab:flipper_counts

Amplify each identified feature individually (clamp +/-150) on 10 hand-picked SSv2
class pairs (7 motion + 3 depth controls); count top-1 flips to the paired class.

```
python -m causal_analysis.exp06_flip_handpicked_pairs.steer_direction_flip
# or via the cross-model battery (default PAIRS10):
python -m causal_analysis.common.steer_expansion --family videomae --stage pairs
```
Pair definitions: `tab:pairs` in the paper. Full per-(feature,pair) detail is exp13.
