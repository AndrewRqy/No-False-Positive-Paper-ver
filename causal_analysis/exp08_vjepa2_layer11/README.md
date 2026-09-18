# V-JEPA2 layer-11 SAE causal

Paper: app:depth, tab:vj2_alt_erasure

Full causal battery on a V-JEPA2 layer-11 SAE (amplification + span erasure + pair screen).

```
python -m causal_analysis.common.steer_expansion --family vjepa2_l11 --stage cache
python -m causal_analysis.common.steer_expansion --family vjepa2_l11 --stage amp_allclass
python -m causal_analysis.common.steer_expansion --family vjepa2_l11 --stage erasure_allclass
python -m causal_analysis.common.steer_expansion --family vjepa2_l11 --stage pairs
```
Checkpoint paths for this family are in `causal_analysis/common/steer_expansion.py` (FAMILY['vjepa2_l11']).
