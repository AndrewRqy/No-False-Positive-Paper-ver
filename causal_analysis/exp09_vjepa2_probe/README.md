# V-JEPA2 probe-layer SAE causal

Paper: app:depth, tab:vj2_alt_erasure

Same battery on the SAE trained on the attentive-probe last self-attention output.

```
python -m causal_analysis.common.steer_expansion --family vjepa2_probe --stage cache
python -m causal_analysis.common.steer_expansion --family vjepa2_probe --stage amp_allclass
python -m causal_analysis.common.steer_expansion --family vjepa2_probe --stage erasure_allclass
python -m causal_analysis.common.steer_expansion --family vjepa2_probe --stage pairs
```
See FAMILY['vjepa2_probe'] (layer=2, attach='pooler').
